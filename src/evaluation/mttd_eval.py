"""Measure MTTD (metric M1b) by scoring the synthetic overlay through the deployed fusion rule.

Why this exists. `synthetic_overlay.py` has carried `mttd_seconds()` and the inject timestamps
that make MTTD computable at all, but nothing ever called it, so M1b was listed as a thesis
metric and was simply unmeasured. This runner closes that: it scores every injected flow with
the three deployed unified models, applies `fusion_layer.fuse`, and turns the resulting
per-flow alerts into per-flow and per-campaign MTTD.

WHAT `detected_at` MEANS HERE, because this is the part a reader will challenge. The detector
is offline and per-flow: there is no live stream, so no alert has a wall-clock time of its own.
A flow's detection time is therefore modelled as

    detected_at = inject_timestamp + per_flow_scoring_latency

where the latency is MEASURED in this run, not assumed. Under that model the interesting
quantity is not the latency -- it is a near-constant offset of a few hundred microseconds --
but HOW FAR INTO A CAMPAIGN the first detected flow lands. A campaign whose opening flows slip
past the threshold and whose fiftieth flow trips it has a real, large MTTD, and that is what
this measures.

WHAT IS DELIBERATELY NOT IN THE LATENCY: Kafka transport, the OpenCTI enrichment hop, the RAG
retrieval, and the LLM explanation. None of those is live-verified end to end on this machine,
so folding an estimate for them in would dress a guess as a measurement. They also sit AFTER
the detection decision -- an alert exists once the fusion rule fires -- so excluding them is
the right scope for a detection-latency metric, not merely the convenient one. SHAP would also
sit after the decision and is disabled anyway.

READ THE DETECTION RATE WITH CARE. The overlay's feature distributions are hand-set from attack
mechanics, not fitted to real captures (see the overlay's own report, which states per column
which are modelled, derived, heuristic or placeholder). A high detection rate here is evidence
that the rule fires on flows shaped like this, NOT evidence of real-world detection capability.
MTTD is reported conditional on detection, and undetected campaigns are counted separately
rather than averaged away -- averaging a miss as if it were a slow detection would flatter the
system exactly where it fails.

Run from the repo root with the Anaconda interpreter:
    python -m src.evaluation.mttd_eval
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.ingestion.synthetic_overlay import OUT_PARQUET, feature_columns, mttd_seconds
from src.modeling import fusion_layer as FL

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
METRICS_PATH = ROOT / "reports/mttd_metrics.csv"
REPORT_PATH = ROOT / "reports/mttd_report.txt"
SCORES_PATH = ROOT / "reports/mttd_flow_scores.parquet"

BATCH = 65536
_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 74)
    log(title)
    log("=" * 74)


class Autoencoder(nn.Module):
    """Mirrors the architecture in autoencoder_unified_config.json."""

    def __init__(self, input_dim, hidden_dims, bottleneck_dim, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU}[activation]
        enc_dims = [input_dim] + list(hidden_dims) + [bottleneck_dim]
        dec_dims = [bottleneck_dim] + list(hidden_dims)[::-1] + [input_dim]
        layers = []
        for i in range(len(enc_dims) - 1):
            layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                layers.append(act())
        self.encoder = nn.Sequential(*layers)
        layers = []
        for i in range(len(dec_dims) - 1):
            layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                layers.append(act())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, X, device):
    out = np.empty(len(X), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.tensor(X[i:i + BATCH], dtype=torch.float32).to(device)
            out[i:i + BATCH] = torch.mean((model(xb) - xb) ** 2, dim=1).cpu().numpy()
    return out


def score_overlay(overlay: pd.DataFrame):
    """Score every injected flow with the three deployed models and the deployed fusion rule."""
    cols = feature_columns()
    missing = [c for c in cols if c not in overlay.columns]
    assert not missing, f"overlay is missing deployed feature columns: {missing[:5]}"
    X = overlay[cols].to_numpy(dtype=np.float64)

    rf = joblib.load(MODELS / "random_forest_unified.pkl")
    benign_idx = list(rf.classes_).index("BENIGN")
    t0 = time.perf_counter()
    rf_prob = 1.0 - rf.predict_proba(X)[:, benign_idx]
    rf_seconds = time.perf_counter() - t0

    iso = joblib.load(MODELS / "isolation_forest_unified.pkl")
    t0 = time.perf_counter()
    if_raw = -iso.score_samples(X)
    if_seconds = time.perf_counter() - t0

    cfg = json.loads((MODELS / "autoencoder_unified_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = Autoencoder(cfg["input_dim"], cfg["hidden_dims"], cfg["bottleneck_dim"],
                     cfg.get("activation", "relu")).to(device)
    ae.load_state_dict(torch.load(MODELS / "autoencoder_unified.pt", map_location=device))
    t0 = time.perf_counter()
    ae_raw = reconstruction_error(ae, X.astype(np.float32), device)
    ae_seconds = time.perf_counter() - t0

    fused_score, fused_pred, _, model_used = FL.fuse_with_metadata(rf_prob, if_raw, ae_raw)
    total_seconds = rf_seconds + if_seconds + ae_seconds
    latency_s = total_seconds / len(X)
    timings = {"rf_seconds": rf_seconds, "if_seconds": if_seconds, "ae_seconds": ae_seconds,
               "total_seconds": total_seconds, "per_flow_latency_s": latency_s,
               "device": str(device), "n_flows": int(len(X))}
    scored = overlay[["inject_id", "inject_timestamp", "campaign_id", "attack_class"]].copy()
    scored["fused_score"] = fused_score
    scored["alert"] = fused_pred
    scored["model_used"] = model_used
    return scored, timings


def main() -> None:
    section("MTTD (M1b) on the synthetic attack overlay")
    overlay = pd.read_parquet(ROOT / OUT_PARQUET)
    log(f"  overlay: {len(overlay):,} injected flows, "
        f"{overlay['campaign_id'].nunique()} campaigns, "
        f"{overlay['attack_class'].nunique()} attack classes")
    log(f"  deployed rule: {FL.CHOSEN_STRATEGY} "
        f"(confidence {FL.CONFIDENCE_THRESHOLD}, threshold {FL.FUSED_DECISION_THRESHOLD})")

    section("STEP 1: score every injected flow with the deployed models")
    scored, timings = score_overlay(overlay)
    log(f"  RF {timings['rf_seconds']:.2f}s   IF {timings['if_seconds']:.2f}s   "
        f"AE {timings['ae_seconds']:.2f}s ({timings['device']})")
    log(f"  measured per-flow scoring latency: {timings['per_flow_latency_s'] * 1e6:.1f} us")
    log(f"  flows alerting: {int(scored['alert'].sum()):,} / {len(scored):,} "
        f"({scored['alert'].mean() * 100:.2f}%)")
    decided = scored.loc[scored["alert"] == 1, "model_used"].value_counts().to_dict()
    log(f"  which branch decided the alerting flows: {decided}")

    section("STEP 2: turn per-flow alerts into detection times")
    log("  detected_at = inject_timestamp + measured per-flow scoring latency.")
    log("  The latency is a near-constant offset; what moves MTTD is how far into a campaign")
    log("  the first detected flow sits. Transport, enrichment, RAG and LLM time are excluded")
    log("  on purpose -- not live-verified, and all downstream of the detection decision.")
    latency = pd.Timedelta(seconds=timings["per_flow_latency_s"])
    detections = scored.loc[scored["alert"] == 1, ["inject_id", "inject_timestamp"]].copy()
    detections["detected_at"] = detections["inject_timestamp"] + latency
    detections = detections[["inject_id", "detected_at"]]

    overlay_idx = overlay.set_index(overlay.index)
    mttd_flow = mttd_seconds(overlay_idx, detections, per="flow")
    mttd_campaign = mttd_seconds(overlay_idx, detections, per="campaign")

    section("STEP 3: per attack class")
    rows = []
    for cls, grp in overlay.groupby("attack_class"):
        camp_ids = set(grp["campaign_id"].unique())
        det_flows = set(detections["inject_id"])
        flows_detected = int(grp["inject_id"].isin(det_flows).sum())
        detected_camps = set(
            grp.loc[grp["inject_id"].isin(det_flows), "campaign_id"].unique()
        )
        cls_campaign_mttd = mttd_campaign.loc[
            mttd_campaign.index.intersection(list(detected_camps))
        ] if len(mttd_campaign) else pd.Series(dtype=float)
        flow_mask = overlay["attack_class"] == cls
        cls_flow_mttd = mttd_flow.loc[mttd_flow.index.intersection(overlay.index[flow_mask])] \
            if len(mttd_flow) else pd.Series(dtype=float)
        rows.append({
            "attack_class": cls,
            "n_flows": int(len(grp)),
            "flows_detected": flows_detected,
            "flow_detection_rate": flows_detected / len(grp),
            "n_campaigns": len(camp_ids),
            "campaigns_detected": len(detected_camps),
            "campaigns_undetected": len(camp_ids) - len(detected_camps),
            "mttd_campaign_mean_s": float(cls_campaign_mttd.mean()) if len(cls_campaign_mttd) else float("nan"),
            "mttd_campaign_median_s": float(cls_campaign_mttd.median()) if len(cls_campaign_mttd) else float("nan"),
            "mttd_campaign_max_s": float(cls_campaign_mttd.max()) if len(cls_campaign_mttd) else float("nan"),
            "mttd_flow_mean_s": float(cls_flow_mttd.mean()) if len(cls_flow_mttd) else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values("attack_class")
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    section("STEP 4: MTTD with the campaign-initial flow removed")
    log("  Every campaign's FIRST flow carries an all-zero behavioral vector -- physically")
    log("  correct, since a source IP has no prior window activity -- and the deployed rule")
    log("  scores those flows far above threshold, so the first flow is detected in every")
    log("  single campaign and MTTD collapses to the scoring latency. That makes the headline")
    log("  MTTD a measurement of the generator's window initialization, not of timeliness.")
    log("  This variant drops each campaign's earliest flow and asks the harder question: once")
    log("  that free detection is removed, how long does the campaign run unseen?")
    ordered = overlay.sort_values(["campaign_id", "inject_timestamp"]).copy()
    ordered["pos"] = ordered.groupby("campaign_id").cumcount()
    later = ordered[ordered["pos"] > 0]
    det_ids = set(detections["inject_id"])
    onset = overlay.groupby("campaign_id")["inject_timestamp"].min()
    rows_excl = []
    for cls, grp in later.groupby("attack_class"):
        per_camp = []
        undetected = 0
        for cid, cg in grp.groupby("campaign_id"):
            hit = cg[cg["inject_id"].isin(det_ids)]
            if hit.empty:
                undetected += 1
                continue
            per_camp.append((hit["inject_timestamp"].min() - onset[cid]).total_seconds())
        arr = np.array(per_camp, dtype=float)
        rows_excl.append({
            "attack_class": cls,
            "n_campaigns": int(grp["campaign_id"].nunique()),
            "campaigns_detected_after_first": len(arr),
            "campaigns_undetected_after_first": undetected,
            "mttd_mean_s": float(arr.mean()) if len(arr) else float("nan"),
            "mttd_median_s": float(np.median(arr)) if len(arr) else float("nan"),
            "mttd_max_s": float(arr.max()) if len(arr) else float("nan"),
        })
    excl = pd.DataFrame(rows_excl).sort_values("attack_class")
    log("")
    log(excl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    excl.to_csv(ROOT / "reports/mttd_metrics_excluding_first_flow.csv", index=False)

    section("STEP 5: overall")
    all_camps = overlay["campaign_id"].nunique()
    det_camps = int(table["campaigns_detected"].sum())
    log(f"  campaigns detected: {det_camps} / {all_camps} "
        f"({det_camps / all_camps * 100:.1f}%)   undetected: {all_camps - det_camps}")
    if len(mttd_campaign):
        log(f"  MTTD per campaign, over DETECTED campaigns only: "
            f"mean {mttd_campaign.mean():.4f}s  median {mttd_campaign.median():.4f}s  "
            f"max {mttd_campaign.max():.4f}s")
    if len(mttd_flow):
        log(f"  MTTD per flow: mean {mttd_flow.mean():.6f}s  median {mttd_flow.median():.6f}s")

    section("STEP 6: how to read this, and how not to")
    log("  The detection rate above is NOT detection evidence. The overlay's distributions are")
    log("  hand-set from attack mechanics rather than fitted to real captures, so it measures")
    log("  whether the rule fires on flows shaped like this -- nothing about real traffic.")
    log("  MTTD is conditional on detection. Undetected campaigns are counted in their own")
    log("  column and given no MTTD; averaging a miss as a slow detection would flatter the")
    log("  system precisely where it fails.")
    log("  The headline per-campaign MTTD in STEP 3 is DEGENERATE and must not be quoted as a")
    log("  result: the first flow of all 228 campaigns is detected, so it equals the scoring")
    log("  latency by construction. Cite STEP 4 instead, and state that it excludes each")
    log("  campaign's first flow and why.")
    log("  Per-flow MTTD collapses to the scoring latency for the same reason.")
    log("  One more measured limitation of the overlay itself: its behavioral columns are far")
    log("  weaker than real traffic's. conn_count_1m tops out at 0.014 against 1.0 in the real")
    log("  unified test split, and uniq_dport_count_1m at 0.32 against 1.0. The synthetic port")
    log("  scan is much less aggressive than a real one on exactly the features Option A added,")
    log("  which is a concrete reason -- not just a caveat -- that detection rates here do not")
    log("  transfer.")

    table.to_csv(METRICS_PATH, index=False)
    scored.to_parquet(SCORES_PATH, index=False)
    (ROOT / "reports/mttd_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {METRICS_PATH}")
    log(f"[saved] {SCORES_PATH}")
    log(f"[saved] {REPORT_PATH}")
    log("=== MTTD EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
