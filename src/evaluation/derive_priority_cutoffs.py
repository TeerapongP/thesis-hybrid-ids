# -*- coding: utf-8 -*-
"""Derive a HIGH-priority cutoff for the fallback branch, instead of sharing RF's.

The problem this answers, measured 2026-11-06: `derive_priority` applies one cutoff
(`fused_score >= 0.80`) to a score whose two branches are on incompatible scales. RF's
branch emits a probability reaching 1.0; the fallback emits `(if_norm + ae_norm) / 2`,
which never exceeded 0.7031 over 250,000 test rows. So no fallback-decided alert could
reach HIGH, and the LINE channel was RF-only -- while the fallback exists precisely to
catch what RF cannot.

Why a second constant rather than rescaling the fallback score: the score itself is what
`fusion_layer` thresholds at 0.03 and what SHAP attributes in its own units, and rescaling
it would invalidate both. The branches stay as they are and the priority rule learns that
they are different.

**Nothing here is hand-picked.** Two anchors are computed and printed side by side, both
expressed relative to the RF branch's existing 0.80 so the new number inherits its
meaning rather than inventing one:

  percentile-matched
      Find where 0.80 sits in the RF branch's train-BENIGN score distribution, then take
      that same percentile of the fallback branch's train-BENIGN scores. "HIGH" then means
      "this far into the tail of normal traffic", identically on both branches. This is the
      house convention -- `siem_baseline.py` derives every threshold as a train-BENIGN
      percentile for the same reason.

  share-matched
      Find what fraction of *alerting* RF rows are HIGH, then take the fallback score that
      makes the same fraction of alerting fallback rows HIGH. "HIGH" then means "the top
      N% of what this branch alerts on", so an analyst receives a comparable share of
      pages from each branch.

Both can fail, and the failure is informative rather than a crash: if RF's train-BENIGN
distribution is so concentrated at 0 that 0.80 sits above its 100th percentile, the
percentile anchor is undefined and the script says so.

    python -m src.evaluation.derive_priority_cutoffs

Writes reports/priority_cutoff_derivation.txt and _candidates.csv. Choosing which anchor
to deploy is a separate, deliberate edit to llm_explain -- this script only measures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.modeling import fusion_layer as F  # noqa: E402
from src.modeling.train_unified_ensemble import Autoencoder  # noqa: E402
from src.pipeline import feature_columns  # noqa: E402

TRAIN = ROOT / "data" / "features" / "unified" / "cicids2017_unified_train.parquet"
TEST = ROOT / "data" / "features" / "unified" / "cicids2017_unified_test.parquet"
REPORT = ROOT / "reports" / "priority_cutoff_derivation.txt"
CANDIDATES = ROOT / "reports" / "priority_cutoff_candidates.csv"

RF_HIGH_CUTOFF = 0.80  # llm_explain.PRIORITY_HIGH_MIN_SCORE; deliberately left unchanged
LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def score_all(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """RF / IF / AE raw scores plus the fused score and which branch decided each row."""
    X = df[cols].to_numpy(dtype=np.float32)

    rf = joblib.load(ROOT / "models" / "random_forest_unified.pkl")
    t0 = time.perf_counter()
    proba = rf.predict_proba(X)
    p_attack = 1.0 - proba[:, list(rf.classes_).index("BENIGN")]
    log(f"  RF scored {len(X):,} rows in {time.perf_counter() - t0:.1f}s")

    iso = joblib.load(ROOT / "models" / "isolation_forest_unified.pkl")
    t0 = time.perf_counter()
    if_raw = -iso.score_samples(X)
    log(f"  IF scored in {time.perf_counter() - t0:.1f}s")

    cfg = json.loads((ROOT / "models" / "autoencoder_unified_config.json").read_text())
    bottleneck = cfg.get("bottleneck_dim", cfg.get("bottleneck"))
    ae = Autoencoder(len(cols), cfg["hidden_dims"], bottleneck)
    ae.load_state_dict(torch.load(ROOT / "models" / "autoencoder_unified.pt", map_location="cpu"))
    ae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    t0 = time.perf_counter()
    errs = []
    with torch.no_grad():
        for start in range(0, len(X), 100_000):
            batch = torch.tensor(X[start:start + 100_000], device=device)
            errs.append(((ae(batch) - batch) ** 2).mean(dim=1).cpu().numpy())
    ae_err = np.concatenate(errs)
    log(f"  AE scored in {time.perf_counter() - t0:.1f}s on {device}")

    fused, pred, _, used = F.fuse_with_metadata(p_attack, if_raw, ae_err)
    return {"rf": p_attack, "fused": fused, "pred": pred, "used": used}


def percentile_of(values: np.ndarray, x: float) -> float:
    """What percentile of `values` the value `x` sits at (0-100)."""
    return 100.0 * float((values < x).mean())


def main() -> int:
    log("=" * 78)
    log("Per-branch HIGH-priority cutoff derivation")
    log("=" * 78)
    log(f"RF branch HIGH cutoff (unchanged, from llm_explain): {RF_HIGH_CUTOFF}")
    log(f"fusion alert threshold: {F.FUSED_DECISION_THRESHOLD}")
    log(f"RF confidence gate: {F.CONFIDENCE_THRESHOLD}")
    log()

    train = pd.read_parquet(TRAIN)
    cols = feature_columns(train)
    log(f"train rows: {len(train):,}  features: {len(cols)}")
    log("scoring train split ...")
    tr = score_all(train, cols)

    benign = (train["Label"] == "BENIGN").to_numpy()
    rf_branch = tr["used"] == "random_forest"
    fb_branch = tr["used"] == "if_ae_fallback"
    log()
    log(f"train BENIGN rows: {int(benign.sum()):,}")
    log(f"  decided by RF:       {int((benign & rf_branch).sum()):,}")
    log(f"  decided by fallback: {int((benign & fb_branch).sum()):,}")

    rf_benign = tr["fused"][benign & rf_branch]
    fb_benign = tr["fused"][benign & fb_branch]
    if len(fb_benign) == 0:
        log("\nABORT: no BENIGN rows reached the fallback branch in train; cannot anchor.")
        REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        return 1

    log()
    log("fallback-branch fused score, train BENIGN distribution:")
    for q in (50, 90, 99, 99.9, 99.99, 100):
        log(f"  p{q:<6} {np.percentile(fb_benign, q):.4f}")

    # ---- anchor 1: same percentile of each branch's train-BENIGN distribution ----
    pct = percentile_of(rf_benign, RF_HIGH_CUTOFF)
    log()
    log(f"anchor 1 -- percentile-matched")
    log(f"  {RF_HIGH_CUTOFF} sits at p{pct:.6f} of the RF branch's train-BENIGN scores")
    if pct >= 100.0:
        pctile_cutoff = float("nan")
        log("  UNDEFINED: 0.80 is above every RF train-BENIGN score, so the matching")
        log("  percentile is p100 and carries no information. Anchor unusable.")
    else:
        pctile_cutoff = float(np.percentile(fb_benign, pct))
        log(f"  the same percentile of fallback train-BENIGN = {pctile_cutoff:.4f}")

    # ---- anchor 2: same HIGH share among each branch's alerting rows ----
    log()
    log("anchor 2 -- share-matched (measured on the TEST split, where alerting rows live)")
    test = pd.read_parquet(TEST)
    log(f"  test rows: {len(test):,}; scoring ...")
    te = score_all(test, cols)
    alerting = te["pred"] == 1
    rf_alert = te["fused"][alerting & (te["used"] == "random_forest")]
    fb_alert = te["fused"][alerting & (te["used"] == "if_ae_fallback")]
    log(f"  alerting rows: RF {len(rf_alert):,}  fallback {len(fb_alert):,}")
    if len(rf_alert) == 0 or len(fb_alert) == 0:
        share_cutoff = float("nan")
        log("  UNDEFINED: one branch has no alerting rows on this split.")
    else:
        high_share = float((rf_alert >= RF_HIGH_CUTOFF).mean())
        log(f"  RF alerting rows that are HIGH: {high_share:.4%}")
        share_cutoff = float(np.quantile(fb_alert, 1.0 - high_share))
        log(f"  fallback score giving the same share = {share_cutoff:.4f}")
        log(f"  fallback alerting score range: {fb_alert.min():.4f} - {fb_alert.max():.4f}")

    # ---- anchor 3: the fallback branch's OWN train-BENIGN tail ----
    #
    # Needed because both anchors above try to inherit meaning from RF's 0.80 and that
    # number turns out to carry none: the 0.95 confidence gate forces an RF-decided row to
    # score >= 0.975 or <= 0.025, so every RF-decided ALERT is already far above 0.80 and
    # the RF branch can never produce a MEDIUM. Matching a vacuous threshold reproduces its
    # vacuity -- the share-matched anchor lands on 0.0301, the alert threshold itself, which
    # would page an analyst for every fallback alert including its false positives.
    #
    # So this anchor ignores RF and asks the question directly on the branch that needs an
    # answer: how far into the tail of NORMAL traffic must this branch's score sit before a
    # human should be woken? That is the same form as every siem_baseline threshold.
    log()
    log("anchor 3 -- percentile of the FALLBACK branch's own train-BENIGN tail")
    log("  (independent of RF's cutoff, which the two anchors above showed to be vacuous)")
    sweep = []
    for q in (99.0, 99.5, 99.9, 99.95, 99.99):
        cutoff = float(np.percentile(fb_benign, q))
        n_alert_high = int((fb_alert >= cutoff).sum()) if len(fb_alert) else 0
        sweep.append((q, cutoff, n_alert_high))
        log(f"  p{q:<6} cutoff {cutoff:.4f} -> {n_alert_high:,} of {len(fb_alert):,} "
            f"fallback alerts HIGH")

    # ---- what each candidate would do ----
    log()
    log("=" * 78)
    log("Effect of each candidate on the TEST split")
    log("=" * 78)
    candidates = [("current (shared 0.80)", RF_HIGH_CUTOFF),
                  ("percentile-matched", pctile_cutoff),
                  ("share-matched", share_cutoff)]
    candidates += [(f"own-tail p{q}", c) for q, c, _ in sweep]
    rows = []
    for name, cutoff in candidates:
        if cutoff != cutoff:  # NaN
            log(f"{name:24s} unusable")
            rows.append({"candidate": name, "fallback_high_cutoff": "",
                         "fallback_alerts": len(fb_alert), "fallback_high": "",
                         "fallback_high_pct": "", "rf_high": int((rf_alert >= RF_HIGH_CUTOFF).sum())}
                        )
            continue
        n_high = int((fb_alert >= cutoff).sum())
        log(f"{name:24s} cutoff {cutoff:.4f} -> {n_high:,} of {len(fb_alert):,} "
            f"fallback alerts become HIGH ({100 * n_high / max(1, len(fb_alert)):.2f}%)")
        rows.append({"candidate": name, "fallback_high_cutoff": round(cutoff, 6),
                     "fallback_alerts": len(fb_alert), "fallback_high": n_high,
                     "fallback_high_pct": round(100 * n_high / max(1, len(fb_alert)), 4),
                     "rf_high": int((rf_alert >= RF_HIGH_CUTOFF).sum())})

    # Attacks that RF named correctly but abstained on: the population the fix is for.
    lbl = test["Label"].to_numpy()
    fb_mask = alerting & (te["used"] == "if_ae_fallback")
    real_attack = fb_mask & (lbl != "BENIGN")
    log()
    log(f"fallback alerts that are real attacks: {int(real_attack.sum()):,}")
    for name, cutoff in candidates:
        if cutoff != cutoff:
            continue
        caught = int((te["fused"][real_attack] >= cutoff).sum())
        fp = int((te["fused"][fb_mask & (lbl == "BENIGN")] >= cutoff).sum())
        log(f"  {name:24s} -> {caught:,} real attacks reach HIGH, "
            f"{fp:,} BENIGN rows also reach HIGH")

    pd.DataFrame(rows).to_csv(CANDIDATES, index=False)
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    print(f"wrote {CANDIDATES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
