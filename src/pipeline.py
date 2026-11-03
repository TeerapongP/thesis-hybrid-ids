"""pipeline.py -- the single end-to-end entry point: one network flow -> a dispatched alert.

Every layer of this thesis is implemented and measured, but until this module nothing ran them
as one program: the only code that chained ML -> fusion -> RAG -> LLM was
src/evaluation/eval_llm_faithfulness.py, an evaluation harness that never reaches the
notification layer. This module is the demonstration path, in the deployed order:

    1. pick rows from data/features/unified/cicids2017_unified_test.parquet
    2. score with models/random_forest_unified.pkl, isolation_forest_unified.pkl,
       autoencoder_unified.pt  (94 features, selected by exclusion -- never positionally)
    3. fuse with src.modeling.fusion_layer.fuse_with_metadata (the deployed cascade; the rule
       is imported, never reimplemented here)
    4. optionally SHAP-attribute the deciding branch (src.modeling.shap_explain.FusionExplainer)
    5. retrieve MITRE context with src.rag.rag_retrieval.RagRetriever (top_k=3), queried with
       the Random Forest predicted label
    6. generate the Thai explanation with src.llm.llm_explain.explain_alert
    7. dispatch with src.notifications.alert_dispatcher.dispatch_alert

TWO INTERPRETERS, ONE RUN. Steps 2-5 need pandas/torch/sentence-transformers, which only the
Anaconda interpreter has; steps 6-7 need langchain/pika/elasticsearch/influxdb_client, which
only the uv .venv has. No single interpreter can import both. The arrangement is the one
eval_llm_faithfulness.py already uses and is reused rather than reinvented: the uv entry point
shells out to Anaconda for `--stage score`, which writes an intermediate JSON, then runs the
LLM and dispatch stages itself.

    uv run python -m src.pipeline --random 3 --dry-run

DRY RUN IS THE DEFAULT, because dispatching sends real notifications. `--dispatch` is the
only way to actually call dispatch_alert; without it every layer still runs and the assembled
alert record is printed instead.

IDENTIFIERS. The unified feature matrix carries no Source IP, Destination IP or Timestamp:
build_unified_population drops identifier columns from model input on purpose, because Source
IP would let a model memorise which hosts attacked in this capture. They are recovered the
same way eval_llm_faithfulness recovers them -- by re-deriving the same split from the same
source with the same seed and indexing it POSITIONALLY -- and the alignment is ASSERTED on
Label, never assumed. Only the identifier columns are read from the source parquet, which is
sound because the stratified split depends on row order and the Label column alone; the Label
assertion is the guard that this stays true.

SEAMS FOUND BY RUNNING THIS, recorded rather than papered over:

  * NOTHING GATES NON-ALERTING FLOWS. A row with fused_prediction = 0 still goes through RAG,
    the LLM and dispatch, arriving as a LOW-priority alert in the Gmail digest queue, because
    derive_priority maps "not an attack" to LOW rather than to "no alert". Verified on test row
    44999 (RF P(attack) = 0.0050, fused_score 0.0050, fused_prediction 0): a Thai explanation
    was generated and the dry run reported it would be queued. For a demonstration that is
    useful -- every layer is exercised on benign traffic too -- but a deployment would need a
    suppression step here, and the deployed code does not have one.
  * `confidence` IS RF CERTAINTY IN EITHER DIRECTION, which the LLM reads as certainty THAT IT
    IS AN ATTACK. On the same benign row it wrote "low confidence (0.99)" in Thai, combining
    the correct number with the wrong reading of it. The quantity comes straight from the
    fusion rule's own `|P(attack) - 0.5| * 2`, so this is a presentation seam between the
    fusion layer and the explanation layer, not an LLM defect.
  * THE RAG QUERY IS THE RF PREDICTED LABEL, including on BENIGN rows, where it degrades to a
    generic anomaly sentence and retrieves T1595/T1046/T1071.001 regardless of the flow. No
    production query builder exists in the repo; eval_llm_faithfulness has the same limitation.
  * LINE AND GMAIL ARE DRY-RUN ONLY in this environment (no outbound internet, Gmail OAuth2
    credentials deliberately blank). Expected, not a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ML_PYTHON = os.environ.get("ML_PYTHON", "C:/Users/devMrX/anaconda3/python.exe")

UNIFIED_TEST = ROOT / "data/features/unified/cicids2017_unified_test.parquet"
STAGE_JSON = ROOT / "reports/pipeline_last_run.json"

# Same exclusion list as shap_explain / train_unified_ensemble. The feature set is selected by
# excluding these and keeping the numeric remainder, so a column added upstream cannot silently
# shift a positional slice.
NON_FEATURE_COLS = [
    "Label", "attack_cat", "technique_id", "technique_name", "tactic", "mapping_confidence",
    "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique",
    "orig_row_index",
]
EXPECTED_FEATURES = 94
DEFAULT_TOP_K = 3
SEED = 42
# The RAG query for rows Random Forest predicts BENIGN. Same sentence eval_llm_faithfulness
# uses, so the two produce comparable retrievals.
GENERIC_ANOMALY_QUERY = "anomalous network traffic flagged by anomaly detector"


# ----------------------------------------------------------------------------
# Pure helpers (stdlib only, importable under either interpreter)
# ----------------------------------------------------------------------------
def select_rows(
    labels: Sequence[str],
    rows: Sequence[int] | None = None,
    label: str | None = None,
    random_n: int | None = None,
    seed: int = SEED,
) -> list[int]:
    """Positional indices into the test split, from exactly one selection mode.

    `rows` wins if given. `label` narrows the candidate pool and, with `random_n`, draws that
    many from it (default 1). `random_n` alone draws from the whole split.
    """
    import numpy as np

    n_total = len(labels)
    if rows:
        bad = [r for r in rows if not 0 <= r < n_total]
        if bad:
            raise ValueError(f"--row out of range for a {n_total}-row split: {bad}")
        return list(rows)

    if label is None and random_n is None:
        raise ValueError("pick rows with --row N, --label NAME or --random N")

    pool = np.arange(n_total)
    if label is not None:
        arr = np.asarray(labels, dtype=object)
        pool = np.flatnonzero(arr == label)
        if pool.size == 0:
            raise ValueError(f"no rows with Label {label!r} in the test split")
    want = random_n if random_n is not None else 1
    if want > pool.size:
        raise ValueError(f"asked for {want} rows but only {pool.size} are available")
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(pool, size=want, replace=False))


def feature_columns(df) -> list[str]:
    """The 94 model inputs: numeric columns that are not reference/label-derived columns."""
    import pandas as pd

    cols = [c for c in df.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) != EXPECTED_FEATURES:
        raise AssertionError(f"expected {EXPECTED_FEATURES} feature columns, got {len(cols)}")
    return cols


def recover_identifiers(raw_test, pos_idx: Sequence[int], expected_labels: Sequence[str]):
    """Rows of the re-derived split that correspond POSITIONALLY to `pos_idx` of the matrix.

    The Label equality assertion is the whole safety argument for the positional index: if the
    re-derived split ever stopped matching the saved feature matrix row-for-row, every alert
    would carry another flow's IPs and nothing else would notice.
    """
    if len(raw_test) != len(expected_labels):
        raise AssertionError(
            f"re-derived split has {len(raw_test)} rows, feature matrix has {len(expected_labels)}"
        )
    picked = raw_test.iloc[list(pos_idx)].reset_index(drop=True)
    got = [str(v) for v in picked["Label"].tolist()]
    want = [str(expected_labels[i]) for i in pos_idx]
    if got != want:
        raise AssertionError(
            "positional alignment between the feature matrix and the re-derived split is wrong: "
            f"{got} != {want}"
        )
    return picked


def _iso(value: Any) -> str:
    """ISO-8601 string -- alert_dispatcher parses timestamps with datetime.fromisoformat."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_alert_context(
    ident_row: Mapping[str, Any],
    rf_predicted_label: str,
    fused_score: float,
    fused_prediction: int,
    confidence: float,
    model_used: str,
    score_latency_s: float,
) -> dict[str, Any]:
    """The alert context every downstream layer consumes.

    `ingested_at` is the flow's own capture time and `detected_at` is that plus the MEASURED
    per-row scoring latency, which is what src/evaluation/mttd_eval.py does. Transport,
    enrichment, RAG and LLM time are excluded deliberately: none is live-verified end to end
    and all of it happens after the detection decision.
    """
    ts = _iso(ident_row["Timestamp"])
    detected = _iso(_add_seconds(ident_row["Timestamp"], score_latency_s))
    return {
        "source_ip": str(ident_row["Source IP"]),
        "dst_ip": str(ident_row["Destination IP"]),
        "dst_port": int(ident_row["Destination Port"]),
        "timestamp": ts,
        "ingested_at": ts,
        "detected_at": detected,
        "rf_predicted_label": rf_predicted_label,
        "fused_score": float(fused_score),
        "fused_prediction": int(fused_prediction),
        "confidence": float(confidence),
        "model_used": str(model_used),
    }


def _add_seconds(timestamp: Any, seconds: float):
    import datetime as _dt

    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if isinstance(timestamp, _dt.datetime):
        return timestamp + _dt.timedelta(seconds=seconds)
    return timestamp


def rag_query_for(rf_predicted_label: str) -> str:
    return rf_predicted_label if rf_predicted_label != "BENIGN" else GENERIC_ANOMALY_QUERY


def ollama_status(base_url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """(reachable, detail). A clear message beats a stack trace from deep inside langchain."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            models = [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]
        return True, f"{len(models)} model(s) available: {', '.join(models) or 'none'}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# Stage 1 (Anaconda): score, fuse, SHAP, retrieve
# ----------------------------------------------------------------------------
def score_stage(args) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(ROOT))
    from src.features import build_unified_population as BUP
    from src.modeling import fusion_layer
    from src.rag.rag_retrieval import RagRetriever

    test = pd.read_parquet(UNIFIED_TEST)
    labels = test["Label"].tolist()
    pos_idx = select_rows(labels, rows=args.row, label=args.label, random_n=args.random, seed=args.seed)
    print(f"[score] selected {len(pos_idx)} row(s): {pos_idx}", flush=True)

    feat_cols = feature_columns(test)
    sample = test.iloc[pos_idx].reset_index(drop=True)
    x = sample[feat_cols].to_numpy(dtype=np.float64)

    print("[score] re-deriving the unified test split to recover IP/port/timestamp ...", flush=True)
    ident_cols = ["Label", "Source IP", "Destination IP", "Destination Port", "Timestamp"]
    _, _, raw_test = BUP.split_population(pd.read_parquet(BUP.SOURCE, columns=ident_cols))
    ident = recover_identifiers(raw_test, pos_idx, labels)

    rf = joblib.load(ROOT / "models/random_forest_unified.pkl")
    iso = joblib.load(ROOT / "models/isolation_forest_unified.pkl")
    cfg = json.loads((ROOT / "models/autoencoder_unified_config.json").read_text(encoding="utf-8"))
    ae = _build_autoencoder(nn, cfg)
    ae.load_state_dict(torch.load(ROOT / "models/autoencoder_unified.pt", map_location="cpu"))
    ae.eval()

    t0 = time.perf_counter()
    proba = rf.predict_proba(x)
    classes = list(rf.classes_)
    rf_attack = 1.0 - proba[:, classes.index("BENIGN")]
    rf_label = np.asarray(classes)[proba.argmax(1)]
    if_score = -iso.score_samples(x)
    with torch.no_grad():
        t = torch.as_tensor(x.astype(np.float32))
        ae_err = torch.mean((ae(t) - t) ** 2, dim=1).numpy()
    fused, pred, margin, model_used = fusion_layer.fuse_with_metadata(rf_attack, if_score, ae_err)
    score_latency_s = (time.perf_counter() - t0) / len(pos_idx)
    print(f"[score] fused {len(pos_idx)} row(s) in {score_latency_s * len(pos_idx):.3f}s "
          f"({score_latency_s * 1000:.1f} ms/row)", flush=True)

    shap_records: list[dict[str, Any] | None] = [None] * len(pos_idx)
    if args.shap:
        from src.modeling.shap_explain import FusionExplainer

        print("[score] SHAP-attributing the deciding branch (~0.2 s/alert) ...", flush=True)
        explainer = FusionExplainer()
        shap_records = [dict(r) for r in explainer.explain(sample, top_k=5)]

    retriever = RagRetriever()
    alerts = []
    for i, row_index in enumerate(pos_idx):
        ctx = build_alert_context(
            ident.iloc[i], str(rf_label[i]), float(fused[i]), int(pred[i]),
            float(abs(rf_attack[i] - 0.5) * 2), str(model_used[i]), score_latency_s,
        )
        query = rag_query_for(ctx["rf_predicted_label"])
        hits = retriever.retrieve(query, top_k=args.top_k)
        alerts.append({
            "test_row_index": int(row_index),
            "true_label": str(sample.loc[i, "Label"]),
            "alert_context": ctx,
            "scores": {
                "rf_attack_probability": float(rf_attack[i]),
                "rf_predicted_label": str(rf_label[i]),
                "if_anomaly_score": float(if_score[i]),
                "ae_reconstruction_error": float(ae_err[i]),
                "if_norm": float(fusion_layer.normalize_if_score(if_score[i])),
                "ae_norm": float(fusion_layer.normalize_ae_score(ae_err[i])),
                "margin": float(margin[i]),
            },
            "shap": shap_records[i],
            "rag_query": query,
            "rag_hits": [
                {"rank": h["rank"], "id": h["id"], "metadata": h["metadata"],
                 "document": h["document"], "combined_score": float(h["combined_score"])}
                for h in hits
            ],
        })

    payload = {
        "stage": "score",
        "n_features": len(feat_cols),
        "fusion_strategy": fusion_layer.CHOSEN_STRATEGY,
        "score_latency_ms_per_row": score_latency_s * 1000,
        "shap": bool(args.shap),
        "alerts": alerts,
    }
    STAGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STAGE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[score] wrote {STAGE_JSON}", flush=True)
    return payload


def _build_autoencoder(nn, cfg):
    """Same architecture as train_unified_ensemble.py / autoencoder_unified_config.json."""

    def stack(dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class AE(nn.Module):
        def __init__(self, d, hidden, bottleneck):
            super().__init__()
            self.encoder = stack([d] + list(hidden) + [bottleneck])
            self.decoder = stack([bottleneck] + list(hidden)[::-1] + [d])

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AE(cfg["input_dim"], cfg["hidden_dims"], cfg["bottleneck_dim"])


# ----------------------------------------------------------------------------
# Stage 2 (uv .venv): LLM explanation + dispatch
# ----------------------------------------------------------------------------
def llm_stage(payload: dict[str, Any]) -> None:
    """Fill each alert's `llm_output`, or `llm_error` with the reason it has none."""
    from src.llm import llm_explain

    config = llm_explain.load_env_config()
    up, detail = ollama_status(config["base_url"])
    if not up:
        raise RuntimeError(
            f"the LLM backend at {config['base_url']} is not reachable ({detail}). "
            f"Start it (`ollama serve`, model {config['model']}) or re-run with --no-llm."
        )
    print(f"[llm] {config['base_url']} reachable -- {detail}", flush=True)

    for alert in payload["alerts"]:
        ctx = alert["alert_context"]
        t0 = time.perf_counter()
        try:
            alert["llm_output"] = llm_explain.explain_alert(
                ctx, alert["rag_hits"], ctx["fused_score"], ctx["fused_prediction"],
                ctx["confidence"], config,
            )
            alert["llm_error"] = None
        except ValueError as exc:
            # explain_alert raises when the model names an entity absent from FACTS. That
            # guard is deliberate -- a wrong hostname sends an analyst to the wrong machine --
            # so it is surfaced per alert, not swallowed.
            alert["llm_output"], alert["llm_error"] = None, f"{type(exc).__name__}: {exc}"
            print(f"[llm] row {alert['test_row_index']}: FACTS guard rejected the output -- {exc}",
                  flush=True)
        alert["llm_wall_s"] = round(time.perf_counter() - t0, 2)


def dispatch_stage(payload: dict[str, Any], dry_run: bool) -> None:
    from src.notifications import alert_dispatcher

    for alert in payload["alerts"]:
        out = alert.get("llm_output")
        if not out:
            alert["dispatch"] = {"status": "skipped", "reason": alert.get("llm_error") or "no llm_output"}
            continue
        ctx, fused = alert["alert_context"], alert["alert_context"]["fused_score"]
        if dry_run:
            model_used = alert_dispatcher.derive_model_used(ctx, out)
            record = alert_dispatcher.assemble_alert_record(ctx, fused, out, model_used, "DRY-RUN")
            channel = {"HIGH": "LINE (immediate)", "MEDIUM": "Gmail (immediate)"}.get(
                out["priority"], "Gmail daily digest (SQLite queue)"
            )
            alert["dispatch"] = {
                "status": "dry_run", "would_route_to": channel,
                "routing_key": f"alert.{out['priority']}", "record": record,
            }
        else:
            alert["dispatch"] = alert_dispatcher.dispatch_alert(ctx, fused, out)


# ----------------------------------------------------------------------------
# Human-readable summary
# ----------------------------------------------------------------------------
def render_summary(payload: dict[str, Any]) -> str:
    bar = "=" * 78
    L = [bar, "HYBRID IDS -- END-TO-END PIPELINE RUN", bar,
         f"fusion strategy: {payload['fusion_strategy']}   "
         f"model features: {payload['n_features']}   "
         f"scoring: {payload['score_latency_ms_per_row']:.1f} ms/row", ""]
    for alert in payload["alerts"]:
        ctx, sc = alert["alert_context"], alert["scores"]
        L += [bar,
              f"row {alert['test_row_index']}  true Label: {alert['true_label']}",
              bar,
              f"  flow            {ctx['source_ip']} -> {ctx['dst_ip']}:{ctx['dst_port']}  @ {ctx['timestamp']}",
              f"  RF              P(attack)={sc['rf_attack_probability']:.4f}  predicted={sc['rf_predicted_label']}",
              f"  IF              raw={sc['if_anomaly_score']:.4f}  normalized={sc['if_norm']:.4f}",
              f"  AE              mse={sc['ae_reconstruction_error']:.6f}  normalized={sc['ae_norm']:.4f}",
              f"  FUSION          fused_score={ctx['fused_score']:.4f}  attack={bool(ctx['fused_prediction'])}"
              f"  decided_by={ctx['model_used']}  margin={sc['margin']:+.4f}"]
        if alert.get("shap"):
            L.append(f"  SHAP            risk_score={alert['shap']['risk_score']}  branch={alert['shap']['model_used']}")
            for f in alert["shap"]["top_features"]:
                L.append(f"                    {f['feature']:<28} value={f['value']:>10.4f}  shap={f['shap_value']:+.5f}  {f['direction']}")
        L.append(f"  RAG             query={alert['rag_query']!r}")
        for hit in alert["rag_hits"]:
            meta = hit["metadata"]
            L.append(f"                    #{hit['rank']} {hit['id']}  "
                     f"{meta.get('technique_id', '-')} {meta.get('technique_name', '')}  "
                     f"score={hit['combined_score']:.4f}")
        out = alert.get("llm_output")
        if out:
            L += [f"  LLM             priority={out['priority']}  technique={out['technique_id']}  "
                  f"tactic={out['mitre_tactic']}  {alert.get('llm_wall_s', 0)}s",
                  f"                  {out['explanation_th']}"]
            for i, step in enumerate(out["remediation"], 1):
                L.append(f"                    {i}. {step}")
        elif alert.get("llm_error"):
            L.append(f"  LLM             FAILED: {alert['llm_error']}")
        else:
            L.append("  LLM             skipped (--no-llm)")
        d = alert.get("dispatch")
        if d and d["status"] == "dry_run":
            rec = d["record"]
            L += [f"  DISPATCH        DRY RUN -- nothing sent. Would publish {d['routing_key']} "
                  f"to RabbitMQ, write InfluxDB + Elasticsearch,",
                  f"                  and notify via {d['would_route_to']}.",
                  f"                  risk_score={rec['risk_score']}  model_used={rec['model_used']}  "
                  f"anomaly_type={rec['anomaly_type']}  mttd_ms={rec['mttd_ms']}"]
        elif d and d["status"] == "skipped":
            L.append(f"  DISPATCH        skipped ({d['reason']})")
        elif d:
            L.append(f"  DISPATCH        alert_id={d.get('alert_id')}  priority={d.get('priority')}  "
                     f"rabbitmq={d.get('rabbitmq', {}).get('status')}  influxdb={d.get('influxdb', {}).get('status')}  "
                     f"elasticsearch={d.get('elasticsearch', {}).get('status')}  "
                     f"notification={d.get('notification', {}).get('status')}")
        else:
            L.append("  DISPATCH        not reached")
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run one or more flows through every layer of the Hybrid IDS.",
    )
    p.add_argument("--row", type=int, action="append", help="test-split row index (repeatable)")
    p.add_argument("--label", help='pick rows with this Label, e.g. "PortScan"')
    p.add_argument("--random", type=int, metavar="N", help="pick N random rows")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="RAG hits per alert")
    p.add_argument("--shap", action="store_true", help="SHAP-attribute the deciding branch (~0.2 s/alert)")
    p.add_argument("--no-llm", action="store_true", help="stop after fusion + RAG (no Ollama needed)")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="run every layer but send nothing (DEFAULT)")
    p.add_argument("--dispatch", action="store_true",
                   help="actually call dispatch_alert -- this sends real notifications")
    p.add_argument("--json", action="store_true", help="emit the machine-readable run JSON")
    p.add_argument("--quiet", action="store_true",
                   help="internal: the scoring subprocess prints progress only, not the summary")
    p.add_argument("--stage", choices=["score", "all"], default="all",
                   help="internal: 'score' is the Anaconda half of the run")
    return p


def _forward_args(args) -> list[str]:
    out: list[str] = ["--stage", "score", "--quiet", "--seed", str(args.seed), "--top-k", str(args.top_k)]
    for r in args.row or []:
        out += ["--row", str(r)]
    if args.label:
        out += ["--label", args.label]
    if args.random:
        out += ["--random", str(args.random)]
    if args.shap:
        out.append("--shap")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    dry_run = not args.dispatch

    if args.stage == "score":
        payload = score_stage(args)
        if args.quiet:
            return 0
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_summary(payload))
        return 0

    # The uv half: shell out to Anaconda for everything that needs pandas/torch.
    cmd = [ML_PYTHON, "-m", "src.pipeline", *_forward_args(args)]
    print(f"[pipeline] scoring stage -> {ML_PYTHON}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if proc.returncode != 0:
        print("[pipeline] scoring stage failed; nothing downstream was run", file=sys.stderr)
        return proc.returncode
    payload = json.loads(STAGE_JSON.read_text(encoding="utf-8"))

    if args.no_llm:
        try:
            from src.llm import llm_explain

            for alert in payload["alerts"]:
                ctx = alert["alert_context"]
                alert["priority_if_dispatched"] = llm_explain.derive_priority(
                    ctx["fused_score"], ctx["fused_prediction"]
                )
        except Exception as exc:  # langchain missing: report, do not fake a priority
            print(f"[pipeline] priority preview unavailable ({type(exc).__name__}: {exc})", flush=True)
        print("[pipeline] --no-llm: stopping after fusion + RAG; nothing dispatched", flush=True)
    else:
        llm_stage(payload)
        dispatch_stage(payload, dry_run)
        if dry_run:
            print("[pipeline] --dry-run (default): dispatch_alert was NOT called", flush=True)

    payload["dry_run"] = dry_run
    STAGE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_summary(payload))
        for alert in payload["alerts"]:
            if "priority_if_dispatched" in alert:
                print(f"  row {alert['test_row_index']}: priority would be "
                      f"{alert['priority_if_dispatched']} (derive_priority, no LLM called)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
