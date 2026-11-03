"""The LLM-only arm of the 4-way ablation: can the LLM detect without the ML ensemble?

WHY THIS ARM NEEDED BUILDING AT ALL
The 4-way ablation is specified as rule-based SIEM / ML-only / LLM-only / proposed hybrid.
Two of those four had never been measured, and one of them cannot be measured the way the
specification implies:

  * ML-only vs hybrid have IDENTICAL detection numbers. The LLM layer writes explanations;
    it does not touch the decision. So a detection-metric table with both arms in it would
    print the same row twice and imply a comparison that does not exist. The real difference
    between those two arms is analyst outcome, which is RQ4's A/B study, not this script.
  * LLM-only had no code path at all. `llm_explain.explain_alert` takes `fused_score` and
    `fused_prediction` as INPUTS -- it explains a decision the ensemble already made, and
    never makes one.

So this script builds the missing capability: give the LLM a flow and ask it to judge,
with no ML output anywhere in the prompt.

WHAT THE MODEL IS SHOWN, AND WHY
Physical units, not model features. The 94 model inputs are MinMax-scaled to [0,1], and a
number like `Flow Duration = 0.0032` carries no meaning a language model can reason about.
The raw values are recovered by re-deriving the same split from the same source with the
same seed and indexing positionally -- the alignment is asserted on Label, not assumed.

What is deliberately WITHHELD: `rf_predicted_label`, the fused score, and every other model
output. Including any of them would make this arm a measurement of the ensemble wearing an
LLM costume. The true `Label` is obviously withheld too.

WHY PRECISION IS NOT REPORTED AS COMPARABLE
This arm runs on a deliberately balanced subsample, because estimating a false-positive rate
needs benign rows and running 504,193 of them through an 8B model is not feasible. Precision
depends on prevalence, so this arm's precision cannot be compared against the SIEM and hybrid
arms, which are measured on the natural-prevalence test split. Recall and FPR do not depend
on prevalence and are the figures to compare. The report states this rather than printing a
precision column that invites the wrong comparison.

Run from the repo root with uv (needs LangChain):
    uv run python -m src.evaluation.eval_llm_only_baseline --limit 10   # smoke test
    uv run python -m src.evaluation.eval_llm_only_baseline              # full run
Scoring needs no ML models at all, so unlike the other evaluators there is no Anaconda stage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ML_PYTHON = os.environ.get("ML_PYTHON", "C:/Users/devMrX/anaconda3/python.exe")

FLOWS_PATH = ROOT / "reports/llm_only_flows.jsonl"
RAW_PATH = ROOT / "reports/llm_only_raw.jsonl"
METRICS_PATH = ROOT / "reports/llm_only_metrics.csv"
REPORT_PATH = ROOT / "reports/llm_only_report.txt"

SEED = 42
BENIGN_N = 100
PER_CATEGORY_CAP = 7

# Columns shown to the model, in physical units. Chosen to be what an analyst would actually
# look at on a flow record -- not the full 94, because dumping every CICFlowMeter statistic
# into the prompt buries the signal and costs tokens for no benefit.
SHOWN_COLUMNS = [
    "Destination Port", "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "Flow IAT Mean", "Flow IAT Std",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "conn_count_1m", "uniq_dport_count_1m", "fail_ratio_1m", "repeat_ratio_1m",
    "conn_count_5m", "uniq_dport_count_5m",
]


# ---------------------------------------------------------------------------
# Stage 1 (Anaconda): build the flow records in physical units
# ---------------------------------------------------------------------------
def build_flows() -> None:
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from src.features import build_unified_population as BUP

    rng = np.random.default_rng(SEED)
    test = pd.read_parquet(ROOT / "data/features/unified/cicids2017_unified_test.parquet",
                           columns=["Label"])

    picks = []
    for label, n_avail in test["Label"].value_counts().items():
        n = BENIGN_N if label == "BENIGN" else min(PER_CATEGORY_CAP, int(n_avail))
        idx = test.index[test["Label"] == label].to_numpy()
        picks.append(rng.choice(idx, size=n, replace=False))
    pos_idx = np.sort(np.concatenate(picks))

    print(f"[flows] sampled {len(pos_idx)} rows", flush=True)

    # Physical units come from the source, not the scaled matrix. Same split, same seed,
    # indexed positionally -- valid because val/test get no SMOTE and are written in split
    # order with the index reset.
    print("[flows] re-deriving the unified test split for raw values ...", flush=True)
    _, _, raw_test = BUP.split_population(pd.read_parquet(BUP.SOURCE))
    assert len(raw_test) == len(test), "re-derived split does not match the feature matrix"
    rows = raw_test.iloc[pos_idx].reset_index(drop=True)
    assert (rows["Label"].to_numpy() == test.iloc[pos_idx]["Label"].to_numpy()).all(), \
        "positional alignment between the raw split and the feature matrix is wrong"

    missing = [c for c in SHOWN_COLUMNS if c not in rows.columns]
    assert not missing, f"columns not present in the source: {missing}"

    with FLOWS_PATH.open("w", encoding="utf-8") as fh:
        for i in range(len(rows)):
            r = rows.iloc[i]
            flow = {}
            for c in SHOWN_COLUMNS:
                v = r[c]
                flow[c] = (None if pd.isna(v)
                           else int(v) if float(v).is_integer() else round(float(v), 4))
            fh.write(json.dumps({
                "flow_id": f"F{i:03d}",
                "true_label": str(r["Label"]),          # ground truth, never shown to the model
                "test_row_index": int(pos_idx[i]),
                "flow": flow,
            }, ensure_ascii=False) + "\n")
    counts = Counter(str(x) for x in rows["Label"])
    print(f"[flows] wrote {len(rows)} flows to {FLOWS_PATH}", flush=True)
    print(f"[flows] composition: {dict(counts)}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 (uv): ask the LLM to classify
# ---------------------------------------------------------------------------
ATTACK_TYPES = [
    "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "Other",
]

PROMPT = (
    "คุณเป็นนักวิเคราะห์ SOC กำลังตรวจสอบบันทึกการไหลของข้อมูลเครือข่าย (network flow) หนึ่งรายการ\n"
    "จงตัดสินจากค่าในบันทึกนี้เท่านั้นว่าเป็นการโจมตีหรือการจราจรปกติ\n"
    "ตอบเป็น JSON เท่านั้น ห้ามอธิบายนอกโครงสร้าง\n"
    "  is_attack   : true ถ้าเป็นการโจมตี, false ถ้าเป็นการจราจรปกติ\n"
    f"  attack_type : ถ้า is_attack เป็น true ให้เลือกหนึ่งค่าจาก {ATTACK_TYPES} "
    "ถ้าเป็น false ให้ใส่ null\n"
    "  reason      : เหตุผลสั้น ๆ ภาษาไทย อ้างอิงค่าที่เห็นในบันทึก\n"
    "หน่วยของค่า: Flow Duration เป็นไมโครวินาที, ความยาวแพ็กเก็ตเป็นไบต์, "
    "conn_count/uniq_dport_count เป็นจำนวนนับต่อไอพีต้นทางในหน้าต่างเวลา 1 หรือ 5 นาที\n"
    "FLOW={flow}"
)


def _schema():
    from pydantic import BaseModel, ConfigDict

    class LlmVerdict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        is_attack: bool
        attack_type: str | None = None
        reason: str

    return LlmVerdict


def run_llm(limit: int | None) -> None:
    sys.path.insert(0, str(ROOT))
    from src.llm.langchain_client import build_chat_model
    from src.llm.llm_explain import load_env_config

    flows = [json.loads(l) for l in FLOWS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["flow_id"]] = rec
    if done:
        print(f"[llm] resuming: {len(done)} flows already scored", flush=True)

    todo = [f for f in flows if f["flow_id"] not in done]
    if limit:
        todo = todo[:limit]

    config = load_env_config()
    schema = _schema()
    model = build_chat_model(config).with_structured_output(schema, method="json_schema")

    with RAW_PATH.open("a", encoding="utf-8") as fh:
        for i, f in enumerate(todo, 1):
            t0 = time.perf_counter()
            rec: dict[str, Any] = {
                "flow_id": f["flow_id"], "true_label": f["true_label"],
                "test_row_index": f["test_row_index"],
            }
            try:
                out = model.invoke(PROMPT.format(flow=json.dumps(f["flow"], ensure_ascii=False)))
                v = schema.model_validate(out).model_dump()
                rec.update({"status": "ok", "verdict": v})
                mark = f"attack={v['is_attack']} type={v['attack_type']}"
            except Exception as exc:                      # noqa: BLE001 - recorded, not hidden
                rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "verdict": None})
                mark = "ERROR"
            rec["wall_s"] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {f['flow_id']} {f['true_label']:26s} "
                  f"{rec['wall_s']:5.1f}s {mark}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: score
# ---------------------------------------------------------------------------
def summarize() -> dict[str, Any]:
    recs = [json.loads(l) for l in RAW_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]

    tp = sum(1 for r in ok if r["true_label"] != "BENIGN" and r["verdict"]["is_attack"])
    fn = sum(1 for r in ok if r["true_label"] != "BENIGN" and not r["verdict"]["is_attack"])
    fp = sum(1 for r in ok if r["true_label"] == "BENIGN" and r["verdict"]["is_attack"])
    tn = sum(1 for r in ok if r["true_label"] == "BENIGN" and not r["verdict"]["is_attack"])

    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in ok:
        if r["true_label"] == "BENIGN":
            continue
        per_cat[r["true_label"]][1] += 1
        if r["verdict"]["is_attack"]:
            per_cat[r["true_label"]][0] += 1

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "n_total": len(recs), "n_ok": len(ok), "n_error": len(recs) - len(ok),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "fpr": fpr,
        "per_category": {k: {"detected": v[0], "n": v[1], "recall": v[0] / v[1]}
                         for k, v in sorted(per_cat.items())},
        "mean_wall_s": round(sum(r["wall_s"] for r in ok) / len(ok), 2) if ok else 0.0,
    }


def write_report(s: dict[str, Any]) -> None:
    bar = "=" * 74
    L = [bar, "LLM-ONLY DETECTION BASELINE (4-way ablation arm)", bar,
         "Command: uv run python -m src.evaluation.eval_llm_only_baseline", "",
         bar, "1. WHAT THIS ARM IS", bar,
         "The Thai LLM is given ONE network flow in physical units and asked to judge it.",
         "No ML output reaches the prompt: no RF predicted label, no fused score, no anomaly",
         "score. Including any of them would make this a measurement of the ensemble rather",
         "than of the LLM. The true label is withheld too.",
         f"Fields shown to the model: {len(SHOWN_COLUMNS)} flow statistics in physical units",
         "(the 94 model features are MinMax-scaled and carry no meaning a language model can",
         "reason about).", "",
         bar, "2. SAMPLE", bar,
         f"Scored: {s['n_ok']} flows ({s['n_error']} errors out of {s['n_total']}).",
         f"Attack rows {s['tp'] + s['fn']}, benign rows {s['fp'] + s['tn']}.",
         "Deliberately balanced: estimating a false-positive rate needs benign rows, and",
         "running the full 504,193-row test split through an 8B model is not feasible.", "",
         bar, "3. RESULT", bar,
         f"  Recall : {s['recall']:.4f}   ({s['tp']} of {s['tp'] + s['fn']} attacks detected)",
         f"  FPR    : {s['fpr']:.4f}   ({s['fp']} of {s['fp'] + s['tn']} benign flows flagged)",
         f"  Mean latency: {s['mean_wall_s']} s/flow", "",
         "PRECISION IS DELIBERATELY NOT REPORTED HERE. It depends on class prevalence, and",
         "this arm's sample is balanced while the SIEM and hybrid arms are measured on the",
         "natural-prevalence test split. A precision column would invite a comparison that is",
         "not valid. Recall and FPR do not depend on prevalence and are the figures to compare.", "",
         bar, "4. PER ATTACK CATEGORY", bar,
         f"  {'category':28s} {'detected':>9s} {'n':>5s} {'recall':>8s}"]
    for cat, v in s["per_category"].items():
        L.append(f"  {cat:28s} {v['detected']:>9d} {v['n']:>5d} {v['recall']:>8.3f}")
    L += ["", "  Per-category counts are 2-7 flows, so these are descriptive only. A single",
          "  flow moves a category's recall by 0.14 or more.", "",
          bar, "5. HOW TO USE THIS IN THE 4-WAY ABLATION", bar,
          "The four arms do not all measure the same thing, and the table must say so:",
          "  rule-based SIEM : detection, natural prevalence, full test split",
          "  LLM-only        : detection, balanced subsample -- compare recall/FPR only",
          "  ML-only         : detection -- IDENTICAL to the hybrid, because the LLM layer",
          "                    writes explanations and does not touch the decision",
          "  hybrid          : detection, natural prevalence, full test split",
          "So ML-only and hybrid differ only in ANALYST OUTCOME, which is the A/B triage-time",
          "study (RQ4), not a detection metric. Presenting all four in one detection table",
          "would print the same row twice and imply a comparison that does not exist.", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def write_metrics(s: dict[str, Any]) -> None:
    import csv
    with METRICS_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# LLM-only detection baseline. Balanced subsample: compare recall/FPR,"])
        w.writerow(["# NOT precision -- the other ablation arms use the natural-prevalence split."])
        w.writerow(["metric", "value", "num", "den"])
        w.writerow(["recall", f"{s['recall']:.6f}", s["tp"], s["tp"] + s["fn"]])
        w.writerow(["fpr", f"{s['fpr']:.6f}", s["fp"], s["fp"] + s["tn"]])
        w.writerow(["errors", s["n_error"], s["n_error"], s["n_total"]])
        w.writerow(["mean_wall_s", s["mean_wall_s"], "", ""])
        w.writerow([])
        w.writerow(["# per attack category (descriptive only, n = 2-7 per row)"])
        w.writerow(["category", "detected", "n", "recall"])
        for cat, v in s["per_category"].items():
            w.writerow([cat, v["detected"], v["n"], f"{v['recall']:.4f}"])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["flows", "all"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--rebuild-flows", action="store_true")
    args = ap.parse_args()

    if args.stage == "flows":
        build_flows()
        return 0
    if args.rescore:
        s = summarize()
        write_metrics(s)
        write_report(s)
        return 0
    if args.rebuild_flows or not FLOWS_PATH.exists():
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        subprocess.run([ML_PYTHON, "-m", "src.evaluation.eval_llm_only_baseline",
                        "--stage", "flows"], cwd=ROOT, env=env, check=True)
    run_llm(args.limit)
    s = summarize()
    write_metrics(s)
    write_report(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
