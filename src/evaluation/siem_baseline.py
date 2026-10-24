"""Rule-based SIEM baseline on CICIDS-2017 (the 'rule-based SIEM' arm of the 4-way ablation).

Every rule is one row of RULES. Each condition compares one feature column to a
threshold that is derived, never hand-written: the threshold is a percentile of
that column's distribution over BENIGN rows of the TRAIN split, applied unchanged
to test. Features are MinMax-scaled to train, so raw packet counts are meaningless
here; a percentile of the scaled column is the same cut as the percentile of the raw one.

Run from the repo root: python -m src.evaluation.siem_baseline
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/features/behavioral/cicids2017_behavioral_features_train.parquet"
TEST_PATH = ROOT / "data/features/behavioral/cicids2017_behavioral_features_test.parquet"
MITRE_PATH = ROOT / "config/mitre_mapping.csv"
METRICS_PATH = ROOT / "reports/siem_baseline_metrics.csv"
REPORT_PATH = ROOT / "reports/siem_baseline_report.txt"

BENIGN = "BENIGN"
OPS = {">=": operator.ge, "<=": operator.le}


@dataclass(frozen=True)
class Condition:
    column: str
    op: str          # ">=" fires at or above the threshold, "<=" at or below
    quantile: float  # percentile of the train-BENIGN distribution used as the threshold


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    technique_id: str
    conditions: tuple[Condition, ...]  # AND-ed; empty only when implementable is False
    targets: tuple[str, ...]           # attack categories the rule is meant to catch
    implementable: bool = True
    reason: str = ""                   # why not implementable


RULES: tuple[Rule, ...] = (
    Rule("R1", "Port scan", "T1046",
         (Condition("uniq_dport_count_1m", ">=", 0.99), Condition("conn_count_1m", ">=", 0.95)),
         ("PortScan",)),
    Rule("R2", "Brute force (repeated same-port logins)", "T1110.001",
         (Condition("repeat_ratio_5m", ">=", 0.99), Condition("conn_count_5m", ">=", 0.90)),
         ("FTP-Patator", "SSH-Patator", "Web Attack - Brute Force")),
    Rule("R3", "Connection flood", "T1498",
         (Condition("conn_count_1m", ">=", 0.99),),
         ("DDoS", "DoS Hulk", "DoS GoldenEye")),
    Rule("R4", "Slow-rate DoS", "T1499.002",
         (Condition("Flow Duration", ">=", 0.99), Condition("Flow Packets/s", "<=", 0.25)),
         ("DoS slowloris", "DoS Slowhttptest")),
    Rule("R5", "Repeated-target beaconing", "T1071.001",
         (Condition("repeat_ratio_5m", ">=", 0.99), Condition("conn_count_5m", "<=", 0.50)),
         ("Bot",)),
    Rule("R6", "Distributed credential stuffing", "T1110",
         (), ("Web Attack - Brute Force",), False,
         "needs per-destination aggregation across many source IPs; every behavioral column is per source IP"),
    Rule("R7", "Web exploit signature (XSS, SQLi, Heartbleed)", "T1190",
         (), ("Web Attack - XSS", "Web Attack - Sql Injection", "Heartbleed"), False,
         "needs payload or URL inspection; CICFlowMeter flow statistics carry no payload content"),
)


def implementable(rules=RULES) -> list[Rule]:
    return [r for r in rules if r.implementable]


def derive_thresholds(train: pd.DataFrame, rules=RULES) -> dict[tuple[str, str], float]:
    """Threshold per (rule_id, column), read from train BENIGN rows only."""
    benign = train[train["Label"] == BENIGN]
    return {
        (r.rule_id, c.column): float(benign[c.column].quantile(c.quantile))
        for r in implementable(rules) for c in r.conditions
    }


def apply_rule(df: pd.DataFrame, rule: Rule, thresholds: dict[tuple[str, str], float]) -> np.ndarray:
    fired = np.ones(len(df), dtype=bool)
    for c in rule.conditions:
        fired &= OPS[c.op](df[c.column].to_numpy(), thresholds[(rule.rule_id, c.column)])
    return fired


def evaluate(test: pd.DataFrame, thresholds, rules=RULES):
    """Return (rule hit matrix, y_true) for the implementable rules."""
    hits = pd.DataFrame({r.rule_id: apply_rule(test, r, thresholds) for r in implementable(rules)})
    y_true = (test["Label"] != BENIGN).to_numpy()
    return hits, y_true


def binary_metrics(pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def per_category_recall(pred: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for cat in sorted(set(labels) - {BENIGN}):
        mask = (labels == cat).to_numpy()
        rows.append({"category": cat, "n_rows": int(mask.sum()), "recall": float(pred[mask].mean())})
    return pd.DataFrame(rows).sort_values("recall").reset_index(drop=True)


def _fmt(thresholds, rule):
    return " AND ".join(
        f"{c.column} {c.op} {thresholds[(rule.rule_id, c.column)]:.6f} (train-BENIGN p{c.quantile * 100:g})"
        for c in rule.conditions)


def write_outputs(test, thresholds, hits, y_true, techniques, rules=RULES):
    labels = test["Label"].reset_index(drop=True)
    siem = hits.any(axis=1).to_numpy()
    overall = binary_metrics(siem, y_true)
    cat = per_category_recall(siem, labels)

    rule_rows = []
    for r in implementable(rules):
        fired = hits[r.rule_id].to_numpy()
        m = binary_metrics(fired, y_true)
        in_target = labels.isin(r.targets).to_numpy()
        m.update(rule_id=r.rule_id, name=r.name, technique_id=r.technique_id,
                 target_recall=float(fired[in_target].mean()) if in_target.any() else float("nan"))
        rule_rows.append(m)
    rule_df = pd.DataFrame(rule_rows)[
        ["rule_id", "name", "technique_id", "precision", "recall", "f1", "fpr", "target_recall", "tp", "fp"]]

    cat_rule = pd.DataFrame({
        r.rule_id: [float(hits[r.rule_id].to_numpy()[(labels == c).to_numpy()].mean()) for c in cat["category"]]
        for r in implementable(rules)})
    cat_rule.insert(0, "category", cat["category"])

    with open(METRICS_PATH, "w", newline="") as f:
        f.write("# overall binary metrics: any implementable rule fires = alert (test split, thresholds from train BENIGN only)\n")
        pd.DataFrame([{k: overall[k] for k in ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")}]).to_csv(f, index=False)
        f.write("\n# per-rule binary metrics; target_recall = recall over the categories the rule is meant to catch\n")
        rule_df.to_csv(f, index=False)
        f.write("\n# per-category recall of the combined SIEM (all 14 attack categories)\n")
        cat.to_csv(f, index=False)
        f.write("\n# per-category recall by individual rule\n")
        cat_rule.to_csv(f, index=False)

    bar = "=" * 70
    L = [bar, "RULE-BASED SIEM BASELINE (CICIDS-2017, behavioral test split)", bar,
         "Command: python -m src.evaluation.siem_baseline", "",
         bar, "1. SCOPE", bar,
         f"Test rows: {len(test):,} ({int(y_true.sum()):,} attack, {int((~y_true).sum()):,} BENIGN).",
         "Each rule is one row of RULES in src/evaluation/siem_baseline.py. The SIEM alerts when any",
         "implementable rule fires. Rules and percentiles were fixed before any test number was seen",
         "and were not tuned afterwards.", "",
         bar, "2. THRESHOLD DERIVATION", bar,
         "Features in the parquet are MinMax-scaled to the train split, so raw packet-count thresholds are",
         "meaningless. Every threshold is a percentile of that column over BENIGN rows of",
         "cicids2017_behavioral_features_train.parquet, computed by derive_thresholds(train). Nothing is",
         "read from the test split except to score it. Percentiles are a-priori analyst choices (p99 for",
         "abnormally high, p25 for abnormally low rate), not searched.", ""]
    for r in implementable(rules):
        L.append(f"{r.rule_id} {r.name} [{r.technique_id} {techniques.get(r.technique_id, '?')}]")
        L.append(f"    {_fmt(thresholds, r)}")
    L += ["", bar, "3. OVERALL (combined SIEM)", bar,
          f"precision {overall['precision']:.4f}  recall {overall['recall']:.4f}  F1 {overall['f1']:.4f}  FPR {overall['fpr']:.4f}",
          f"TP {overall['tp']:,}  FP {overall['fp']:,}  FN {overall['fn']:,}  TN {overall['tn']:,}",
          "Reference: the ML ensemble reaches F1 0.9865 on this same test split.", "",
          bar, "4. PER RULE", bar, rule_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
          bar, "5. PER-CATEGORY RECALL (combined SIEM, all 14 attack categories)", bar,
          cat.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
          bar, "6. NOT IMPLEMENTABLE FROM AVAILABLE COLUMNS", bar]
    for r in rules:
        if not r.implementable:
            L.append(f"{r.rule_id} {r.name} [{r.technique_id}]: {r.reason}. "
                     f"Targets: {', '.join(r.targets)}. Recorded, not faked.")
    L += ["", bar, "7. WHAT EACH RULE ACTUALLY CATCHES VS ITS INTENT (categories with rule recall >= 0.5)", bar]
    for r in implementable(rules):
        caught = [c for c, v in zip(cat_rule["category"], cat_rule[r.rule_id]) if v >= 0.5]
        L.append(f"{r.rule_id} intended: {', '.join(r.targets)} | actually catches: {', '.join(caught) or 'nothing'}")
    L += ["", bar, "8. CAVEATS", bar,
          "Window features are computed per source IP over the whole capture, so BENIGN heavy talkers",
          "(servers, scanners-by-design) trip volume rules; this is the realistic cost of static rules.",
          "The test set carries natural class prevalence, so precision depends on it; recall and FPR do not.", ""]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    return overall, rule_df, cat


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    techniques = (pd.read_csv(MITRE_PATH).drop_duplicates("technique_id")
                  .set_index("technique_id")["technique_name"].to_dict())
    thresholds = derive_thresholds(train)
    del train
    hits, y_true = evaluate(test, thresholds)
    write_outputs(test, thresholds, hits, y_true, techniques)
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
