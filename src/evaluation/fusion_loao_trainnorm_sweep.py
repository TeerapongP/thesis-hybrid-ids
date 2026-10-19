"""
fusion_loao_trainnorm_sweep.py

Sweep fusion thresholds on the TRAIN-FOLD-NORMALIZED LOAO scores produced by
`src/evaluation/fusion_loao_trainnorm.py`, and report every candidate at its own
fairly-swept decision threshold.

Why this is a separate script: `fusion_loao_trainnorm.py` deliberately persists only
per-fold scores (`LOAO_METRIC_SCOPE = "component_models_only_no_fusion"`) so thresholds
can be re-swept later without retraining ten folds. This script is that second half.

What it fixes relative to the legacy reports (now suffixed
`*_legacy_test_fold_normalization.csv`): those normalized each fold's IF-v2/AE-v2
zero-day scores with that same test fold's own min/max, so the normalized score of a
test row depended on the rest of the test set. Here `if_norm`/`ae_norm` arrive already
normalized with TRAIN-fold constants, so nothing about the held-out category leaks into
the score scale.

Two selection protocols are reported, and they are not interchangeable:
  * `tuned-on-all`  -- thresholds picked on the pooled scores of all 10 categories, then
    scored on those same categories. Reproduces the legacy procedure's optimism. Use it
    only to compare against the legacy number, never as a headline result.
  * `nested`        -- for each report category, thresholds are picked on the OTHER nine
    categories only, then applied to the held-out one. This is the defensible zero-day
    number: selection never sees the category it is scored on.

Run from the repo root:  python -m src.evaluation.fusion_loao_trainnorm_sweep
"""

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_DIR = REPORTS_DIR / "fusion_layer_loao_trainnorm_checkpoints"
OUT_SWEEP = REPORTS_DIR / "fusion_layer_loao_trainnorm_swept_baseline.csv"
OUT_NESTED = REPORTS_DIR / "fusion_layer_loao_trainnorm_nested_selection.csv"

CATEGORIES = [
    "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
    "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Attack - Brute Force",
]

CONFIDENCE_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 1.0]
DECISION_GRID = np.concatenate([
    np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]),
    np.linspace(0.05, 0.95, 21),
])


# ----------------------------------------------------------------------------
# candidate score functions -- mirror src/modeling/fusion_layer.py exactly
# ----------------------------------------------------------------------------
def score_confidence_weighted(rf, if_, ae, confidence_threshold):
    rf_confidence = np.abs(rf - 0.5) * 2
    return np.where(rf_confidence >= confidence_threshold, rf, (if_ + ae) / 2)


def score_weighted_average(rf, if_, ae, _ct=None):
    return (rf + if_ + ae) / 3


def score_max(rf, if_, ae, _ct=None):
    return np.maximum(np.maximum(rf, if_), ae)


CANDIDATES = {
    "fusion: confidence_weighted": ("confidence", score_confidence_weighted),
    "fusion: weighted_average": ("plain", score_weighted_average),
    "fusion: max_score": ("plain", score_max),
    "RF solo": ("plain", lambda rf, if_, ae, _ct=None: rf),
    "IF-v2 solo": ("plain", lambda rf, if_, ae, _ct=None: if_),
    "AE-v2 solo": ("plain", lambda rf, if_, ae, _ct=None: ae),
}


def prf1(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return precision, recall, f1, fpr


def load_scores():
    """Newest signature directory that holds all 10 categories' score files."""
    if not CHECKPOINT_DIR.exists():
        raise SystemExit(f"no checkpoint dir: {CHECKPOINT_DIR} -- run fusion_loao_trainnorm first")
    sig_dirs = sorted((d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    for d in sig_dirs:
        frames = {}
        for path in sorted(d.glob("*_scores.parquet")):
            df = pd.read_parquet(path)
            cat = df["category"].iloc[0]
            # keep the newest file per category (checkpoint ids are uuid-suffixed)
            if cat not in frames or path.stat().st_mtime > frames[cat][1]:
                frames[cat] = (df, path.stat().st_mtime)
        cats = set(frames)
        if cats >= set(CATEGORIES):
            pooled = pd.concat([frames[c][0] for c in CATEGORIES], ignore_index=True)
            print(f"Using signature dir: {d.name}")
            print(f"Pooled rows: {len(pooled):,} across {len(CATEGORIES)} categories")
            return pooled, d.name
        if cats:
            print(f"skipping {d.name}: only {len(cats)}/{len(CATEGORIES)} categories present")
    raise SystemExit("no signature dir has all 10 category score files yet")


def best_threshold(df, score_fn, kind):
    """Pick (confidence_threshold, decision_threshold) maximizing F1 on `df`."""
    rf, if_, ae = df["rf_norm"].to_numpy(), df["if_norm"].to_numpy(), df["ae_norm"].to_numpy()
    y = df["y_true"].to_numpy()
    grid = CONFIDENCE_GRID if kind == "confidence" else [None]
    best = (-1.0, None, None)
    for ct in grid:
        sc = score_fn(rf, if_, ae, ct)
        for dt in DECISION_GRID:
            f1 = prf1((sc > dt).astype(int), y)[2]
            if f1 > best[0]:
                best = (f1, ct, float(dt))
    return best[1], best[2], best[0]


def main():
    pooled, signature = load_scores()
    y_pooled = pooled["y_true"].to_numpy()

    # ------------------------------------------------------------------
    # 1. tuned-on-all: every candidate at its own pooled-optimal threshold
    # ------------------------------------------------------------------
    sweep_rows = []
    per_cat_rows = []
    for name, (kind, fn) in CANDIDATES.items():
        ct, dt, _ = best_threshold(pooled, fn, kind)
        sc = fn(pooled["rf_norm"].to_numpy(), pooled["if_norm"].to_numpy(),
                pooled["ae_norm"].to_numpy(), ct)
        pred = (sc > dt).astype(int)
        p, r, f1, fpr = prf1(pred, y_pooled)
        sweep_rows.append({
            "candidate": name,
            "selected_confidence_threshold": ct,
            "swept_decision_threshold": dt,
            "precision": p, "recall": r, "f1": f1, "fpr": fpr,
            "auc_roc": roc_auc_score(y_pooled, sc),
        })
        for cat in CATEGORIES:
            mask = (pooled["category"] == cat).to_numpy()
            cp, cr, cf1, cfpr = prf1(pred[mask], y_pooled[mask])
            per_cat_rows.append({
                "candidate": name, "category": cat,
                "n_attack_rows": int(y_pooled[mask].sum()),
                "recall": cr, "fpr": cfpr, "precision": cp, "f1": cf1,
            })
    sweep_df = pd.DataFrame(sweep_rows).sort_values("f1", ascending=False)
    per_cat_df = pd.DataFrame(per_cat_rows)

    # ------------------------------------------------------------------
    # 2. nested: thresholds chosen on the other 9 categories only
    # ------------------------------------------------------------------
    nested_rows = []
    nested_pooled = []
    for name, (kind, fn) in CANDIDATES.items():
        preds = np.zeros(len(pooled), dtype=int)
        scores = np.zeros(len(pooled), dtype=float)
        for cat in CATEGORIES:
            held = (pooled["category"] == cat).to_numpy()
            ct, dt, _ = best_threshold(pooled.loc[~held], fn, kind)
            sub = pooled.loc[held]
            sc = fn(sub["rf_norm"].to_numpy(), sub["if_norm"].to_numpy(), sub["ae_norm"].to_numpy(), ct)
            pred = (sc > dt).astype(int)
            preds[held] = pred
            scores[held] = sc
            cp, cr, cf1, cfpr = prf1(pred, sub["y_true"].to_numpy())
            nested_rows.append({
                "candidate": name, "selection": "nested", "report_category": cat,
                "selected_confidence_threshold": ct, "selected_decision_threshold": dt,
                "n_attack_rows": int(sub["y_true"].sum()),
                "recall": cr, "fpr": cfpr, "precision": cp, "f1": cf1,
            })
        p, r, f1, fpr = prf1(preds, y_pooled)
        nested_pooled.append({
            "candidate": name, "selection": "nested",
            "precision": p, "recall": r, "f1": f1, "fpr": fpr,
        })
        tuned = sweep_df.loc[sweep_df["candidate"] == name].iloc[0]
        nested_pooled.append({
            "candidate": name, "selection": "tuned-on-all",
            "precision": tuned["precision"], "recall": tuned["recall"],
            "f1": tuned["f1"], "fpr": tuned["fpr"],
        })
    nested_pooled_df = pd.DataFrame(nested_pooled).sort_values(["candidate", "selection"])
    nested_df = pd.DataFrame(nested_rows)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    header = [
        "# LOAO threshold sweep on TRAIN-FOLD-NORMALIZED scores (valid replacement for the",
        "# *_legacy_test_fold_normalization.csv reports, which normalized each fold's test",
        "# scores with that same test fold's own min/max).",
        f"# Source checkpoint signature: {signature}",
        f"# Pooled rows: {len(pooled)}; attack rows: {int(y_pooled.sum())}",
        f"# Decision-threshold grid: {len(DECISION_GRID)} values {DECISION_GRID.min():.3f}..{DECISION_GRID.max():.3f}",
        f"# Confidence-threshold grid (confidence_weighted only): {CONFIDENCE_GRID}",
        "# 'tuned-on-all' is optimistic by construction -- cite the nested numbers instead.",
        "#",
    ]
    with open(OUT_SWEEP, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        f.write("# --- pooled: candidate comparison, each at its own pooled-optimal threshold ---\n")
        sweep_df.to_csv(f, index=False)
        f.write("#\n# --- per-category recall/FPR at those same pooled-optimal thresholds ---\n")
        per_cat_df.to_csv(f, index=False)
    print("Saved:", OUT_SWEEP)

    with open(OUT_NESTED, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        f.write("# --- pooled: tuned-on-all vs nested, per candidate ---\n")
        nested_pooled_df.to_csv(f, index=False)
        f.write("#\n# --- per held-out category: thresholds selected on the other 9 ---\n")
        nested_df.to_csv(f, index=False)
    print("Saved:", OUT_NESTED)

    print("\n=== pooled LOAO (train-fold normalization) ===")
    print(nested_pooled_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== per-category recall, nested selection ===")
    piv = nested_df.pivot(index="report_category", columns="candidate", values="recall")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    summary = {
        "source_signature": signature,
        "normalization": "train_fold_fitted",
        "pooled_rows": int(len(pooled)),
        "pooled_attack_rows": int(y_pooled.sum()),
        "nested_pooled_f1": {r["candidate"]: r["f1"] for r in nested_pooled if r["selection"] == "nested"},
        "tuned_on_all_pooled_f1": {r["candidate"]: r["f1"] for r in nested_pooled if r["selection"] == "tuned-on-all"},
    }
    with open(REPORTS_DIR / "fusion_layer_loao_trainnorm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved:", REPORTS_DIR / "fusion_layer_loao_trainnorm_summary.json")


if __name__ == "__main__":
    main()
