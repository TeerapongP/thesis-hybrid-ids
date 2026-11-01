"""Does Random Forest's zero-day recall come from the behavioral features or the population?

The question this answers. On the old joined population, RF's nested LOAO F1 was 0.8327 with
recall 0.7705, and the rf_ae_or rule beat it by 9.3pp. On the unified population RF solo
reaches F1 0.9737 with recall 0.9593, which erases the fusion rule's entire margin -- the
deployed OR now scores 0.9724, BELOW RF alone. That reversal decides whether the thesis keeps
a three-model ensemble or has to admit a single model is enough, so the cause matters.

Two things changed at the same time, and they are confounded:

  1. RF gained the 12 time-windowed behavioral columns. It had never had them: Option A
     deliberately left RF on the 80-feature set, on the stated grounds that behavioral
     features could not fix a supervised-generalization limit. That conclusion came from a
     prototype evaluation, not from measuring RF with the columns.
  2. The population changed -- 100% coverage instead of 80.36%, a different split, different
     rows, and ~297k BENIGN rows that the row-hash join had dropped.

This script removes exactly one of the two. It runs the identical LOAO procedure on the
identical unified folds with the 12 behavioral columns DROPPED, giving 82 features instead of
94. Everything else -- folds, seed, imputation, scaler, capped SMOTE, hyperparameters -- is
imported from fusion_loao_unified so it cannot drift.

Reading the result:
  recall falls back toward 0.77  -> the behavioral columns are the cause, Option A's
                                    conclusion about RF was wrong, and the ensemble's
                                    collapsed margin is a real consequence of fixing RF.
  recall stays near 0.96         -> the population is the cause, and the earlier 0.7705
                                    was depressed by the join's missing BENIGN rows rather
                                    than by RF lacking features.

Only Random Forest is trained. IF and AE are not re-run: their scores are unchanged by RF's
feature set, and spending 20 minutes a fold to reproduce identical numbers would buy nothing.

Run from the repo root with the Anaconda interpreter:
    python -m src.evaluation.rf_behavioral_ablation_loao
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.fusion_loao_trainnorm import (
    CATEGORIES,
    RANDOM_STATE,
    RF_CLASS_WEIGHT,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_SENTINEL_COLS,
    add_missing_flags_and_impute_fold,
    apply_smote_loao,
    compute_binary_metrics,
    slugify_category,
)
from src.evaluation.fusion_loao_unified import (
    LOAO_DIR,
    select_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SCORES_DIR = REPORTS / "rf_behavioral_ablation_scores"
RESULTS_CSV = REPORTS / "rf_behavioral_ablation_loao_results.csv"
REPORT = REPORTS / "rf_behavioral_ablation_loao_report.txt"

BEHAVIORAL_SUFFIXES = ("_1m", "_3m", "_5m")
EXPECTED_FEATURES_WITHOUT_BEHAVIORAL = 82

# The reference numbers this ablation is interpreted against. Both are nested LOAO, pooled.
REFERENCE = {
    "unified, 94 features (with behavioral)": {"f1": 0.9737, "recall": 0.9593, "fpr": 0.0011},
    "old joined population, 80 features": {"f1": 0.8327, "recall": 0.7705, "fpr": 0.0106},
}

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 74)
    log(title)
    log("=" * 74)


def run_fold(category: str) -> tuple[dict, pd.DataFrame]:
    slug = slugify_category(category)
    train_fold = pd.read_parquet(LOAO_DIR / f"loao_{slug}_train.parquet")
    test_fold = pd.read_parquet(LOAO_DIR / f"loao_{slug}_zeroday_test.parquet")
    assert category not in set(train_fold["Label"]), \
        f"{category} leaked into its own LOAO training fold"
    y_true = (test_fold["Label"] != "BENIGN").astype(int).to_numpy()

    train_fold, test_fold = add_missing_flags_and_impute_fold(train_fold, test_fold)
    all_features = select_feature_columns(train_fold)
    behavioral = [c for c in all_features if c.endswith(BEHAVIORAL_SUFFIXES)]
    assert len(behavioral) == 12, f"expected 12 behavioral columns, found {len(behavioral)}"
    feature_cols = [c for c in all_features if c not in set(behavioral)]
    assert len(feature_cols) == EXPECTED_FEATURES_WITHOUT_BEHAVIORAL, \
        f"expected {EXPECTED_FEATURES_WITHOUT_BEHAVIORAL} features, got {len(feature_cols)}"

    flag_cols = [f"{c}_missing" for c in RF_SENTINEL_COLS]
    scale_cols = [c for c in feature_cols if c not in flag_cols]
    scaler = MinMaxScaler().fit(train_fold[scale_cols])
    train_fold[scale_cols] = scaler.transform(train_fold[scale_cols])
    test_fold[scale_cols] = scaler.transform(test_fold[scale_cols])

    X_train = train_fold[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_fold[feature_cols].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    X_res, y_res = apply_smote_loao(X_train, train_fold["Label"].to_numpy())
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_res, y_res)
    benign_idx = list(rf.classes_).index("BENIGN")
    rf_norm = 1.0 - rf.predict_proba(X_test)[:, benign_idx]
    argmax_recall, argmax_fpr = compute_binary_metrics(
        (rf.predict(X_test) != "BENIGN").astype(int), y_true
    )
    seconds = time.perf_counter() - t0
    log(f"    RF (82 feat)  argmax recall {argmax_recall:.4f}  fpr {argmax_fpr:.4f}   ({seconds:.0f}s)")

    result = {
        "category": category,
        "feature_count": len(feature_cols),
        "behavioral_columns_dropped": len(behavioral),
        "n_test_rows": int(len(test_fold)),
        "n_attack_rows": int(y_true.sum()),
        "rf_argmax_recall": argmax_recall,
        "rf_argmax_fpr": argmax_fpr,
        "rf_fit_seconds": seconds,
    }
    # Pooled nested threshold selection needs the raw scores, exactly as the main run does;
    # argmax alone would understate RF and is not comparable to the swept figures.
    scores = pd.DataFrame({"category": category, "rf_norm": rf_norm, "y_true": y_true})
    return result, scores


def nested_sweep(pooled: pd.DataFrame) -> dict:
    """Thresholds chosen on the other nine categories, applied to the held-out one."""
    grid = np.concatenate([np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]),
                           np.linspace(0.05, 0.95, 19)])
    cat = pooled["category"].to_numpy()
    score = pooled["rf_norm"].to_numpy()
    y = pooled["y_true"].to_numpy()

    def prf1(pred, yy):
        tp = int(np.sum((pred == 1) & (yy == 1))); fp = int(np.sum((pred == 1) & (yy == 0)))
        fn = int(np.sum((pred == 0) & (yy == 1))); tn = int(np.sum((pred == 0) & (yy == 0)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0.0), (fp / (fp + tn) if (fp + tn) else float("nan"))

    pred = np.zeros(len(pooled), dtype=int)
    per_category = []
    for c in CATEGORIES:
        held = cat == c
        best = (-1.0, None)
        for t in grid:
            f1 = prf1((score[~held] > t).astype(int), y[~held])[2]
            if f1 > best[0]:
                best = (f1, float(t))
        chosen = best[1]
        pred[held] = (score[held] > chosen).astype(int)
        p, r, f1, fpr = prf1(pred[held], y[held])
        per_category.append({"category": c, "selected_threshold": chosen,
                             "n_attack_rows": int(y[held].sum()),
                             "precision": p, "recall": r, "f1": f1, "fpr": fpr})
    p, r, f1, fpr = prf1(pred, y)
    return {"pooled": {"precision": p, "recall": r, "f1": f1, "fpr": fpr},
            "per_category": per_category}


def main() -> None:
    section("RF behavioral-column ablation on the unified LOAO folds")
    log("  identical folds, seed, imputation, scaler, SMOTE and hyperparameters as the main run")
    log(f"  the only change: {12} behavioral columns dropped, 94 -> "
        f"{EXPECTED_FEATURES_WITHOUT_BEHAVIORAL} features")
    log("  Isolation Forest and the Autoencoder are not re-run; RF's feature set does not")
    log("  change their scores, so re-deriving them would reproduce identical numbers.")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    results, frames = [], []
    for i, category in enumerate(CATEGORIES, 1):
        log("")
        log(f"[{i}/{len(CATEGORIES)}] {category}")
        cache = SCORES_DIR / f"{slugify_category(category)}.parquet"
        meta = SCORES_DIR / f"{slugify_category(category)}.json"
        if cache.exists() and meta.exists():
            scores = pd.read_parquet(cache)
            result = json.loads(meta.read_text(encoding="utf-8"))
            log("    reused checkpoint from an earlier run")
        else:
            result, scores = run_fold(category)
            scores.to_parquet(cache, index=False)
            meta.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        frames.append(scores)

    section("per-fold argmax results (82 features)")
    df = pd.DataFrame(results)
    log(df[["category", "n_attack_rows", "rf_argmax_recall", "rf_argmax_fpr"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    section("nested threshold selection, pooled -- the comparable number")
    pooled = pd.concat(frames, ignore_index=True)
    swept = nested_sweep(pooled)
    m = swept["pooled"]
    log(f"  pooled rows {len(pooled):,}   attack rows {int(pooled['y_true'].sum()):,}")
    log(f"  RF solo, 82 features:  P {m['precision']:.4f}  R {m['recall']:.4f}  "
        f"F1 {m['f1']:.4f}  FPR {m['fpr']:.4f}")
    log("")
    log("  against the two reference runs (both nested, pooled):")
    for name, ref in REFERENCE.items():
        log(f"    {name:42s} F1 {ref['f1']:.4f}  R {ref['recall']:.4f}  FPR {ref['fpr']:.4f}")

    section("what this says")
    with_beh = REFERENCE["unified, 94 features (with behavioral)"]
    old = REFERENCE["old joined population, 80 features"]
    d_beh = with_beh["recall"] - m["recall"]
    d_old = m["recall"] - old["recall"]
    log(f"  dropping the behavioral columns costs {d_beh:+.4f} recall on the same population.")
    log(f"  the remaining gap to the old joined-population run is {d_old:+.4f} recall.")
    if abs(d_beh) > abs(d_old):
        log("  The behavioral columns account for most of the change: RF's zero-day recall came")
        log("  from the features, so Option A's conclusion that behavioral features could not")
        log("  help RF was wrong, and the fusion rule's lost margin is a real consequence of")
        log("  finally giving RF those columns.")
    else:
        log("  The population accounts for most of the change: RF's zero-day recall came from")
        log("  the rows the row-hash join used to drop, not from the behavioral columns. The")
        log("  earlier 0.7705 was depressed by the join, and Option A's conclusion about RF")
        log("  stands.")
    log("  Either way this is one ablation on one dataset and does not establish which fusion")
    log("  rule to deploy; it establishes what caused the number to move.")

    rows = pd.DataFrame(swept["per_category"])
    out = df.merge(rows, on="category", suffixes=("", "_swept"))
    out.to_csv(RESULTS_CSV, index=False)
    section("per-category, nested selection")
    log(rows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    REPORT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {RESULTS_CSV}")
    log(f"[saved] {REPORT}")
    log("=== RF BEHAVIORAL ABLATION COMPLETE ===")


if __name__ == "__main__":
    main()
