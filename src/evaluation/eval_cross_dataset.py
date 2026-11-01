"""M1c: cross-dataset generalizability, CICIDS-2017 <-> CIC-UNSW-NB15.

THE QUESTION. Does a model trained on one campus-traffic dataset detect attacks in another?
That is the generalizability claim a thesis makes implicitly whenever it reports one dataset's
numbers, and it is the claim CICIDS-2017's saturated in-dataset metrics cannot support.

THREE NUMBERS, NOT TWO. A cross-dataset score on its own is uninterpretable: a drop could mean
the model fails to generalize, or merely that the shared feature set is weaker than the full
one. So each direction reports:

  1. source in-dataset, FULL feature set   -- what the model can do with everything it has
  2. source in-dataset, 76 SHARED features -- the cost of dropping to the common columns
  3. source -> target, 76 shared features  -- the generalization result

Without (2) you cannot tell (3)'s shortfall from the price of (1)->(2). That distinction is the
whole reason this script trains more than one model.

RANDOM FOREST ONLY, and that is a limitation, not a choice. The 12 behavioral columns derive
from `Source IP` and `Timestamp`, which CIC-UNSW-NB15's `Data.csv` does not carry, so neither
Isolation Forest nor the Autoencoder can be scored on the target at all. M1c therefore measures
the supervised member in isolation and says nothing about the deployed cascade's
generalizability. Do not report it as an ensemble result.

BINARY, BOTH SIDES. CICIDS-2017 carries 15 string classes and CIC-UNSW-NB15 a 0/1 integer, and
their attack taxonomies do not align (DoS Hulk is not Exploits). The only honest common target
is attack-vs-benign, so every model here is trained and scored binary. A multi-class
cross-dataset score would require a category mapping that does not exist.

SCALING. `build_cross_dataset_eval.py` fits sentinel imputation and the MinMaxScaler on the
SOURCE train split and applies that transform to the target's raw features. The source's own
`*_features_*.parquet` were scaled by a scaler fit on the same source train, and MinMax is
per-column independent, so restricting them to the 76 shared columns yields exactly the values
a 76-column scaler would have produced. The two sides are therefore on one axis.

Run from the repo root with the Anaconda interpreter:
    python -m src.evaluation.eval_cross_dataset
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data/features"
CROSS = FEATURES / "crossdataset_eval"
METRICS_PATH = ROOT / "reports/cross_dataset_m1c_metrics.csv"
REPORT_PATH = ROOT / "reports/cross_dataset_m1c_report.txt"

RANDOM_STATE = 42
RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = "balanced"

REFERENCE_COLS = ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"]
NON_FEATURE_COLS = REFERENCE_COLS + [
    "attack_cat", "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded",
    "is_known_technique", "row_index", "orig_row_index",
]

DIRECTIONS = [("cicids2017", "unswnb15"), ("unswnb15", "cicids2017")]

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 76)
    log(title)
    log("=" * 76)


def binary_labels(label: pd.Series) -> np.ndarray:
    """attack=1 / benign=0, for either dataset's label convention.

    CICIDS-2017 uses the string "BENIGN"; CIC-UNSW-NB15 uses 0. Handling both here keeps the
    convention in one place rather than scattered through the directions.
    """
    if pd.api.types.is_numeric_dtype(label):
        return (label.to_numpy() != 0).astype(int)
    return (label.to_numpy() != "BENIGN").astype(int)


def feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(frame[c])]


def metrics(y_true, pred, score) -> dict:
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
    }


def fit_rf(X, y, label: str):
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X, y)
    log(f"    trained {label}: {X.shape[0]:,} rows x {X.shape[1]} features "
        f"in {time.perf_counter() - t0:.0f}s")
    return rf


def score_rf(rf, X):
    proba = rf.predict_proba(X)
    attack_idx = list(rf.classes_).index(1)
    return proba[:, attack_idx], rf.predict(X)


def run_direction(source: str, target: str, rows: list) -> None:
    section(f"{source} -> {target}")
    train = pd.read_parquet(FEATURES / f"{source}_features_train.parquet")
    test = pd.read_parquet(FEATURES / f"{source}_features_test.parquet")
    cross = pd.read_parquet(CROSS / f"{source}_to_{target}_test.parquet")

    full_cols = feature_cols(train)
    shared_cols = feature_cols(cross)
    assert set(shared_cols) <= set(full_cols), \
        "shared columns are not a subset of the source's own features"
    log(f"  source train {len(train):,} rows | full features {len(full_cols)} | "
        f"shared features {len(shared_cols)}")
    log(f"  target test  {len(cross):,} rows")
    dropped = sorted(set(full_cols) - set(shared_cols))
    log(f"  dropped to reach the shared set ({len(dropped)}): {dropped}")

    y_train = binary_labels(train["Label"])
    y_test = binary_labels(test["Label"])
    y_cross = binary_labels(cross["Label"])
    log(f"  target attack prevalence {y_cross.mean() * 100:.2f}% "
        f"vs source test {y_test.mean() * 100:.2f}%")

    # (1) in-dataset, full feature set
    rf_full = fit_rf(train[full_cols].to_numpy(), y_train, f"{source} full ({len(full_cols)}f)")
    score, pred = score_rf(rf_full, test[full_cols].to_numpy())
    m = metrics(y_test, pred, score)
    rows.append({"direction": f"{source}->{target}", "model": f"{source} RF, full {len(full_cols)}f",
                 "evaluated_on": f"{source} test (in-dataset)", "n": len(test), **m})
    log(f"    in-dataset, full:   P {m['precision']:.4f} R {m['recall']:.4f} "
        f"F1 {m['f1']:.4f} AUC {m['auc_roc']:.4f} FPR {m['fpr']:.4f}")

    # (2) in-dataset, shared feature set -- the control that makes (3) readable
    rf_shared = fit_rf(train[shared_cols].to_numpy(), y_train,
                       f"{source} shared ({len(shared_cols)}f)")
    score, pred = score_rf(rf_shared, test[shared_cols].to_numpy())
    m_shared = metrics(y_test, pred, score)
    rows.append({"direction": f"{source}->{target}",
                 "model": f"{source} RF, shared {len(shared_cols)}f",
                 "evaluated_on": f"{source} test (in-dataset)", "n": len(test), **m_shared})
    log(f"    in-dataset, shared: P {m_shared['precision']:.4f} R {m_shared['recall']:.4f} "
        f"F1 {m_shared['f1']:.4f} AUC {m_shared['auc_roc']:.4f} FPR {m_shared['fpr']:.4f}")

    # (3) cross-dataset
    score, pred = score_rf(rf_shared, cross[shared_cols].to_numpy())
    m_cross = metrics(y_cross, pred, score)
    rows.append({"direction": f"{source}->{target}",
                 "model": f"{source} RF, shared {len(shared_cols)}f",
                 "evaluated_on": f"{target} test (cross-dataset)", "n": len(cross), **m_cross})
    log(f"    cross-dataset:      P {m_cross['precision']:.4f} R {m_cross['recall']:.4f} "
        f"F1 {m_cross['f1']:.4f} AUC {m_cross['auc_roc']:.4f} FPR {m_cross['fpr']:.4f}")

    log("")
    log(f"  cost of dropping to the shared set: F1 {m['f1']:.4f} -> {m_shared['f1']:.4f} "
        f"({m_shared['f1'] - m['f1']:+.4f})")
    log(f"  generalization gap:                 F1 {m_shared['f1']:.4f} -> {m_cross['f1']:.4f} "
        f"({m_cross['f1'] - m_shared['f1']:+.4f})")


def main() -> None:
    section("M1c CROSS-DATASET GENERALIZABILITY (Random Forest only)")
    log("  Random Forest is the only member that can take part: the 12 behavioral columns")
    log("  derive from Source IP and Timestamp, which CIC-UNSW-NB15's Data.csv does not carry,")
    log("  so IF and AE cannot be scored on the target at all. This measures the supervised")
    log("  member in isolation and says nothing about the deployed cascade.")
    log("  Binary attack-vs-benign on both sides: the two attack taxonomies do not align, so a")
    log("  multi-class cross-dataset score would need a category mapping that does not exist.")
    log("")

    rows: list[dict] = []
    for source, target in DIRECTIONS:
        run_direction(source, target, rows)
        log("")

    section("SUMMARY")
    table = pd.DataFrame(rows)
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    table.to_csv(METRICS_PATH, index=False)

    section("HOW TO READ THIS")
    log("  Compare the three rows of a direction in order. The first-to-second step is the price")
    log("  of restricting to the shared columns; the second-to-third step is generalization.")
    log("  Attributing the whole drop to generalization would overstate it, which is exactly the")
    log("  mistake the middle row exists to prevent.")
    log("  Prevalence differs between the two test sets, so precision is not comparable across")
    log("  directions at face value. Recall, FPR and AUC are.")
    log("  A cross-dataset collapse is a finding, not a failure of the experiment: these are two")
    log("  different testbeds with different attack tooling, captured years apart. What it")
    log("  establishes is the boundary of the single-dataset claim.")

    REPORT_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {METRICS_PATH}")
    log(f"[saved] {REPORT_PATH}")
    log("=== M1c COMPLETE ===")


if __name__ == "__main__":
    main()
