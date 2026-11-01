"""Compare the chosen supervised model against alternatives the thesis never tested.

Why this exists. The advisor asked why these three models, given how many exist. The
honest gap in the answer was that no alternative had ever been measured: the choice of
Random Forest rested on convention, not evidence. This measures the obvious supervised
alternatives on exactly the same unified population, same features, same split, so the
comparison is like for like and the answer stops being an assertion.

Scope on purpose. Supervised classifiers only. The unsupervised side already has its own
evidence trail: Isolation Forest and Autoencoder-v2 have a per-category recall correlation
of 0.925 to 0.977, and the LOAO ablation already removed Isolation Forest from the deployed
decision. Adding a third unsupervised detector would repeat an experiment whose answer is
known, so One-Class SVM is left out and that is a decision, not an oversight.

XGBoost and LightGBM run on the GPU where the library supports it. scikit-learn has no GPU
backend at all, which is why Random Forest trains on CPU; that is a library limitation
rather than a configuration choice, and it is worth stating in the write-up because the
wall-clock column otherwise invites the wrong conclusion.

Run from the repo root with the Anaconda interpreter, after train_unified_ensemble.py:
    python -m src.evaluation.baseline_model_comparison
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features/unified"
MODELS = ROOT / "models"
METRICS = ROOT / "reports/baseline_model_comparison.csv"
REPORT = ROOT / "reports/baseline_model_comparison_report.txt"

RANDOM_STATE = 42
NON_FEATURE_COLS = [
    "Label", "technique_id", "technique_name", "tactic", "mapping_confidence",
    "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique",
]

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 70)
    log(title)
    log("=" * 70)


def metrics(y_true, pred, score):
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, score),
    }


def per_category_recall(labels, pred):
    out = {}
    for cat in sorted(set(labels)):
        if cat == "BENIGN":
            continue
        mask = labels == cat
        out[cat] = float(pred[mask].mean())
    return out


def main() -> None:
    section("STEP 0: load the unified population")
    train = pd.read_parquet(FEAT / "cicids2017_unified_train.parquet")
    val = pd.read_parquet(FEAT / "cicids2017_unified_val.parquet")
    feature_cols = [c for c in train.columns
                    if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(train[c])]
    X_train = train[feature_cols].to_numpy(dtype=np.float32)
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    labels_train = train["Label"].to_numpy()
    labels_val = val["Label"].to_numpy()
    y_val = (labels_val != "BENIGN").astype(int)
    classes = sorted(set(labels_train))
    cls_index = {c: i for i, c in enumerate(classes)}
    benign_i = cls_index["BENIGN"]
    y_train_int = np.array([cls_index[c] for c in labels_train], dtype=np.int32)
    log(f"  train {len(train):,}  val {len(val):,}  features {len(feature_cols)}  classes {len(classes)}")
    log("  identical population, features and split for every candidate below")

    rows, recalls = [], {}

    # ── incumbent, already trained ───────────────────────────────────
    section("STEP 1: Random Forest (incumbent, scikit-learn, CPU only)")
    rf_path = MODELS / "random_forest_unified.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        t0 = time.perf_counter()
        proba = rf.predict_proba(X_val.astype(np.float64))
        infer = time.perf_counter() - t0
        bi = list(rf.classes_).index("BENIGN")
        score = 1.0 - proba[:, bi]
        pred = (rf.classes_[proba.argmax(1)] != "BENIGN").astype(int)
        m = metrics(y_val, pred, score)
        rows.append({"model": "RandomForest (sklearn, CPU)", "train_seconds": float("nan"),
                     "infer_seconds": infer, **m})
        recalls["RandomForest"] = per_category_recall(labels_val, pred)
        log(f"  loaded the already-trained model; val F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")
        log("  train time is not comparable here because it was trained by another script")
    else:
        log(f"  SKIPPED: {rf_path.name} not found. Run train_unified_ensemble.py first.")

    # ── XGBoost ──────────────────────────────────────────────────────
    section("STEP 2: XGBoost (GPU)")
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=12, learning_rate=0.3,
        objective="multi:softprob", num_class=len(classes),
        tree_method="hist", device="cuda",
        random_state=RANDOM_STATE, verbosity=0,
    )
    t0 = time.perf_counter()
    xgb_clf.fit(X_train, y_train_int)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = xgb_clf.predict_proba(X_val)
    infer = time.perf_counter() - t0
    score = 1.0 - proba[:, benign_i]
    pred = (proba.argmax(1) != benign_i).astype(int)
    m = metrics(y_val, pred, score)
    rows.append({"model": "XGBoost (GPU)", "train_seconds": train_s, "infer_seconds": infer, **m})
    recalls["XGBoost"] = per_category_recall(labels_val, pred)
    xgb_clf.save_model(MODELS / "xgboost_unified.json")
    log(f"  trained in {train_s:.1f}s   val F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")
    log("  saved models/xgboost_unified.json")

    # ── LightGBM, if available ───────────────────────────────────────
    section("STEP 3: LightGBM (CPU, optional)")
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=12, learning_rate=0.3,
                                     objective="multiclass", num_class=len(classes),
                                     random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        t0 = time.perf_counter()
        lgb_clf.fit(X_train, y_train_int)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = lgb_clf.predict_proba(X_val)
        infer = time.perf_counter() - t0
        score = 1.0 - proba[:, benign_i]
        pred = (proba.argmax(1) != benign_i).astype(int)
        m = metrics(y_val, pred, score)
        rows.append({"model": "LightGBM (CPU)", "train_seconds": train_s, "infer_seconds": infer, **m})
        recalls["LightGBM"] = per_category_recall(labels_val, pred)
        joblib.dump(lgb_clf, MODELS / "lightgbm_unified.pkl")
        log(f"  trained in {train_s:.1f}s   val F1 {m['f1']:.4f}  AUC {m['auc_roc']:.4f}")
    except ImportError:
        log("  SKIPPED: lightgbm is not installed. Not an error; XGBoost already covers")
        log("  the gradient-boosted-tree family for this comparison.")

    # ── results ──────────────────────────────────────────────────────
    section("STEP 4: comparison")
    table = pd.DataFrame(rows)
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    table.to_csv(METRICS, index=False)

    section("STEP 5: per-category recall on val")
    cat_df = pd.DataFrame(recalls)
    # Support belongs next to the recall or the table invites over-reading. Several
    # CICIDS-2017 categories have single-digit val counts, where one row moves recall
    # by 0.25 or more and a gap between two models is noise rather than a difference.
    support = pd.Series({c: int((labels_val == c).sum()) for c in cat_df.index}, name="n")
    cat_df.insert(0, "n", support)
    cat_df.index.name = "category"
    log(cat_df.to_string(float_format=lambda v: f"{v:.4f}"))
    thin = cat_df.index[cat_df["n"] < 100].tolist()
    if thin:
        log("")
        log(f"  recall is not interpretable for n < 100: {', '.join(thin)}")
        log("  one row changes those figures by more than any real model difference")

    section("STEP 6: how to read this")
    log("  These are IN-DATASET numbers on a split the models trained against, and")
    log("  CICIDS-2017 saturates there: differences of a few thousandths of F1 do not")
    log("  establish that one model is better. The metric that separates candidates in")
    log("  this thesis is zero-day LOAO, which this script does not run.")
    log("  Wall-clock is not a like-for-like comparison either. scikit-learn has no GPU")
    log("  backend, so Random Forest is CPU-bound by the library, not by configuration.")
    log("  What this run does settle is the previously unanswerable question of whether")
    log("  an untested alternative is dramatically better on the same data. It is not a")
    log("  reason on its own to replace Random Forest.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {METRICS}")
    log(f"[saved] {REPORT}")
    log("=== BASELINE COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
