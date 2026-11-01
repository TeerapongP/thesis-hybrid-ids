"""Nested LOAO on the unified population: all three members, one feature set, one split.

Why this replaces fusion_loao_trainnorm.py. That script had to carry two parallel feature
spaces through every fold -- 80 columns for Random Forest from the MachineLearningCSV
extraction, 92 for the unsupervised members from the behavioral one -- because the two
populations were only reconcilable through a row-hash join that matched 76.4% of BENIGN
rows. build_unified_population.py removed the join, so a fold now has one imputation, one
scaler and one 94-column matrix shared by all three members. That is the whole point of the
advisor's objection being fixed at the data layer rather than argued about.

What is kept deliberately, because it is what made the previous results trustworthy:

  TRAIN-FOLD normalization. IF and AE raw scores are min-max normalized with constants fit
  on the fold's TRAINING rows, never on the zero-day test rows. Normalizing a test fold with
  its own min/max leaks the test distribution into the score and silently inflates every
  candidate; an earlier run did exactly that and had to be relabelled invalid. The deployed
  constants in fusion_layer.py are fit on the full train split for the same reason, so a
  threshold selected here transfers to deployment.

  Per-fold checkpoints. Ten folds cost roughly 45 minutes, nearly all of it Random Forest.
  Thresholds have to be re-swept many times during selection, so scores are persisted per
  fold and the sweep reads them back instead of retraining. The signature hashes the inputs
  and settings, so changing a fold's algorithm invalidates stale checkpoints rather than
  silently mixing them.

  No fused metrics here, by design. This script measures the component models and writes
  their normalized zero-day scores. Threshold selection belongs in fusion_or_sweep.py, where
  it is nested: thresholds are chosen on the other nine categories and applied to the
  held-out one. Choosing a threshold on the fold you then report is the mistake that made the
  retired cascade look competitive.

Run from the repo root with the Anaconda interpreter (needs the CUDA torch build):
    python -m src.evaluation.fusion_loao_unified
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.fusion_loao_trainnorm import (
    AE_CHOSEN_PERCENTILE,
    BENIGN_INTERNAL_VAL_FRAC,
    CATEGORIES,
    IF_CONTAMINATION,
    IF_N_ESTIMATORS_LOAO,
    RANDOM_STATE,
    RF_CLASS_WEIGHT,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_SENTINEL_COLS,
    Autoencoder,  # noqa: F401  (imported for provenance; constructed by train_autoencoder)
    add_missing_flags_and_impute_fold,
    apply_smote_loao,
    compute_binary_metrics,
    compute_reconstruction_error,
    slugify_category,
    train_autoencoder,
)

ROOT = Path(__file__).resolve().parents[2]
LOAO_DIR = ROOT / "data/processed/unified_loao"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CHECKPOINT_DIR = REPORTS / "fusion_layer_loao_unified_checkpoints"
RESULTS_CSV = REPORTS / "fusion_layer_loao_unified_results.csv"
REPORT = REPORTS / "fusion_loao_unified_report.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_FEATURES = 94
# Identifiers never reach a model. Source IP in particular would let it memorise which hosts
# attacked in this capture, which is leakage wearing a feature's clothes.
IDENTIFIER_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "day_group"]
REFERENCE_COLS = ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"]
# Bumped by hand whenever the fold algorithm changes in a way that makes old scores
# incomparable. It is part of the signature, so bumping it invalidates every checkpoint.
FOLD_ALGORITHM_VERSION = 1

_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(str(msg))


def section(title: str) -> None:
    log("=" * 74)
    log(title)
    log("=" * 74)


def run_signature(ae_cfg: dict) -> str:
    """Hash the fold inputs and settings so stale checkpoints cannot be reused silently."""
    payload = {
        "fold_algorithm_version": FOLD_ALGORITHM_VERSION,
        "expected_features": EXPECTED_FEATURES,
        "rf": {"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH,
               "class_weight": RF_CLASS_WEIGHT},
        "if": {"n_estimators": IF_N_ESTIMATORS_LOAO, "contamination": IF_CONTAMINATION},
        "ae": {k: ae_cfg.get(k) for k in ("hidden_dims", "bottleneck_dim", "activation",
                                          "learning_rate", "epochs", "batch_size",
                                          "weight_decay", "early_stopping_patience")},
        "ae_percentile": AE_CHOSEN_PERCENTILE,
        "benign_internal_val_frac": BENIGN_INTERNAL_VAL_FRAC,
        "random_state": RANDOM_STATE,
        "normalization": "train_fold_fitted",
        "categories": CATEGORIES,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def checkpoint_paths(signature: str, category: str):
    directory = CHECKPOINT_DIR / signature
    slug = slugify_category(category)
    return directory / f"{slug}.json", directory / f"{slug}_scores.parquet"


def load_checkpoint(signature: str, category: str):
    result_path, scores_path = checkpoint_paths(signature, category)
    if not (result_path.exists() and scores_path.exists()):
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        scores = pd.read_parquet(scores_path)
    except Exception:
        return None
    if result.get("run_signature") != signature or scores.empty:
        return None
    return result, scores


def save_checkpoint(signature: str, result: dict, scores: pd.DataFrame) -> None:
    result_path, scores_path = checkpoint_paths(signature, result["category"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(result, run_signature=signature)
    result_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    scores.assign(run_signature=signature).to_parquet(scores_path, index=False)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(IDENTIFIER_COLS) | set(REFERENCE_COLS) | {"orig_row_index"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def run_fold(category: str, ae_cfg: dict) -> tuple[dict, pd.DataFrame]:
    slug = slugify_category(category)
    train_fold = pd.read_parquet(LOAO_DIR / f"loao_{slug}_train.parquet")
    test_fold = pd.read_parquet(LOAO_DIR / f"loao_{slug}_zeroday_test.parquet")
    assert category not in set(train_fold["Label"]), \
        f"{category} leaked into its own LOAO training fold"
    y_true = (test_fold["Label"] != "BENIGN").astype(int).to_numpy()

    # One imputation and one scaler for all three members: the whole reason this script
    # exists instead of the two-feature-space predecessor.
    train_fold, test_fold = add_missing_flags_and_impute_fold(train_fold, test_fold)
    feature_cols = select_feature_columns(train_fold)
    assert len(feature_cols) == EXPECTED_FEATURES, \
        f"expected {EXPECTED_FEATURES} features, got {len(feature_cols)}"
    flag_cols = [f"{c}_missing" for c in RF_SENTINEL_COLS]
    scale_cols = [c for c in feature_cols if c not in flag_cols]

    scaler = MinMaxScaler().fit(train_fold[scale_cols])
    train_fold[scale_cols] = scaler.transform(train_fold[scale_cols])
    test_fold[scale_cols] = scaler.transform(test_fold[scale_cols])

    X_train = train_fold[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_fold[feature_cols].to_numpy(dtype=np.float64)
    labels_train = train_fold["Label"].to_numpy()

    # ── Random Forest ────────────────────────────────────────────────
    t0 = time.perf_counter()
    X_res, y_res = apply_smote_loao(X_train, labels_train)
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_res, y_res)
    benign_idx = list(rf.classes_).index("BENIGN")
    rf_norm = 1.0 - rf.predict_proba(X_test)[:, benign_idx]
    rf_recall, rf_fpr = compute_binary_metrics(
        (rf.predict(X_test) != "BENIGN").astype(int), y_true
    )
    rf_seconds = time.perf_counter() - t0
    log(f"    RF   recall {rf_recall:.4f}  fpr {rf_fpr:.4f}   ({rf_seconds:.0f}s, "
        f"{len(X_res):,} rows after capped SMOTE)")

    # ── Isolation Forest ─────────────────────────────────────────────
    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS_LOAO, contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train)
    if_raw_train = -iso.score_samples(X_train)
    if_raw_test = -iso.score_samples(X_test)
    if_recall, if_fpr = compute_binary_metrics((iso.predict(X_test) == -1).astype(int), y_true)
    log(f"    IF   recall {if_recall:.4f}  fpr {if_fpr:.4f}")

    # ── Autoencoder, BENIGN rows of this fold only ───────────────────
    benign_fold = train_fold[train_fold["Label"] == "BENIGN"]
    benign_fit, benign_internal_val = train_test_split(
        benign_fold, test_size=BENIGN_INTERNAL_VAL_FRAC, random_state=RANDOM_STATE
    )
    ae = train_autoencoder(
        benign_fit[feature_cols].to_numpy(dtype=np.float32),
        benign_internal_val[feature_cols].to_numpy(dtype=np.float32),
        input_dim=len(feature_cols), config=ae_cfg, device=DEVICE,
    )
    ae_raw_train = compute_reconstruction_error(ae, X_train.astype(np.float32), DEVICE)
    ae_raw_test = compute_reconstruction_error(ae, X_test.astype(np.float32), DEVICE)
    ae_internal = compute_reconstruction_error(
        ae, benign_internal_val[feature_cols].to_numpy(dtype=np.float32), DEVICE
    )
    # The AE's own operating threshold comes from BENIGN rows held out INSIDE the training
    # fold, never from the zero-day rows it is about to be judged on.
    ae_threshold = float(np.percentile(ae_internal, AE_CHOSEN_PERCENTILE))
    ae_recall, ae_fpr = compute_binary_metrics((ae_raw_test > ae_threshold).astype(int), y_true)
    log(f"    AE   recall {ae_recall:.4f}  fpr {ae_fpr:.4f}   (threshold {ae_threshold:.8f})")

    # ── train-fold normalization ─────────────────────────────────────
    if_min, if_max = float(if_raw_train.min()), float(if_raw_train.max())
    ae_min, ae_max = float(ae_raw_train.min()), float(ae_raw_train.max())
    if_norm = (if_raw_test - if_min) / (if_max - if_min)
    ae_norm = (ae_raw_test - ae_min) / (ae_max - ae_min)
    # Intentionally unclipped: a zero-day row more anomalous than anything in training
    # SHOULD land above 1.0, and clipping here would hide that from the threshold sweep.

    result = {
        "category": category,
        "evaluation_scope": "component_models_only_no_fusion",
        "population": "unified",
        "feature_count": len(feature_cols),
        "n_test_rows": int(len(test_fold)),
        "n_attack_rows": int(y_true.sum()),
        "n_benign_rows": int((y_true == 0).sum()),
        "rf_recall": rf_recall, "rf_fpr": rf_fpr,
        "if_recall": if_recall, "if_fpr": if_fpr,
        "ae_recall": ae_recall, "ae_fpr": ae_fpr,
        "ae_fold_threshold": ae_threshold,
        "normalization": "train_fold_fitted",
        "if_train_min": if_min, "if_train_max": if_max,
        "ae_train_min": ae_min, "ae_train_max": ae_max,
        "pct_if_norm_outside_unit": float(100 * np.mean((if_norm < 0) | (if_norm > 1))),
        "pct_ae_norm_outside_unit": float(100 * np.mean((ae_norm < 0) | (ae_norm > 1))),
        "rf_fit_seconds": rf_seconds,
    }
    scores = pd.DataFrame({
        "category": category,
        "rf_norm": rf_norm, "if_norm": if_norm, "ae_norm": ae_norm,
        "if_raw": if_raw_test, "ae_raw": ae_raw_test,
        "y_true": y_true,
    })
    return result, scores


def main() -> None:
    section("unified-population LOAO: component models, train-fold normalization")
    ae_cfg = json.loads((MODELS / "autoencoder_unified_config.json").read_text(encoding="utf-8"))
    signature = run_signature(ae_cfg)
    log(f"  device {DEVICE}   signature {signature}")
    log(f"  checkpoints  {CHECKPOINT_DIR / signature}")
    log("  fused metrics are NOT computed here; run fusion_or_sweep.py for nested selection")

    results = []
    for i, category in enumerate(CATEGORIES, 1):
        log("")
        log(f"[{i}/{len(CATEGORIES)}] {category}")
        cached = load_checkpoint(signature, category)
        if cached is not None:
            result, _ = cached
            log("    reused checkpoint from an earlier run")
        else:
            result, scores = run_fold(category, ae_cfg)
            save_checkpoint(signature, result, scores)
        results.append(result)

    section("per-category component results")
    df = pd.DataFrame(results)
    cols = ["category", "n_attack_rows", "rf_recall", "if_recall", "ae_recall",
            "rf_fpr", "if_fpr", "ae_fpr"]
    log(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(RESULTS_CSV, index=False)

    section("how to read this")
    log("  Recall is not comparable across rows at face value: each zero-day fold keeps that")
    log("  category's natural prevalence, which ranges from about 29% attack rows down to")
    log("  0.35%. Report per category, never as an unweighted mean.")
    log("  These are component numbers. No fusion rule has been applied and no threshold has")
    log("  been selected, because selecting one on these same folds and then reporting them")
    log("  is the error that made the retired cascade look competitive.")
    log(f"  Next: python -m src.evaluation.fusion_or_sweep --checkpoint-dir {CHECKPOINT_DIR.name}")

    REPORT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"[saved] {RESULTS_CSV}")
    log(f"[saved] {REPORT}")
    log("=== UNIFIED LOAO COMPLETE ===")


if __name__ == "__main__":
    main()
