"""Fusion Layer LOAO re-run with train-fold-fitted score normalization.

notebooks/08_fusion_layer.ipynb S7.1 min-max normalizes each fold's IF-v2/AE-v2 scores using
the *zero-day test set's* own score range, so the held-out attacks help define the scale they
are then detected on, and the transform cannot be reproduced at inference time on a stream.

This script re-runs the identical 10-fold joint LOAO loop with one change: the min-max
constants are fit on the **train fold** and applied to the test fold. Everything else -- the
population, split function, per-model feature views, imputation conventions, hyperparameters --
is copied from that notebook unchanged, so the two runs are directly comparable.

Notes:
  * Normalization is fit on the whole train fold (not BENIGN-only): the direct analogue of
    feature_extractor.py fitting its MinMaxScaler on the full train split.
  * Transformed test scores are NOT clipped to [0, 1]. Out-of-range values are expected and
    are the project's established precedent (CLAUDE.md records val legitimately landing outside
    [0, 1] for the same reason). Clipping would flatten the ranking among the highest-scoring
    rows, which is exactly where detections live.
  * Raw (pre-normalization) test scores are persisted alongside the normalized ones, so any
    alternative normalization can be evaluated later without retraining.

Writes to new paths; nothing produced by 08_fusion_layer.ipynb is modified.

Run from the repo root:
    python -m src.evaluation.fusion_loao_trainnorm

Resumable: each category is checkpointed on completion, so re-running skips finished folds.
"""

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
BEHAVIORAL_PROCESSED_DIR = ROOT / "data" / "processed" / "behavioral"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_DIR = REPORTS_DIR / "fusion_layer_loao_trainnorm_checkpoints"
RESULTS_CSV = REPORTS_DIR / "fusion_layer_loao_trainnorm_results.csv"

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = [
    "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
    "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Attack - Brute Force",
]

RF_SENTINEL_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
LOAO_BENIGN_HOLDOUT_SIZE = 0.2

# Pinned from 02_random_forest.ipynb S1/S7 and 06_behavioral_v2_retrain.ipynb, via
# 08_fusion_layer.ipynb S7.1. Not re-tuned here -- only the normalization changes.
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_CLASS_WEIGHT = "balanced"
SMOTE_FLAT_CAP = 50_000
SMOTE_TARGET_MULTIPLIER = 20
SMOTE_RANDOM_STATE = 42
IF_CONTAMINATION = 0.20
IF_N_ESTIMATORS_LOAO = 100
AE_CHOSEN_PERCENTILE = 99
BENIGN_INTERNAL_VAL_FRAC = 0.10


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim, activation="relu"):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
        encoder_dims = [input_dim] + list(hidden_dims) + [bottleneck_dim]
        decoder_dims = [bottleneck_dim] + list(hidden_dims)[::-1] + [input_dim]

        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:
                encoder_layers.append(act_cls())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(act_cls())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def compute_reconstruction_error(model, X, device, batch_size=200_000):
    """Batched so a full train fold (~1.6M x 92) fits in VRAM -- 08_fusion_layer.ipynb only ever
    scored test folds in one shot, which no longer holds now that train folds are scored too."""
    model.eval()
    errors = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            recon = model(chunk)
            errors[start:start + len(chunk)] = torch.mean((recon - chunk) ** 2, dim=1).cpu().numpy()
    return errors


def train_autoencoder(X_fit, X_internal_val, input_dim, config, device):
    model = Autoencoder(input_dim, config["hidden_dims"], config["bottleneck_dim"], config["activation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()

    fit_tensor = torch.tensor(X_fit, dtype=torch.float32).to(device)
    internal_val_tensor = torch.tensor(X_internal_val, dtype=torch.float32).to(device)
    n_fit = fit_tensor.shape[0]
    batch_size = config["batch_size"]

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch_size):
            batch = fit_tensor[perm[start:start + batch_size]]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(internal_val_tensor), internal_val_tensor).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    return model


def build_loao_split(df, holdout_category):
    benign_mask = df["Label"] == "BENIGN"
    holdout_mask = df["Label"] == holdout_category

    benign_train_df, benign_zeroday_df = train_test_split(
        df.loc[benign_mask], test_size=LOAO_BENIGN_HOLDOUT_SIZE, random_state=RANDOM_STATE
    )
    train_df_fold = pd.concat([benign_train_df, df.loc[~benign_mask & ~holdout_mask]], ignore_index=True)
    zeroday_test_df_fold = pd.concat([benign_zeroday_df, df.loc[holdout_mask]], ignore_index=True)
    return train_df_fold, zeroday_test_df_fold


def add_missing_flags_and_impute_fold(train_df_fold, test_df_fold, sentinel_cols=RF_SENTINEL_COLS):
    train_df_fold = train_df_fold.copy()
    test_df_fold = test_df_fold.copy()
    for c in sentinel_cols:
        flag_col = f"{c}_missing"
        train_df_fold[flag_col] = train_df_fold[c].isna().astype(int)
        test_df_fold[flag_col] = test_df_fold[c].isna().astype(int)
        median = train_df_fold[c].median(skipna=True)
        train_df_fold[c] = train_df_fold[c].fillna(median)
        test_df_fold[c] = test_df_fold[c].fillna(median)
    return train_df_fold, test_df_fold


def build_smote_sampling_strategy(y):
    counts = pd.Series(y).value_counts()
    majority_count = int(counts.max())
    strategy = {}
    for cls, n in counts.items():
        n = int(n)
        if n >= SMOTE_FLAT_CAP:
            continue
        target = min(majority_count, n * SMOTE_TARGET_MULTIPLIER, SMOTE_FLAT_CAP)
        if target > n:
            strategy[cls] = target
    return strategy


def apply_smote_loao(X, y):
    strategy = build_smote_sampling_strategy(y)
    if not strategy:
        return X, y
    min_class_n = min(int((pd.Series(y) == cls).sum()) for cls in strategy)
    smote = SMOTE(
        sampling_strategy=strategy, k_neighbors=max(1, min(5, min_class_n - 1)), random_state=SMOTE_RANDOM_STATE
    )
    return smote.fit_resample(X, y)


def compute_binary_metrics(pred, y_true):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return recall, fpr


def slugify_category(cat):
    return re.sub(r"[\s\-]+", "_", cat.strip().lower())


def save_checkpoint(result, scores_df):
    slug = slugify_category(result["category"])

    path = CHECKPOINT_DIR / f"{slug}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    tmp.replace(path)

    sp = CHECKPOINT_DIR / f"{slug}_scores.parquet"
    sp_tmp = sp.with_suffix(".parquet.tmp")
    scores_df.to_parquet(sp_tmp, index=False)
    sp_tmp.replace(sp)


def load_checkpoint(cat):
    slug = slugify_category(cat)
    path = CHECKPOINT_DIR / f"{slug}.json"
    sp = CHECKPOINT_DIR / f"{slug}_scores.parquet"
    if path.exists() and sp.exists():
        with open(path) as f:
            return json.load(f)
    return None


def run_fold(cat, joint_full_df, rf_raw_feature_cols, ae_config):
    train_df_fold, test_df_fold = build_loao_split(joint_full_df, cat)
    y_true_fold = (test_df_fold["Label"] != "BENIGN").astype(int).to_numpy()

    # ---- RF view: canonical 80-feature contract, fold-local scale, capped SMOTE ----
    rf_train_fold, rf_test_fold = add_missing_flags_and_impute_fold(train_df_fold, test_df_fold)
    rf_scaler = MinMaxScaler().fit(rf_train_fold[rf_raw_feature_cols])
    rf_train_fold[rf_raw_feature_cols] = rf_scaler.transform(rf_train_fold[rf_raw_feature_cols])
    rf_test_fold[rf_raw_feature_cols] = rf_scaler.transform(rf_test_fold[rf_raw_feature_cols])
    rf_feature_cols = rf_raw_feature_cols + [f"{c}_missing" for c in RF_SENTINEL_COLS]
    assert len(rf_feature_cols) == 80, f"expected 80 RF features, got {len(rf_feature_cols)}"
    X_rf_train = rf_train_fold[rf_feature_cols].to_numpy(dtype=np.float64)
    X_rf_test = rf_test_fold[rf_feature_cols].to_numpy(dtype=np.float64)
    X_rf_train_res, y_rf_train_res = apply_smote_loao(X_rf_train, rf_train_fold["Label"].to_numpy())

    rf_fold_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_fold_model.fit(X_rf_train_res, y_rf_train_res)
    benign_idx = list(rf_fold_model.classes_).index("BENIGN")
    rf_attack_probability = 1.0 - rf_fold_model.predict_proba(X_rf_test)[:, benign_idx]
    rf_recall, rf_fpr = compute_binary_metrics(
        (rf_fold_model.predict(X_rf_test) != "BENIGN").astype(int), y_true_fold
    )

    # ---- IF-v2/AE-v2 view: 92 features, impute+flag, fresh scaler ----
    ifae_train_fold, ifae_test_fold = add_missing_flags_and_impute_fold(train_df_fold, test_df_fold)
    fold_feature_cols = [
        c for c in ifae_train_fold.columns
        if c not in ("Label", "orig_row_index") and pd.api.types.is_numeric_dtype(ifae_train_fold[c])
    ]
    assert len(fold_feature_cols) == 92, f"expected 92 behavioral features, got {len(fold_feature_cols)}"

    missing_flag_cols = [f"{c}_missing" for c in RF_SENTINEL_COLS]
    scale_cols = [c for c in fold_feature_cols if c not in missing_flag_cols]
    fold_scaler = MinMaxScaler().fit(ifae_train_fold[scale_cols])
    ifae_train_fold[scale_cols] = fold_scaler.transform(ifae_train_fold[scale_cols])
    ifae_test_fold[scale_cols] = fold_scaler.transform(ifae_test_fold[scale_cols])

    X_ifae_train = ifae_train_fold[fold_feature_cols].to_numpy(dtype=np.float64)
    X_ifae_test = ifae_test_fold[fold_feature_cols].to_numpy(dtype=np.float64)

    if_fold_model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS_LOAO, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_ifae_train)
    if_raw_train = -if_fold_model.score_samples(X_ifae_train)  # THE FIX: train is scored too
    if_raw_test = -if_fold_model.score_samples(X_ifae_test)
    if_recall, if_fpr = compute_binary_metrics((if_fold_model.predict(X_ifae_test) == -1).astype(int), y_true_fold)

    fold_benign_train_df = ifae_train_fold[ifae_train_fold["Label"] == "BENIGN"]
    fold_benign_fit_df, fold_benign_internal_val_df = train_test_split(
        fold_benign_train_df, test_size=BENIGN_INTERNAL_VAL_FRAC, random_state=RANDOM_STATE
    )
    ae_fold_model = train_autoencoder(
        fold_benign_fit_df[fold_feature_cols].to_numpy(dtype=np.float32),
        fold_benign_internal_val_df[fold_feature_cols].to_numpy(dtype=np.float32),
        input_dim=len(fold_feature_cols), config=ae_config, device=DEVICE,
    )
    ae_raw_train = compute_reconstruction_error(
        ae_fold_model, ifae_train_fold[fold_feature_cols].to_numpy(dtype=np.float32), DEVICE
    )
    ae_raw_test = compute_reconstruction_error(
        ae_fold_model, ifae_test_fold[fold_feature_cols].to_numpy(dtype=np.float32), DEVICE
    )
    ae_internal_val_errors = compute_reconstruction_error(
        ae_fold_model, fold_benign_internal_val_df[fold_feature_cols].to_numpy(dtype=np.float32), DEVICE
    )
    ae_fold_threshold = float(np.percentile(ae_internal_val_errors, AE_CHOSEN_PERCENTILE))
    ae_recall, ae_fpr = compute_binary_metrics((ae_raw_test > ae_fold_threshold).astype(int), y_true_fold)

    # ---- THE FIX: min-max constants fit on the TRAIN fold, applied to test, not clipped ----
    if_min, if_max = float(if_raw_train.min()), float(if_raw_train.max())
    ae_min, ae_max = float(ae_raw_train.min()), float(ae_raw_train.max())
    if_norm_test = (if_raw_test - if_min) / (if_max - if_min)
    ae_norm_test = (ae_raw_test - ae_min) / (ae_max - ae_min)

    result = {
        "category": cat,
        "n_test_rows": len(test_df_fold),
        "n_attack_rows": int(y_true_fold.sum()),
        "n_benign_rows": int((y_true_fold == 0).sum()),
        "rf_recall": rf_recall, "rf_fpr": rf_fpr,
        "if_recall": if_recall, "if_fpr": if_fpr,
        "ae_recall": ae_recall, "ae_fpr": ae_fpr,
        "ae_fold_threshold": ae_fold_threshold,
        "normalization": "train_fold_fitted",
        "if_train_min": if_min, "if_train_max": if_max,
        "ae_train_min": ae_min, "ae_train_max": ae_max,
        "if_norm_test_min": float(if_norm_test.min()), "if_norm_test_max": float(if_norm_test.max()),
        "ae_norm_test_min": float(ae_norm_test.min()), "ae_norm_test_max": float(ae_norm_test.max()),
        "pct_if_norm_outside_unit": float(100 * np.mean((if_norm_test < 0) | (if_norm_test > 1))),
        "pct_ae_norm_outside_unit": float(100 * np.mean((ae_norm_test < 0) | (ae_norm_test > 1))),
    }
    scores_df = pd.DataFrame({
        "category": cat,
        "rf_norm": rf_attack_probability,
        "if_norm": if_norm_test,
        "ae_norm": ae_norm_test,
        "if_raw": if_raw_test,
        "ae_raw": ae_raw_test,
        "y_true": y_true_fold,
    })
    return result, scores_df


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ROOT: {ROOT}  |  DEVICE: {DEVICE}")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    with open(MODELS_DIR / "autoencoder_behavioral_v2_config.json") as f:
        ae_config = json.load(f)

    joint_full_df = pd.concat([
        pd.read_parquet(BEHAVIORAL_PROCESSED_DIR / f"cicids2017_behavioral_{split}.parquet")
        for split in ("train", "val", "test")
    ], ignore_index=True)
    assert joint_full_df.shape == (2_026_398, 92), f"unexpected joint population: {joint_full_df.shape}"
    print(f"Joint LOAO population: {joint_full_df.shape[0]:,} rows x {joint_full_df.shape[1]} columns")

    rf_raw_feature_cols = [
        c for c in joint_full_df.columns
        if c not in ("Label", "orig_row_index") and not c.endswith(("_1m", "_3m", "_5m"))
    ]
    assert len(rf_raw_feature_cols) == 78, f"expected 78 raw RF features, got {len(rf_raw_feature_cols)}"

    loop_start = time.perf_counter()
    for cat in CATEGORIES:
        cached = load_checkpoint(cat)
        if cached is not None:
            print(f"{cat:30s}  SKIPPED (checkpoint): RF={cached['rf_recall']:.4f} "
                  f"IF={cached['if_recall']:.4f} AE={cached['ae_recall']:.4f}")
            continue

        fold_start = time.perf_counter()
        result, scores_df = run_fold(cat, joint_full_df, rf_raw_feature_cols, ae_config)
        save_checkpoint(result, scores_df)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        print(f"{cat:30s}  RF={result['rf_recall']:.4f} IF={result['if_recall']:.4f} "
              f"AE={result['ae_recall']:.4f}  |  outside [0,1]: if={result['pct_if_norm_outside_unit']:.2f}% "
              f"ae={result['pct_ae_norm_outside_unit']:.2f}%  ({time.perf_counter() - fold_start:.1f}s)")

    print(f"\nFull loop wall-clock: {time.perf_counter() - loop_start:.1f}s (device={DEVICE})")

    results = [load_checkpoint(cat) for cat in CATEGORIES]
    assert all(r is not None for r in results), "a fold failed silently -- missing checkpoint"
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print("Saved:", RESULTS_CSV)
    print("Per-fold scores (incl. raw, pre-normalization):", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
