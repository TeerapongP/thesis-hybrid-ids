
import json
import hashlib
import os
import re
import time
import uuid
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
LOAO_METRIC_SCOPE = "component_models_only_no_fusion"


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_run_signature(input_paths, config_path, settings, code_path=None, dependency_lock_path=None):
    file_paths = list(input_paths) + [config_path]
    if code_path is not None:
        file_paths.append(code_path)
    if dependency_lock_path is not None:
        file_paths.append(dependency_lock_path)
    file_hashes = {Path(path).name: _sha256_file(path) for path in sorted(map(Path, file_paths))}
    signature_payload = {"files": file_hashes, "settings": settings}
    serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def checkpoint_directory(signature, base_dir=None):
    return Path(base_dir or CHECKPOINT_DIR) / signature


def results_path(signature):
    return RESULTS_CSV.with_name(
        f"{RESULTS_CSV.stem}_{signature[:12]}_{uuid.uuid4().hex[:8]}{RESULTS_CSV.suffix}"
    )


def save_checkpoint(result, scores_df, signature, base_dir=None):
    slug = slugify_category(result["category"])
    directory = checkpoint_directory(signature, base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    result_path = directory / f"{slug}_{checkpoint_id}.json"
    result_tmp = result_path.with_suffix(".json.tmp")
    result_with_signature = dict(result)
    result_with_signature["run_signature"] = signature
    with open(result_tmp, "w", encoding="utf-8") as f:
        json.dump(result_with_signature, f, indent=2)
    result_tmp.replace(result_path)

    scores_path = directory / f"{slug}_{checkpoint_id}_scores.parquet"
    scores_tmp = scores_path.with_suffix(".parquet.tmp")
    scores_with_signature = scores_df.copy()
    scores_with_signature["run_signature"] = signature
    scores_with_signature.to_parquet(scores_tmp, index=False)
    scores_tmp.replace(scores_path)


def load_checkpoint(cat, signature, base_dir=None):
    slug = slugify_category(cat)
    directory = checkpoint_directory(signature, base_dir)
    if not directory.exists():
        return None
    for path in sorted(directory.glob(f"{slug}_*.json"), reverse=True):
        scores_path = path.with_name(f"{path.stem}_scores.parquet")
        if not scores_path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            score_signatures = pd.read_parquet(scores_path, columns=["run_signature"])["run_signature"]
        except Exception:
            continue
        if (
            result.get("category") == cat
            and result.get("run_signature") == signature
            and not score_signatures.empty
            and score_signatures.eq(signature).all()
        ):
            return result
    return None


def run_fold(cat, joint_full_df, rf_raw_feature_cols, ae_config):
    train_df_fold, test_df_fold = build_loao_split(joint_full_df, cat)
    y_true_fold = (test_df_fold["Label"] != "BENIGN").astype(int).to_numpy()

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
    if_raw_train = -if_fold_model.score_samples(X_ifae_train)
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

    if_min, if_max = float(if_raw_train.min()), float(if_raw_train.max())
    ae_min, ae_max = float(ae_raw_train.min()), float(ae_raw_train.max())
    if_norm_test = (if_raw_test - if_min) / (if_max - if_min)
    ae_norm_test = (ae_raw_test - ae_min) / (ae_max - ae_min)

    result = {
        "category": cat,
        "evaluation_scope": LOAO_METRIC_SCOPE,
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
    print(f"ROOT: {ROOT}  |  DEVICE: {DEVICE}")
    print(f"LOAO metric scope: {LOAO_METRIC_SCOPE}; fused zero-day metrics are not computed")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    ae_config_path = MODELS_DIR / "autoencoder_behavioral_v2_config.json"
    with open(ae_config_path, encoding="utf-8") as f:
        ae_config = json.load(f)

    input_paths = [
        BEHAVIORAL_PROCESSED_DIR / f"cicids2017_behavioral_{split}.parquet"
        for split in ("train", "val", "test")
    ]
    settings = {
        "random_state": RANDOM_STATE,
        "categories": CATEGORIES,
        "rf_sentinel_cols": RF_SENTINEL_COLS,
        "loao_benign_holdout_size": LOAO_BENIGN_HOLDOUT_SIZE,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_class_weight": RF_CLASS_WEIGHT,
        "smote_flat_cap": SMOTE_FLAT_CAP,
        "smote_target_multiplier": SMOTE_TARGET_MULTIPLIER,
        "smote_random_state": SMOTE_RANDOM_STATE,
        "if_contamination": IF_CONTAMINATION,
        "if_n_estimators": IF_N_ESTIMATORS_LOAO,
        "ae_chosen_percentile": AE_CHOSEN_PERCENTILE,
        "benign_internal_val_frac": BENIGN_INTERNAL_VAL_FRAC,
        "device_type": DEVICE.type,
    }
    run_signature = compute_run_signature(
        input_paths,
        ae_config_path,
        settings,
        code_path=Path(__file__),
        dependency_lock_path=ROOT / "uv.lock",
    )
    active_checkpoint_dir = checkpoint_directory(run_signature)
    active_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run signature: {run_signature}")
    print(f"Signature-scoped checkpoints: {active_checkpoint_dir}")

    joint_full_df = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)
    assert joint_full_df.shape == (2_026_398, 92), f"unexpected joint population: {joint_full_df.shape}"
    print(f"Joint LOAO population: {joint_full_df.shape[0]:,} rows x {joint_full_df.shape[1]} columns")

    rf_raw_feature_cols = [
        c for c in joint_full_df.columns
        if c not in ("Label", "orig_row_index") and not c.endswith(("_1m", "_3m", "_5m"))
    ]
    assert len(rf_raw_feature_cols) == 78, f"expected 78 raw RF features, got {len(rf_raw_feature_cols)}"

    loop_start = time.perf_counter()
    for cat in CATEGORIES:
        cached = load_checkpoint(cat, run_signature)
        if cached is not None:
            print(f"{cat:30s}  SKIPPED (checkpoint): RF={cached['rf_recall']:.4f} "
                  f"IF={cached['if_recall']:.4f} AE={cached['ae_recall']:.4f}")
            continue

        fold_start = time.perf_counter()
        result, scores_df = run_fold(cat, joint_full_df, rf_raw_feature_cols, ae_config)
        save_checkpoint(result, scores_df, run_signature)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        print(f"{cat:30s}  RF={result['rf_recall']:.4f} IF={result['if_recall']:.4f} "
              f"AE={result['ae_recall']:.4f}  |  outside [0,1]: if={result['pct_if_norm_outside_unit']:.2f}% "
              f"ae={result['pct_ae_norm_outside_unit']:.2f}%  ({time.perf_counter() - fold_start:.1f}s)")

    print(f"\nFull loop wall-clock: {time.perf_counter() - loop_start:.1f}s (device={DEVICE})")

    results = [load_checkpoint(cat, run_signature) for cat in CATEGORIES]
    assert all(r is not None for r in results), "a fold failed silently -- missing checkpoint"
    active_results_path = results_path(run_signature)
    pd.DataFrame(results).to_csv(active_results_path, index=False)
    print("Saved:", active_results_path)
    print("Per-fold scores (incl. raw, pre-normalization):", active_checkpoint_dir)


if __name__ == "__main__":
    main()
