
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED_DATA_DIR = "./data/processed"
FEATURES_DATA_DIR = "./data/features"
SPLIT_NAMES = ["train", "val", "test"]

MAPPING_CONFIDENCE_ENCODING = {"n/a": 0, "low": 1, "medium": 2, "high": 3}

SMOTE_FLAT_CAP = 50_000
SMOTE_TARGET_MULTIPLIER = 20
SMOTE_RANDOM_STATE = 42

DATASETS = {
    "cicids2017": {
        "labeled_path": os.path.join(PROCESSED_DATA_DIR, "cicids2017_labeled.parquet"),
        "split_paths": {s: os.path.join(PROCESSED_DATA_DIR, f"cicids2017_{s}.parquet") for s in SPLIT_NAMES},
        "output_paths": {s: os.path.join(FEATURES_DATA_DIR, f"cicids2017_features_{s}.parquet") for s in SPLIT_NAMES},
        "label_col": "Label",
        "exclude_from_features": ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"],
        "reference_cols": ["technique_id", "technique_name", "tactic", "mapping_confidence"],
        "extra_output_cols": [],
        "sentinel_cols": ["Init_Win_bytes_forward", "Init_Win_bytes_backward"],
    },
    "unswnb15": {
        "labeled_path": os.path.join(PROCESSED_DATA_DIR, "unswnb15_labeled.parquet"),
        "split_paths": {s: os.path.join(PROCESSED_DATA_DIR, f"unswnb15_{s}.parquet") for s in SPLIT_NAMES},
        "output_paths": {s: os.path.join(FEATURES_DATA_DIR, f"unswnb15_features_{s}.parquet") for s in SPLIT_NAMES},
        "label_col": "Label",
        "exclude_from_features": ["Label", "attack_cat", "technique_id", "technique_name", "tactic", "mapping_confidence"],
        "reference_cols": ["technique_id", "technique_name", "tactic", "mapping_confidence"],
        "extra_output_cols": ["attack_cat"],
        "sentinel_cols": [],
    },
}

REPORT_PATH = "./reports/feature_extractor_report.txt"

_REPORT_LINES = []


def log(msg=""):
    msg = str(msg)
    print(msg)
    _REPORT_LINES.append(msg)


def section(title):
    log("=" * 70)
    log(title)
    log("=" * 70)


def safe_read_parquet(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' — ต้องรัน preprocess_pipeline.py และ label_builder.py ก่อน"
        )
    return pd.read_parquet(path)


def compute_row_hash(df, key_cols):
    return pd.util.hash_pandas_object(df[key_cols], index=False)


def join_labels_onto_split(split_df, labeled_df, reference_cols, dataset_name, split_name):
    key_cols = list(split_df.columns)
    hash_col = "_row_hash"

    left = split_df.copy()
    left[hash_col] = compute_row_hash(left, key_cols)

    right = labeled_df[key_cols + reference_cols].copy()
    right[hash_col] = compute_row_hash(right, key_cols)
    right = right[[hash_col] + reference_cols]

    merged = left.merge(right, how="left", on=hash_col, validate="1:1", indicator=True)
    n_unmatched = int((merged["_merge"] != "both").sum())
    if n_unmatched > 0 or len(merged) != len(split_df):
        raise RuntimeError(
            f"join_labels_onto_split: {dataset_name}/{split_name} มี {n_unmatched} แถวที่หา "
            f"row คู่กันใน labeled parquet ไม่เจอ (len(merged)={len(merged)} "
            f"len(split_df)={len(split_df)}) -> ตรวจสอบว่า *_clean.parquet / *_labeled.parquet / "
            f"*_{{train,val,test}}.parquet มาจาก preprocess_pipeline.py+label_builder.py รอบเดียวกัน"
        )
    merged = merged.drop(columns=[hash_col, "_merge"])
    log(f"  {dataset_name}/{split_name}: join label กลับเข้า split สำเร็จ 100% ({len(merged):,} แถว)")
    return merged


def get_network_feature_cols(df, exclude_cols):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]


def fit_sentinel_imputation(train_df, sentinel_cols):
    return {c: train_df[c].median(skipna=True) for c in sentinel_cols}


def apply_sentinel_imputation(df, sentinel_cols, medians):
    df = df.copy()
    for c in sentinel_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(medians[c])
    return df


def fit_integer_encoding(train_series):
    uniques = sorted(train_series.dropna().unique().tolist())
    return {val: i + 1 for i, val in enumerate(uniques)}


def apply_integer_encoding(series, mapping, dataset_name, col_name, split_name):
    is_null = series.isna()
    mapped = series.map(mapping)
    unseen_mask = (~is_null) & mapped.isna()
    n_unseen = int(unseen_mask.sum())
    if n_unseen > 0:
        unseen_vals = sorted(series.loc[unseen_mask].unique().tolist())
        log(
            f"  [WARNING] {dataset_name}/{split_name}: พบค่า {col_name} ที่ไม่เคยเห็นตอน fit บน "
            f"train ({n_unseen} แถว): {unseen_vals} -> เข้ารหัสเป็น -1 (unseen-at-train sentinel, "
            f"แยกจาก 0=null)"
        )
    encoded = mapped.fillna(-1)
    encoded = encoded.where(~is_null, 0)
    return encoded.astype(int)


def apply_mapping_confidence_encoding(series):
    return series.map(MAPPING_CONFIDENCE_ENCODING).fillna(0).astype(int)


def build_is_known_technique(technique_id_series):
    return technique_id_series.notna().astype(int)


def fit_scaler(train_df, feature_cols):
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df, feature_cols, scaler):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def model_input_feature_cols(network_feature_cols, missing_flag_cols):
    return list(network_feature_cols) + list(missing_flag_cols)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_feature_outputs(dataset_key, final_dfs, cfg, run_details):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    artifact_root = Path(cfg.get("artifact_root", FEATURES_DATA_DIR))
    history_dir = artifact_root / "history" / f"{dataset_key}_{run_id}"
    manifest_dir = artifact_root / "provenance"
    output_paths = {split: Path(path) for split, path in cfg["output_paths"].items()}
    source_paths = (
        cfg["source_paths"]
        if "source_paths" in cfg
        else [cfg["labeled_path"], *cfg["split_paths"].values()]
    )
    source_code_paths = [Path(__file__), *map(Path, cfg.get("source_code_paths", []))]
    source_manifest = {
        str(path): {
            "sha256": _sha256_file(path),
            "size_bytes": Path(path).stat().st_size,
        }
        for path in source_paths
    }

    staged_paths = {}
    try:
        for split_name, path in output_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".parquet",
                delete=False,
            ) as handle:
                staged = Path(handle.name)
            staged_paths[split_name] = staged
            final_dfs[split_name].to_parquet(staged, index=False)

        existing_paths = [path for path in output_paths.values() if path.exists()]
        if existing_paths:
            history_dir.mkdir(parents=True, exist_ok=False)
            for path in existing_paths:
                shutil.copy2(path, history_dir / path.name)

        promoted = []
        try:
            for split_name, path in output_paths.items():
                os.replace(staged_paths[split_name], path)
                promoted.append(path)
        except Exception:
            for path in reversed(promoted):
                backup_path = history_dir / path.name
                if backup_path.exists():
                    shutil.copy2(backup_path, path)
                else:
                    path.unlink(missing_ok=True)
            raise

        manifest = {
            "run_id": run_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_key,
            "inputs": source_manifest,
            "source_code_sha256": {
                str(path): _sha256_file(path) for path in source_code_paths
            },
            "dependency_lock_sha256": _sha256_file(Path(__file__).resolve().parents[2] / "uv.lock"),
            "settings": run_details,
            "outputs": {
                split: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for split, path in output_paths.items()
            },
            "preserved_previous_outputs": {
                split: str(history_dir / path.name)
                for split, path in output_paths.items()
                if (history_dir / path.name).exists()
            },
        }
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{dataset_key}_{run_id}.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_dir,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_manifest = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.flush()
        os.replace(temp_manifest, manifest_path)
    finally:
        for staged in staged_paths.values():
            staged.unlink(missing_ok=True)
        if "temp_manifest" in locals():
            temp_manifest.unlink(missing_ok=True)

    return output_paths, manifest_path, history_dir if existing_paths else None


def build_smote_sampling_strategy(y):
    counts = y.value_counts()
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


def apply_smote(X, y, dataset_name):
    strategy = build_smote_sampling_strategy(y)
    if not strategy:
        log(
            f"  {dataset_name}: ไม่มี class ไหนต่ำกว่าเพดาน SMOTE_FLAT_CAP={SMOTE_FLAT_CAP:,} -> "
            f"SMOTE ไม่ทำอะไร (no-op) คืนค่า train เดิม"
        )
        return X, y, False

    log(
        f"  {dataset_name}: SMOTE sampling_strategy ต่อ class ที่จะถูกขยาย (เพดาน = "
        f"min(majority, {SMOTE_TARGET_MULTIPLIER}x ของเดิม, {SMOTE_FLAT_CAP:,})):"
    )
    for cls, target in strategy.items():
        n_before = int((y == cls).sum())
        log(f"    {str(cls):35s} {n_before:>9,d} -> {target:>9,d}")

    min_class_n = min(int((y == cls).sum()) for cls in strategy)
    k_neighbors = max(1, min(5, min_class_n - 1))

    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    log(
        f"  {dataset_name}: SMOTE k_neighbors={k_neighbors}  แถวก่อน={len(X):,}  "
        f"แถวหลัง={len(X_res):,}  (เพิ่มขึ้น {len(X_res) - len(X):,})"
    )
    return X_res, y_res, True


def build_label_reference_map(df, label_col, reference_cols):
    ref_map = {}
    non_deterministic_labels = []
    for label_val, group in df.groupby(label_col, dropna=False)[reference_cols]:
        unique_rows = group.drop_duplicates()
        if len(unique_rows) == 1:
            ref_map[label_val] = unique_rows.iloc[0].to_dict()
        else:
            non_deterministic_labels.append(label_val)
    return ref_map, non_deterministic_labels


def attach_reference_columns(y_series, ref_map, reference_cols, non_deterministic_labels, dataset_name):
    if non_deterministic_labels:
        log(
            f"  [WARNING] {dataset_name}: label ต่อไปนี้ map ไปที่ reference columns มากกว่า 1 "
            f"ชุดค่า -> เติม NaN ให้แถว synthetic ของ label เหล่านี้แทนการเดาค่า: "
            f"{non_deterministic_labels}"
        )
    records = [ref_map.get(label_val, {c: np.nan for c in reference_cols}) for label_val in y_series]
    return pd.DataFrame.from_records(records, columns=reference_cols)


def process_dataset(dataset_key, cfg):
    section(f"=== dataset: {dataset_key} ===")

    labeled_df = safe_read_parquet(cfg["labeled_path"])
    log(f"  โหลด {cfg['labeled_path']}  shape={labeled_df.shape}")

    label_col = cfg["label_col"]
    exclude_cols = cfg["exclude_from_features"]
    reference_cols = cfg["reference_cols"]
    extra_output_cols = cfg["extra_output_cols"]
    output_reference_cols = reference_cols + extra_output_cols
    sentinel_cols = cfg["sentinel_cols"]

    section(f"{dataset_key}: STEP 1 — join คอลัมน์ MITRE กลับเข้า train/val/test")
    split_dfs = {}
    for split_name, path in cfg["split_paths"].items():
        raw_split_df = safe_read_parquet(path)
        split_dfs[split_name] = join_labels_onto_split(
            raw_split_df, labeled_df, reference_cols, dataset_key, split_name
        )

    section(f"{dataset_key}: STEP 2 — network feature columns (numeric dtype, ไม่รวม identifier/label)")
    network_feature_cols = get_network_feature_cols(split_dfs["train"], exclude_cols)
    log(f"  network feature columns ({len(network_feature_cols)}): {network_feature_cols}")

    section(f"{dataset_key}: STEP 3 — impute คอลัมน์ sentinel (fit median จาก train เท่านั้น)")
    sentinel_medians = fit_sentinel_imputation(split_dfs["train"], sentinel_cols)
    for c, med in sentinel_medians.items():
        log(f"  {c}: median (จาก train) = {med}")
    missing_flag_cols = [f"{c}_missing" for c in sentinel_cols]
    for split_name in SPLIT_NAMES:
        split_dfs[split_name] = apply_sentinel_imputation(split_dfs[split_name], sentinel_cols, sentinel_medians)

    section(f"{dataset_key}: STEP 5 — MinMaxScaler (fit บน train เท่านั้น, เฉพาะ network features)")
    scaler = fit_scaler(split_dfs["train"], network_feature_cols)
    log(f"  fit บน train เท่านั้น: {len(network_feature_cols)} คอลัมน์ (ไม่แตะ {missing_flag_cols})")
    for split_name in SPLIT_NAMES:
        split_dfs[split_name] = apply_scaler(split_dfs[split_name], network_feature_cols, scaler)
    log(f"  scale แล้ว: {', '.join(SPLIT_NAMES)} (ใช้ scaler ตัวเดียวกันที่ fit จาก train)")

    final_feature_cols = model_input_feature_cols(network_feature_cols, missing_flag_cols)

    section(f"{dataset_key}: STEP 6 — SMOTE (train เท่านั้น)")
    train_df = split_dfs["train"]
    X_train = train_df[final_feature_cols]
    y_train = train_df[label_col]
    ref_map, non_deterministic_labels = build_label_reference_map(train_df, label_col, output_reference_cols)

    X_train_res, y_train_res, smote_applied = apply_smote(X_train, y_train, dataset_key)

    if smote_applied:
        y_train_res = pd.Series(y_train_res, name=label_col).reset_index(drop=True)
        X_train_res = pd.DataFrame(X_train_res, columns=final_feature_cols).reset_index(drop=True)
        ref_df = attach_reference_columns(
            y_train_res, ref_map, output_reference_cols, non_deterministic_labels, dataset_key
        )
        train_features_df = pd.concat([X_train_res, ref_df, y_train_res], axis=1)
    else:
        train_features_df = pd.concat(
            [
                train_df[final_feature_cols].reset_index(drop=True),
                train_df[output_reference_cols].reset_index(drop=True),
                train_df[[label_col]].reset_index(drop=True),
            ],
            axis=1,
        )

    final_dfs = {"train": train_features_df}
    for split_name in ["val", "test"]:
        df = split_dfs[split_name]
        final_dfs[split_name] = pd.concat(
            [
                df[final_feature_cols].reset_index(drop=True),
                df[output_reference_cols].reset_index(drop=True),
                df[[label_col]].reset_index(drop=True),
            ],
            axis=1,
        )
        log(f"  {dataset_key}/{split_name}: ไม่ถูก SMOTE แตะ ({len(final_dfs[split_name]):,} แถว เท่ากับก่อน SMOTE)")

    section(f"{dataset_key}: STEP 7 — เซฟผลลัพธ์")
    log(f"  network features: {len(network_feature_cols)}")
    log(f"  sentinel missing-flag features (เพิ่มจาก impute STEP 3): {len(missing_flag_cols)} {missing_flag_cols}")
    log("  label-derived CTI encodings ไม่ถูกสร้างเป็น model input")
    log(f"  รวม feature columns ทั้งหมดต่อแถว (ไม่รวม reference/label): {len(final_feature_cols)}")

    output_paths, manifest_path, history_dir = write_feature_outputs(
        dataset_key,
        final_dfs,
        cfg,
        {
            "model_input_feature_columns": final_feature_cols,
            "network_feature_columns": network_feature_cols,
            "missing_flag_columns": missing_flag_cols,
            "sentinel_medians": {
                column: float(value) if pd.notna(value) else None
                for column, value in sentinel_medians.items()
            },
            "smote_applied": bool(smote_applied),
            "smote_flat_cap": SMOTE_FLAT_CAP,
            "smote_target_multiplier": SMOTE_TARGET_MULTIPLIER,
            "smote_random_state": SMOTE_RANDOM_STATE,
        },
    )
    for split_name, path in output_paths.items():
        log(f"  เซฟ {dataset_key}/{split_name} -> {path}  shape={final_dfs[split_name].shape}")
    log(f"  provenance manifest: {manifest_path}")
    if history_dir is not None:
        log(f"  preserved previous outputs: {history_dir}")

    return {
        "network_feature_count": len(network_feature_cols),
        "missing_flag_count": len(missing_flag_cols),
        "cti_feature_count": 0,
        "total_feature_count": len(final_feature_cols),
        "smote_applied": smote_applied,
        "train_rows_before_smote": len(train_df),
        "train_rows_after_smote": len(train_features_df),
        "val_rows": len(final_dfs["val"]),
        "test_rows": len(final_dfs["test"]),
    }


def main():
    results = {}
    for dataset_key, cfg in DATASETS.items():
        results[dataset_key] = process_dataset(dataset_key, cfg)

    section("สรุปผลรวมทุก dataset")
    for dataset_key, r in results.items():
        log(f"--- {dataset_key} ---")
        log(
            f"  feature count: network={r['network_feature_count']}  "
            f"sentinel_missing_flags={r['missing_flag_count']}  cti={r['cti_feature_count']}  "
            f"รวม={r['total_feature_count']}"
        )
        log(f"  SMOTE ถูกเรียกใช้จริงหรือไม่ (มี class ต่ำกว่าเพดาน): {r['smote_applied']}")
        log(
            f"  train rows: ก่อน SMOTE={r['train_rows_before_smote']:,}  "
            f"หลัง SMOTE={r['train_rows_after_smote']:,}"
        )
        log(f"  val rows={r['val_rows']:,}  test rows={r['test_rows']:,} (ไม่ถูก SMOTE แตะเลย)")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
