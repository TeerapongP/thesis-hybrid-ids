
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from src.features import feature_extractor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED_DATA_DIR = "./data/processed"
BEHAVIORAL_PROCESSED_DIR = "./data/processed/behavioral"
BEHAVIORAL_FEATURES_DIR = "./data/features/behavioral"
SPLIT_NAMES = ["train", "val", "test"]

LABELED_PATH = os.path.join(PROCESSED_DATA_DIR, "cicids2017_labeled.parquet")
SPLIT_PATHS = {s: os.path.join(BEHAVIORAL_PROCESSED_DIR, f"cicids2017_behavioral_{s}.parquet") for s in SPLIT_NAMES}
OUTPUT_PATHS = {s: os.path.join(BEHAVIORAL_FEATURES_DIR, f"cicids2017_behavioral_features_{s}.parquet") for s in SPLIT_NAMES}
REPORT_PATH = "./reports/feature_extractor_behavioral_report.txt"

LABEL_COL = "Label"
REFERENCE_COLS = ["technique_id", "technique_name", "tactic", "mapping_confidence"]
EXTRA_OUTPUT_COLS = ["orig_row_index"]
EXCLUDE_FROM_FEATURES = [LABEL_COL] + REFERENCE_COLS + EXTRA_OUTPUT_COLS
SENTINEL_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
ORIG_ROW_INDEX_SYNTHETIC_SENTINEL = -1

MAPPING_CONFIDENCE_ENCODING = {"n/a": 0, "low": 1, "medium": 2, "high": 3}

SMOTE_FLAT_CAP = 50_000
SMOTE_TARGET_MULTIPLIER = 20
SMOTE_RANDOM_STATE = 42

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
            f"ไม่พบไฟล์: '{path}' -> ต้องรัน validate_behavioral_join.py (Phase 1) ก่อน"
        )
    return pd.read_parquet(path)


def compute_row_hash(df, key_cols):
    return pd.util.hash_pandas_object(df[key_cols], index=False)


def join_labels_onto_split(split_df, labeled_df, key_cols, reference_cols, split_name):
    hash_col = "_row_hash"

    assert split_df[key_cols].dtypes.equals(labeled_df[key_cols].dtypes), (
        f"join_labels_onto_split: cicids2017/{split_name} dtype ของ key columns ไม่ตรงกันระหว่าง "
        f"split กับ labeled -> ต้องตรวจก่อน hash (บทเรียนจาก Phase 1: dtype mismatch ทำให้ match 0%)"
    )

    left = split_df.copy()
    left[hash_col] = compute_row_hash(left, key_cols)

    right = labeled_df[key_cols + reference_cols].copy()
    right[hash_col] = compute_row_hash(right, key_cols)
    right = right[[hash_col] + reference_cols]

    merged = left.merge(right, how="left", on=hash_col, validate="1:1", indicator=True)
    n_unmatched = int((merged["_merge"] != "both").sum())
    if n_unmatched > 0 or len(merged) != len(split_df):
        raise RuntimeError(
            f"join_labels_onto_split: cicids2017/{split_name} มี {n_unmatched} แถวที่หา row คู่กันใน "
            f"labeled parquet ไม่เจอ (len(merged)={len(merged)} len(split_df)={len(split_df)}) -> "
            f"ตรวจว่า cicids2017_labeled.parquet มาจาก preprocess_pipeline.py/label_builder.py รอบ "
            f"เดียวกับที่ cicids2017_behavioral_{{train,val,test}}.parquet's 79 คอลัมน์เดิมสืบทอดมาหรือไม่"
        )
    merged = merged.drop(columns=[hash_col, "_merge"])
    log(f"  cicids2017/{split_name}: join label กลับเข้า split สำเร็จ 100% ({len(merged):,} แถว)")
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


def apply_integer_encoding(series, mapping, col_name, split_name):
    is_null = series.isna()
    mapped = series.map(mapping)
    unseen_mask = (~is_null) & mapped.isna()
    n_unseen = int(unseen_mask.sum())
    if n_unseen > 0:
        unseen_vals = sorted(series.loc[unseen_mask].unique().tolist())
        log(
            f"  [WARNING] cicids2017/{split_name}: พบค่า {col_name} ที่ไม่เคยเห็นตอน fit บน train "
            f"({n_unseen} แถว): {unseen_vals} -> เข้ารหัสเป็น -1 (unseen-at-train sentinel, แยกจาก 0=null)"
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


def apply_smote(X, y):
    strategy = build_smote_sampling_strategy(y)
    if not strategy:
        log(f"  cicids2017: ไม่มี class ไหนต่ำกว่าเพดาน SMOTE_FLAT_CAP={SMOTE_FLAT_CAP:,} -> SMOTE ไม่ทำอะไร (no-op)")
        return X, y, False

    log(f"  cicids2017: SMOTE sampling_strategy ต่อ class ที่จะถูกขยาย (เพดาน = min(majority, {SMOTE_TARGET_MULTIPLIER}x ของเดิม, {SMOTE_FLAT_CAP:,})):")
    for cls, target in strategy.items():
        n_before = int((y == cls).sum())
        log(f"    {str(cls):35s} {n_before:>9,d} -> {target:>9,d}")

    min_class_n = min(int((y == cls).sum()) for cls in strategy)
    k_neighbors = max(1, min(5, min_class_n - 1))

    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    log(f"  cicids2017: SMOTE k_neighbors={k_neighbors}  แถวก่อน={len(X):,}  แถวหลัง={len(X_res):,}  (เพิ่มขึ้น {len(X_res) - len(X):,})")
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


def attach_reference_columns(y_series, ref_map, reference_cols, non_deterministic_labels):
    if non_deterministic_labels:
        log(
            f"  [WARNING] cicids2017: label ต่อไปนี้ map ไปที่ reference columns มากกว่า 1 ชุดค่า -> "
            f"เติม NaN ให้แถว synthetic ของ label เหล่านี้แทนการเดาค่า: {non_deterministic_labels}"
        )
    records = [ref_map.get(label_val, {c: np.nan for c in reference_cols}) for label_val in y_series]
    return pd.DataFrame.from_records(records, columns=reference_cols)


def attach_orig_row_index_after_smote(orig_row_index_before, n_after):
    n_before = len(orig_row_index_before)
    n_synthetic = n_after - n_before
    if n_synthetic < 0:
        raise RuntimeError(
            f"attach_orig_row_index_after_smote: n_after ({n_after}) < n_before ({n_before}) -- "
            f"SMOTE ไม่ควร undersample เลย ตรวจสอบ apply_smote"
        )
    synthetic_part = np.full(n_synthetic, ORIG_ROW_INDEX_SYNTHETIC_SENTINEL, dtype=np.int64)
    return np.concatenate([orig_row_index_before.to_numpy(), synthetic_part])


def main():
    section("PHASE 1: โหลด labeled + split (จาก Phase 1 ของ Option A: validate_behavioral_join.py)")
    labeled_df = safe_read_parquet(LABELED_PATH)
    log(f"  โหลด {LABELED_PATH}  shape={labeled_df.shape}")

    key_cols = [c for c in labeled_df.columns if c not in REFERENCE_COLS]
    log(f"  join key (79 คอลัมน์เดิม, ไม่รวม behavioral/orig_row_index/MITRE): {len(key_cols)} คอลัมน์")
    assert len(key_cols) == 79, f"คาดว่า key columns 79 คอลัมน์ แต่ได้ {len(key_cols)}"

    section("STEP 1: join คอลัมน์ MITRE กลับเข้า train/val/test (behavioral)")
    split_dfs = {}
    for split_name, path in SPLIT_PATHS.items():
        raw_split_df = safe_read_parquet(path)
        split_dfs[split_name] = join_labels_onto_split(raw_split_df, labeled_df, key_cols, REFERENCE_COLS, split_name)

    section("STEP 2: network feature columns (numeric, ไม่รวม identifier/label/orig_row_index)")
    network_feature_cols = get_network_feature_cols(split_dfs["train"], EXCLUDE_FROM_FEATURES)
    log(f"  network feature columns ({len(network_feature_cols)}) -- คาดว่า 90 (78 raw + 12 behavioral): {network_feature_cols}")
    assert len(network_feature_cols) == 90, f"คาดว่า 90 network feature columns แต่ได้ {len(network_feature_cols)}"

    section("STEP 3: impute คอลัมน์ sentinel (fit median จาก train เท่านั้น)")
    sentinel_medians = fit_sentinel_imputation(split_dfs["train"], SENTINEL_COLS)
    for c, med in sentinel_medians.items():
        log(f"  {c}: median (จาก train) = {med}")
    missing_flag_cols = [f"{c}_missing" for c in SENTINEL_COLS]
    for split_name in SPLIT_NAMES:
        split_dfs[split_name] = apply_sentinel_imputation(split_dfs[split_name], SENTINEL_COLS, sentinel_medians)

    section("STEP 5: MinMaxScaler (fit บน train เท่านั้น, เฉพาะ network features)")
    scaler = fit_scaler(split_dfs["train"], network_feature_cols)
    log(f"  fit บน train เท่านั้น: {len(network_feature_cols)} คอลัมน์ (ไม่แตะ {missing_flag_cols + EXTRA_OUTPUT_COLS})")
    for split_name in SPLIT_NAMES:
        split_dfs[split_name] = apply_scaler(split_dfs[split_name], network_feature_cols, scaler)

    final_feature_cols = model_input_feature_cols(network_feature_cols, missing_flag_cols)
    output_reference_cols = REFERENCE_COLS

    section("STEP 6: SMOTE (train เท่านั้น)")
    train_df = split_dfs["train"]
    X_train = train_df[final_feature_cols]
    y_train = train_df[LABEL_COL]
    ref_map, non_deterministic_labels = build_label_reference_map(train_df, LABEL_COL, output_reference_cols)

    X_train_res, y_train_res, smote_applied = apply_smote(X_train, y_train)

    if smote_applied:
        y_train_res = pd.Series(y_train_res, name=LABEL_COL).reset_index(drop=True)
        X_train_res = pd.DataFrame(X_train_res, columns=final_feature_cols).reset_index(drop=True)
        ref_df = attach_reference_columns(y_train_res, ref_map, output_reference_cols, non_deterministic_labels)
        orig_row_index_res = attach_orig_row_index_after_smote(train_df["orig_row_index"], len(X_train_res))
        n_synthetic = int((orig_row_index_res == ORIG_ROW_INDEX_SYNTHETIC_SENTINEL).sum())
        log(f"  orig_row_index: {n_synthetic:,} แถว synthetic ได้ sentinel={ORIG_ROW_INDEX_SYNTHETIC_SENTINEL} (ไม่มีตัวตนจริงใน cicids2017_val/test.parquet ให้ join)")
        train_features_df = pd.concat(
            [X_train_res, ref_df, y_train_res, pd.Series(orig_row_index_res, name="orig_row_index")], axis=1
        )
    else:
        train_features_df = pd.concat(
            [
                train_df[final_feature_cols].reset_index(drop=True),
                train_df[output_reference_cols].reset_index(drop=True),
                train_df[[LABEL_COL]].reset_index(drop=True),
                train_df[["orig_row_index"]].reset_index(drop=True),
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
                df[[LABEL_COL]].reset_index(drop=True),
                df[["orig_row_index"]].reset_index(drop=True),
            ],
            axis=1,
        )
        log(f"  cicids2017/{split_name}: ไม่ถูก SMOTE แตะ ({len(final_dfs[split_name]):,} แถว เท่ากับก่อน SMOTE, orig_row_index ทุกแถวเป็นค่าจริง)")

    section("STEP 7: เซฟผลลัพธ์")
    log(f"  network features: {len(network_feature_cols)}")
    log(f"  sentinel missing-flag features: {len(missing_flag_cols)} {missing_flag_cols}")
    log("  label-derived CTI encodings ไม่ถูกสร้างเป็น model input")
    log(f"  รวม feature columns ทั้งหมดต่อแถว (ไม่รวม reference/label/orig_row_index): {len(final_feature_cols)}")

    output_paths, manifest_path, history_dir = feature_extractor.write_feature_outputs(
        "cicids2017_behavioral",
        final_dfs,
        {
            "artifact_root": BEHAVIORAL_FEATURES_DIR,
            "output_paths": OUTPUT_PATHS,
            "source_paths": [LABELED_PATH, *SPLIT_PATHS.values()],
            "source_code_paths": [Path(__file__)],
        },
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
        log(f"  เซฟ cicids2017/{split_name} -> {path}  shape={final_dfs[split_name].shape}")
    log(f"  provenance manifest: {manifest_path}")
    if history_dir is not None:
        log(f"  preserved previous outputs: {history_dir}")

    section("สรุปผลลัพธ์")
    log(f"  SMOTE ถูกเรียกใช้จริงหรือไม่: {smote_applied}")
    log(f"  train rows: ก่อน SMOTE={len(train_df):,}  หลัง SMOTE={len(train_features_df):,}")
    log(f"  val rows={len(final_dfs['val']):,}  test rows={len(final_dfs['test']):,} (ไม่ถูก SMOTE แตะเลย)")
    log("  ยังไม่ retrain Isolation Forest/Autoencoder -- นั่นคือ phase ถัดไป (แผนแยกต่างหาก ยังไม่เริ่ม)")
    log("  orig_row_index carry ผ่านมาถึง output แล้ว (sentinel=-1 สำหรับแถว synthetic ใน train เท่านั้น) "
        "สำหรับ Fusion Layer ในอนาคต join คะแนน IF/AE (val/test เท่านั้น ไม่มี SMOTE) กับคะแนน RF ผ่าน "
        "orig_row_index <-> row_index แทนตำแหน่ง (position) เฉยๆ")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
