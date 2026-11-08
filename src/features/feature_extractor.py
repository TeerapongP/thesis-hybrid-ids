"""
feature_extractor.py
======================
สร้าง feature matrix พร้อมใช้กับโมเดล ML (Isolation Forest / Autoencoder / Random Forest)
จากผลลัพธ์ของ preprocess_pipeline.py (*_train/val/test.parquet ที่ split ไว้แล้ว) และ
label_builder.py (*_labeled.parquet ที่มีคอลัมน์ MITRE: technique_id, technique_name, tactic,
mapping_confidence ต่อกับข้อมูล full clean ไว้แล้ว) — ไม่แตะ/ไม่เรียกใช้ฟังก์ชันใดใน
preprocess_pipeline.py หรือ label_builder.py เลย อ่านแค่ไฟล์ parquet ผลลัพธ์ของทั้งสองไฟล์
เป็น input เท่านั้น

ขั้นตอนหลัก:
  1) join คอลัมน์ MITRE (technique_id ฯลฯ) จาก *_labeled.parquet กลับเข้าไปใน
     *_train/val/test.parquet ที่ split ไว้แล้ว (ไม่ split ใหม่) — split_train_val_test()
     ของ preprocess_pipeline.py ทำ reset_index(drop=True) ก่อนเซฟไฟล์ ทำให้ index เดิมที่จะ
     ใช้ join แบบตรงไปตรงมาหายไป จึง join ด้วยการจับคู่ "ค่าทุกคอลัมน์ดั้งเดิม" แทน (พิสูจน์ได้ว่า
     unique ต่อแถวเสมอ เพราะ clean_common() ของ preprocess_pipeline.py ทำ drop_duplicates()
     ด้วยคอลัมน์ครบชุดไปแล้ว) — เพื่อความเร็ว ใช้ pd.util.hash_pandas_object() แฮชคอลัมน์
     ทั้งหมดเป็นคอลัมน์เดียวก่อน join (แทนการ join ด้วยหลายสิบคอลัมน์พร้อมกันตรงๆ) แล้วตรวจสอบ
     ด้วย merge(validate="1:1", indicator=True) ว่า join สำเร็จ 100% เสมอ — pandas merge จับคู่
     NaN กับ NaN ว่า "เท่ากัน" ได้เอง (ทดสอบแล้ว) จึงไม่ต้องกันคอลัมน์ sentinel (NaN โดยการ
     ออกแบบ) ออกจาก join key เป็นพิเศษ
  2) network features = คอลัมน์ตัวเลข (numeric dtype) ทั้งหมดที่มีอยู่แล้วในข้อมูล ไม่ hand-pick
     ยกเว้นคอลัมน์ identifier/label (Label, attack_cat, technique_id, technique_name, tactic,
     mapping_confidence)
  3) คอลัมน์ sentinel ที่มี NaN โดยการออกแบบ (Init_Win_bytes_forward/backward ของ CICIDS —
     ดู preprocess_pipeline.py; CIC-UNSW-NB15 ไม่มีคอลัมน์ sentinel จึง sentinel_cols = [])
     ยังไม่เคย impute มาก่อนหน้านี้เลย ต้อง impute ที่นี่ก่อนถึงจะเข้า MinMaxScaler/SMOTE ได้
     (ทั้งสองอย่างต้องการข้อมูลตัวเลขที่ไม่มี NaN) — impute ด้วยค่า median ที่ fit จาก train
     เท่านั้น พร้อมเพิ่มคอลัมน์ flag คู่กัน (<col>_missing) เพื่อไม่ให้ข้อมูล "ไม่ได้วัดค่า"
     หายไปเฉยๆ ตามที่ preprocess_pipeline.py ระบุไว้ว่าให้ "impute/flag" ในไฟล์นี้
  4) (ไม่ทำที่นี่แล้ว) CTI features 4 คอลัมน์ — technique_id_encoded, tactic_encoded,
     mapping_confidence_encoded, is_known_technique — เป็นฟังก์ชันของ label จริงผ่าน
     config/mitre_mapping.csv จึงไม่ใช่ข้อมูลที่รู้ได้ ณ เวลา detect และไม่ถูกสร้างเป็น model
     input อีก ฟังก์ชัน fit_integer_encoding/apply_integer_encoding/
     apply_mapping_confidence_encoding/build_is_known_technique ยังอยู่ในไฟล์นี้เพราะ
     build_unified_population.py เรียกใช้ เพื่อเก็บเป็นคอลัมน์อ้างอิงสำหรับชั้น LLM
  5) MinMaxScaler fit บน train เท่านั้น สเกลเฉพาะ network features (รวมคอลัมน์ sentinel ที่
     impute แล้ว) ไม่สเกลคอลัมน์ flag คอลัมน์ missing (เป็น binary)
  6) SMOTE (fit บน train เท่านั้น ไม่แตะ val/test เด็ดขาด) — ใช้ sampling_strategy แบบมีเพดาน
     (ดู SMOTE_FLAT_CAP/SMOTE_TARGET_MULTIPLIER) แทน 'auto' เต็มรูปแบบ เพราะ class ใหญ่สุด
     (BENIGN ของ CICIDS-2017 train ~1.26M แถว) กับเล็กสุด (Heartbleed 7 แถว) ต่างกันมาก
     ~180,000 เท่า — SMOTE แบบ auto จะขยาย train จาก ~1.5M เป็น ~18.9M แถว ซึ่งหนักเกินจำเป็น
     และสร้าง synthetic sample จาก sample จริงแค่ 7-13 ตัวจำนวนมหาศาล (ตัดสินใจร่วมกับผู้ใช้แล้ว)
  7) เซฟผลลัพธ์เป็น {dataset}_features_{split}.parquet ไปที่ data/features/ (คนละโฟลเดอร์กับ
     input ที่อ่านจาก data/processed/) ผ่าน write_feature_outputs(): เขียนไฟล์ชั่วคราวข้าง
     ปลายทางก่อน คัดลอก output เดิมไว้ใน data/features/history/ แล้วจึง os.replace (ถ้า
     promote ไม่สำเร็จจะ rollback) และเขียน manifest ใน data/features/provenance/ (sha256
     ของ input, source code, uv.lock และ output)
"""

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

# ----------------------------------------------------------------------------
# CONFIG — สคริปต์นี้อยู่ใน src/features/ แต่ยังต้องรันจาก repo root เสมอ (เช่น
# `python -m src.features.feature_extractor`) path ด้านล่างจึงอ้างอิงจาก repo root
# PROCESSED_DATA_DIR = input (labeled + train/val/test จาก preprocess_pipeline.py/label_builder.py)
# FEATURES_DATA_DIR = output ของไฟล์นี้เอง (คนละโฟลเดอร์กับ input โดยตั้งใจ)
# ----------------------------------------------------------------------------
PROCESSED_DATA_DIR = "./data/processed"
FEATURES_DATA_DIR = "./data/features"
SPLIT_NAMES = ["train", "val", "test"]

MAPPING_CONFIDENCE_ENCODING = {"n/a": 0, "low": 1, "medium": 2, "high": 3}

# เพดาน SMOTE: ไม่แตะ class ที่ >= SMOTE_FLAT_CAP อยู่แล้ว, ที่เหลือขยายไปที่
# min(majority_count, ขนาดเดิม x SMOTE_TARGET_MULTIPLIER, SMOTE_FLAT_CAP)
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
        # reference_cols: คอลัมน์ MITRE ที่ label_builder.py เพิ่มเข้ามา ต้อง join จาก
        # *_labeled.parquet กลับเข้ามา (ไม่มีอยู่ใน *_train/val/test.parquet เดิม)
        "reference_cols": ["technique_id", "technique_name", "tactic", "mapping_confidence"],
        # extra_output_cols: คอลัมน์ที่มีอยู่ใน *_train/val/test.parquet อยู่แล้ว (ไม่ต้อง join)
        # แค่พกไปกับ output ด้วยเฉยๆ เพื่อการอ้างอิง — CICIDS ไม่มีคอลัมน์แบบนี้เพิ่มเติม (Label
        # ทำหน้าที่ทั้ง label หลักและ attack category อยู่แล้วในคอลัมน์เดียว)
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
        # attack_cat มีอยู่ใน unswnb15_{train,val,test}.parquet อยู่แล้ว (ไม่ใช่คอลัมน์ที่
        # label_builder.py เพิ่ม) ถ้าใส่ไว้ใน reference_cols ด้วยจะทำให้ตอน join เกิดคอลัมน์
        # "attack_cat" ซ้ำ 2 ตัว (จาก key_cols และจาก reference_cols) และทำให้ join พังทั้งหมด
        # (แถวหา match ไม่เจอเลยสักแถว) — ต้องแยกไว้เป็น extra_output_cols แทน
        "extra_output_cols": ["attack_cat"],
        "sentinel_cols": [],
    },
}

REPORT_PATH = "./reports/feature_extractor_report.txt"

# ----------------------------------------------------------------------------
# ตัวช่วยเก็บ output
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# STEP 1: join คอลัมน์ MITRE จาก labeled_df กลับเข้า split_df (ไม่ split ใหม่)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# STEP 2: network feature columns = numeric dtype ทั้งหมด ไม่รวม identifier/label
# ----------------------------------------------------------------------------
def get_network_feature_cols(df, exclude_cols):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]


# ----------------------------------------------------------------------------
# STEP 3: impute คอลัมน์ sentinel (NaN โดยการออกแบบ) ด้วย median ที่ fit จาก train เท่านั้น
# พร้อมเพิ่มคอลัมน์ <col>_missing (1 = เดิมเป็น NaN, 0 = มีค่าจริง) กันข้อมูล "ไม่ได้วัดค่า" หาย
# ----------------------------------------------------------------------------
def fit_sentinel_imputation(train_df, sentinel_cols):
    return {c: train_df[c].median(skipna=True) for c in sentinel_cols}


def apply_sentinel_imputation(df, sentinel_cols, medians):
    df = df.copy()
    for c in sentinel_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(medians[c])
    return df


# ----------------------------------------------------------------------------
# STEP 4: CTI features — ไม่ถูกเรียกใน process_dataset() แล้ว (ดู docstring ด้านบน) ฟังก์ชันด้านล่างเก็บไว้ให้ build_unified_population.py ใช้
# ----------------------------------------------------------------------------
def fit_integer_encoding(train_series):
    """null -> 0 (ไม่ต้องเก็บใน mapping) ค่าที่ไม่ null แต่ละค่าที่ไม่ซ้ำใน train (sort ตาม
    ตัวอักษรเพื่อให้ reproducible) -> 1, 2, 3, ... ตามลำดับ"""
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


# ----------------------------------------------------------------------------
# STEP 5: MinMaxScaler fit บน train เท่านั้น เฉพาะ network feature columns
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# STEP 6: SMOTE (train เท่านั้น) แบบมีเพดาน (ดู docstring ด้านบนของไฟล์)
# ----------------------------------------------------------------------------
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
    """คืน (ref_map, non_deterministic_labels) — ref_map: label value -> dict ของ
    reference_cols เฉพาะ label ที่ map ไปที่ชุดค่า reference_cols เดียวกันเสมอ (deterministic
    1:1) เท่านั้น ถ้า label ใดมีหลายชุดค่า reference_cols ปนกัน (เช่น UNSW ที่ Label ไบนารี
    ครอบคลุมหลาย attack_cat) จะไม่ใส่ label นั้นไว้ใน ref_map เลย ให้ caller เติม NaN แทนการเดา"""
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


# ----------------------------------------------------------------------------
# ประมวลผลทั้ง 1 dataset ครบวงจร (join -> impute -> scale -> SMOTE -> เซฟ)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
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
