"""
build_cross_dataset_eval.py
============================
สร้าง feature matrix สำหรับ M1c (cross-dataset evaluation) เท่านั้น — ไม่แตะ/ไม่เขียนทับ
cicids2017_features_{train,val,test}.parquet หรือ unswnb15_features_{train,val,test}.parquet
ที่ feature_extractor.py เซฟไว้ ไฟล์เหล่านั้นยังเป็นชุด feature เต็มที่ใช้เทรน/ประเมินภายใน
dataset เดิม (M1a, M1b, LOAO) เหมือนเดิมทุกประการ ไฟล์ *_features_{split}_crossdataset.parquet
ชุดเดิมจากวิธีเก่า (intersection ของชื่อคอลัมน์บนไฟล์ที่ scale แล้ว) ก็ไม่ถูกแตะเช่นกัน

วิธีที่ใช้ตอนนี้ต่างจากวิธีเก่า: เดิมแค่หา intersection ของชื่อคอลัมน์ใน train ของสอง dataset
แล้วตัดไฟล์ที่ MinMaxScaler ของแต่ละ dataset scale ไว้ด้วยตัวเอง — ทำให้ flow เดียวกันได้ค่าต่างกัน
ในสองฝั่ง ชื่อคอลัมน์ตรงกันแต่ค่าไม่ได้อยู่บนสเกลเดียวกัน สคริปต์นี้จึงอ่าน split ดิบจาก
data/processed/ แล้วในแต่ละทิศทาง (cicids2017->unswnb15 และ unswnb15->cicids2017) fit
sentinel imputation และ MinMaxScaler บน train ของ SOURCE แล้วใช้ transform เดียวกันกับ test
ของ TARGET ผ่านฟังก์ชันของ feature_extractor.py เอง (fit_sentinel_imputation, fit_scaler ฯลฯ)
เพื่อให้ได้ model input ชุดเดียวกัน ลำดับเดียวกัน ทั้งสองทิศทาง

คอลัมน์ที่เป็น model input = ชื่อคอลัมน์ที่ทั้งสอง dataset มีร่วมกัน ลบคอลัมน์ label/อ้างอิง MITRE,
attack_cat, CTI encodings 4 คอลัมน์ที่เป็นฟังก์ชันของ label จริง และ row_index/orig_row_index
ออก ดังนั้น Destination Port, Fwd Header Length.1 และ flag Init_Win_bytes_*_missing (มีแต่ใน
CICIDS-2017) ไม่อยู่ในชุดร่วม (76 คอลัมน์ตามที่วัดไว้ใน reports/build_cross_dataset_eval_report.txt)
ลำดับคอลัมน์มีผลกับโมเดลที่เทรนจาก dataset หนึ่งแล้วเอาไปสกอร์อีก dataset หนึ่ง จึงตรวจว่า
ชื่อและลำดับตรงกันทั้งสองทิศทางก่อนจบ

Output: data/features/crossdataset_eval/{source}_to_{target}_test.parquet (ไฟล์ test ของ target
ที่ scale ด้วย scaler ของ source + คอลัมน์ Label/อ้างอิง MITRE ไว้ map label) ผู้ใช้หลักคือ
eval_cross_dataset.py

รันหลัง feature_extractor.py (ต้องมี *_features_{train,val,test}.parquet และ
data/processed/{dataset}_{split}.parquet, *_labeled.parquet):
    python -m src.evaluation.build_cross_dataset_eval
"""

import os
import sys

import pandas as pd

from src.features import feature_extractor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATURES_DATA_DIR = "./data/features"
PROCESSED_DATA_DIR = "./data/processed"
CROSS_DATASET_OUTPUT_DIR = os.path.join(FEATURES_DATA_DIR, "crossdataset_eval")
DATASET_KEYS = ["cicids2017", "unswnb15"]
REFERENCE_COLS = ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"]
LABEL_DERIVED_COLS = [
    "technique_id_encoded",
    "tactic_encoded",
    "mapping_confidence_encoded",
    "is_known_technique",
]
NON_FEATURE_COLS = [
    *REFERENCE_COLS,
    "attack_cat",
    *LABEL_DERIVED_COLS,
    "row_index",
    "orig_row_index",
]

REPORT_PATH = "./reports/build_cross_dataset_eval_report.txt"
_REPORT_LINES = []


def log(msg=""):
    msg = str(msg)
    print(msg)
    _REPORT_LINES.append(msg)


def section(title):
    log("=" * 70)
    log(title)
    log("=" * 70)


def input_path(dataset_key, split_name):
    return os.path.join(FEATURES_DATA_DIR, f"{dataset_key}_features_{split_name}.parquet")


def raw_input_path(dataset_key, split_name):
    return os.path.join(PROCESSED_DATA_DIR, f"{dataset_key}_{split_name}.parquet")


def labeled_input_path(dataset_key):
    return feature_extractor.DATASETS[dataset_key]["labeled_path"]


def output_path(source_dataset, target_dataset):
    return os.path.join(
        CROSS_DATASET_OUTPUT_DIR,
        f"{source_dataset}_to_{target_dataset}_test.parquet",
    )


def model_input_feature_cols(columns):
    return [column for column in columns if column not in NON_FEATURE_COLS]


def safe_read_parquet(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' — ต้องรัน preprocess_pipeline.py, label_builder.py และ feature_extractor.py ก่อน"
        )
    return pd.read_parquet(path)


def scale_target_with_source_training(
    source_train_df,
    target_df,
    model_feature_cols,
    sentinel_cols=(),
):
    network_cols = [column for column in model_feature_cols if not column.endswith("_missing")]
    missing_flag_cols = [column for column in model_feature_cols if column.endswith("_missing")]
    if not network_cols:
        raise ValueError("cross-dataset input has no shared network features")

    required_sentinel_flags = {column.removesuffix("_missing") for column in missing_flag_cols}
    shared_sentinels = [
        column
        for column in sentinel_cols
        if column in network_cols and column in required_sentinel_flags
    ]
    unexpected_flags = required_sentinel_flags - set(shared_sentinels)
    if unexpected_flags:
        raise ValueError(f"cannot derive shared missing flags: {sorted(unexpected_flags)}")

    missing_source = [column for column in network_cols if column not in source_train_df]
    missing_target = [column for column in network_cols if column not in target_df]
    if missing_source or missing_target:
        raise ValueError(
            f"raw cross-dataset inputs missing network columns: source={missing_source}, target={missing_target}"
        )

    sentinel_medians = feature_extractor.fit_sentinel_imputation(source_train_df, shared_sentinels)
    source_imputed = feature_extractor.apply_sentinel_imputation(
        source_train_df, shared_sentinels, sentinel_medians
    )
    source_scaler = feature_extractor.fit_scaler(source_imputed, network_cols)

    target_imputed = feature_extractor.apply_sentinel_imputation(
        target_df, shared_sentinels, sentinel_medians
    )
    target_scaled = feature_extractor.apply_scaler(target_imputed, network_cols, source_scaler)
    return target_scaled[model_feature_cols], source_scaler


def _load_referenced_raw_split(dataset_key, split_name):
    raw_split = safe_read_parquet(raw_input_path(dataset_key, split_name))
    labeled = safe_read_parquet(labeled_input_path(dataset_key))
    reference_cols = feature_extractor.DATASETS[dataset_key]["reference_cols"]
    return feature_extractor.join_labels_onto_split(
        raw_split, labeled, reference_cols, dataset_key, split_name
    )


def _build_direction(source_dataset, target_dataset):
    source_features = safe_read_parquet(input_path(source_dataset, "train"))
    target_features = safe_read_parquet(input_path(target_dataset, "test"))
    target_feature_set = set(target_features.columns)
    shared_features = model_input_feature_cols(
        [column for column in source_features.columns if column in target_feature_set]
    )
    if not shared_features:
        raise RuntimeError(f"{source_dataset} and {target_dataset} have no shared model input features")

    source_train = safe_read_parquet(raw_input_path(source_dataset, "train"))
    target_raw = _load_referenced_raw_split(target_dataset, "test")
    target_scaled, scaler = scale_target_with_source_training(
        source_train,
        target_raw,
        shared_features,
        feature_extractor.DATASETS[source_dataset]["sentinel_cols"],
    )
    reference_cols = [column for column in REFERENCE_COLS if column in target_raw.columns]
    target_output = pd.concat(
        [target_scaled, target_raw[reference_cols]],
        axis=1,
    )
    destination = output_path(source_dataset, target_dataset)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    target_output.to_parquet(destination, index=False)
    log(
        f"  {source_dataset} -> {target_dataset}: source-train scaler fit on "
        f"{len(shared_features)} model features; wrote target test {target_output.shape} to {destination}"
    )
    return shared_features


def build_cross_dataset_eval_sets():
    section("STEP 1 — สร้าง target test ด้วย scaler ที่ fit จาก source train ในแต่ละทิศทาง")
    directions = [("cicids2017", "unswnb15"), ("unswnb15", "cicids2017")]
    feature_cols_by_direction = {
        (source, target): _build_direction(source, target)
        for source, target in directions
    }
    first_cols = feature_cols_by_direction[directions[0]]
    if any(columns != first_cols for columns in feature_cols_by_direction.values()):
        raise RuntimeError("model input feature names/order differ between cross-dataset directions")
    return first_cols


def verify_column_alignment(feature_cols):
    section("STEP 2 — ตรวจคอลัมน์ model input ทั้งสองทิศทาง")
    for source_dataset, target_dataset in (("cicids2017", "unswnb15"), ("unswnb15", "cicids2017")):
        cross_df = safe_read_parquet(output_path(source_dataset, target_dataset))
        actual = model_input_feature_cols(cross_df.columns)
        if actual != feature_cols:
            raise RuntimeError(
                f"{source_dataset} -> {target_dataset}: model input feature names/order do not match"
            )
        log(f"  {source_dataset} -> {target_dataset}: {len(actual)} aligned model input columns")
    return feature_cols


def main():
    model_features = build_cross_dataset_eval_sets()
    final_cols = verify_column_alignment(model_features)

    section("สรุป")
    log(f"  model input features (network only): {len(final_cols)}")
    log(f"  label-derived CTI fields excluded from detector input: {LABEL_DERIVED_COLS}")
    log(f"  target outputs: {output_path('cicids2017', 'unswnb15')}, {output_path('unswnb15', 'cicids2017')}")
    log("  crossdataset artifacts from the previous implementation remain untouched")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
