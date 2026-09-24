"""
build_cross_dataset_eval.py
============================
สร้าง feature matrix สำหรับ M1c (cross-dataset evaluation) เท่านั้น — ไม่แตะ/ไม่เขียนทับ
cicids2017_features_{train,val,test}.parquet (84 cอลัมน์) หรือ
unswnb15_features_{train,val,test}.parquet (80 คอลัมน์) ที่ feature_extractor.py เซฟไว้ ไฟล์
เหล่านั้นยังเป็นชุด feature เต็มที่ใช้เทรน/ประเมินภายใน dataset เดิม (M1a, M1b, LOAO) เหมือนเดิม
ทุกประการ

สคริปต์นี้แค่ "อ่าน" ทั้ง 6 ไฟล์ output ของ feature_extractor.py จาก data/features/ แล้วเขียน
ไฟล์ใหม่คู่ขนาน {dataset}_features_{split}_crossdataset.parquet ที่มีเฉพาะคอลัมน์ที่ทั้งสอง
dataset มีร่วมกัน (intersection ของชื่อคอลัมน์ใน train ของทั้งสองฝั่ง) เรียงลำดับเดียวกันทุกไฟล์
(ลำดับคอลัมน์มีผลกับโมเดลที่เทรนจาก dataset หนึ่งแล้วเอาไปสกอร์อีก dataset หนึ่ง) เพื่อให้ M1c
เทรนบน dataset หนึ่งแล้วประเมินข้าม dataset ได้โดยไม่ชน column mismatch — ตัด Destination Port,
Fwd Header Length.1 (มีแต่ใน CICIDS-2017) และ Init_Win_bytes_forward/backward_missing (flag
ของ sentinel column ที่ UNSW-NB15 ไม่มีอีกต่อไปหลัง migrate ไป CIC-UNSW-NB15) ออกจากฝั่ง
CICIDS-2017 ส่วน UNSW-NB15 ไม่มีคอลัมน์ไหนถูกตัดเพิ่ม (ไม่มีคอลัมน์ที่ CICIDS ไม่มีอยู่แล้ว
ยกเว้น attack_cat ซึ่งเป็นคอลัมน์อ้างอิง ไม่ใช่ label หลัก จึงไม่อยู่ใน intersection)

รันหลัง feature_extractor.py (ต้องมีไฟล์ *_features_{train,val,test}.parquet ครบทั้ง 6 ไฟล์แล้ว):
    python -m src.evaluation.build_cross_dataset_eval
"""

import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATURES_DATA_DIR = "./data/features"
SPLIT_NAMES = ["train", "val", "test"]
DATASET_KEYS = ["cicids2017", "unswnb15"]

# คอลัมน์ที่ไม่ใช่ model-input feature จริง (label หลัก + คอลัมน์อ้างอิง MITRE) — ใช้แค่ตอนสรุป
# รายงานตอนท้าย ไม่ได้ใช้กรองตอนเขียนไฟล์ (ไฟล์ crossdataset ยังเก็บคอลัมน์เหล่านี้ไว้เหมือนไฟล์
# หลัก เพื่อให้ยัง map label/อ้างอิง MITRE ได้)
NON_FEATURE_COLS = ["Label", "technique_id", "technique_name", "tactic", "mapping_confidence"]

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


def output_path(dataset_key, split_name):
    return os.path.join(FEATURES_DATA_DIR, f"{dataset_key}_features_{split_name}_crossdataset.parquet")


def safe_read_parquet(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' — ต้องรัน feature_extractor.py ให้ครบก่อน (สร้างไฟล์นี้)"
        )
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------
# หาคอลัมน์ร่วม (intersection) จาก train ของทั้งสอง dataset แล้วเขียนไฟล์ *_crossdataset.parquet
# คู่ขนานสำหรับทั้ง 6 ไฟล์ (train/val/test x 2 dataset) โดยไม่แก้ไฟล์ input เดิมเลย
# ----------------------------------------------------------------------------
def build_cross_dataset_eval_sets():
    section("STEP 1 — โหลด train ของทั้งสอง dataset เพื่อหา shared column names (intersection)")
    cic_train = safe_read_parquet(input_path("cicids2017", "train"))
    unsw_train = safe_read_parquet(input_path("unswnb15", "train"))

    cic_cols = list(cic_train.columns)
    unsw_cols_set = set(unsw_train.columns)
    # เรียงตามลำดับคอลัมน์ของ cicids2017 (เป็น canonical order เดียวที่ใช้กับทั้ง 6 ไฟล์ output)
    shared_cols = [c for c in cic_cols if c in unsw_cols_set]
    cic_only = [c for c in cic_cols if c not in unsw_cols_set]
    unsw_only = [c for c in unsw_train.columns if c not in set(cic_cols)]

    log(f"  cicids2017_features_train.parquet columns: {len(cic_cols)}")
    log(f"  unswnb15_features_train.parquet columns: {len(unsw_train.columns)}")
    log(f"  shared columns (intersection, ตัดออกจากไฟล์ crossdataset ทั้งสองฝั่ง = ไม่ตัด, เก็บไว้): {len(shared_cols)}")
    log(f"  ตัดออกจากฝั่ง cicids2017 เท่านั้น (ไม่มีใน unsw): {cic_only}")
    log(f"  ตัดออกจากฝั่ง unswnb15 เท่านั้น (ไม่มีใน cicids): {unsw_only}")

    section("STEP 2 — เขียน {dataset}_features_{split}_crossdataset.parquet (คอลัมน์/ลำดับเดียวกันทุกไฟล์)")
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    for dataset_key in DATASET_KEYS:
        for split_name in SPLIT_NAMES:
            in_p = input_path(dataset_key, split_name)
            df = safe_read_parquet(in_p)
            missing = [c for c in shared_cols if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"{dataset_key}/{split_name}: ขาดคอลัมน์ที่ควรมีร่วมกันตาม shared_cols: {missing}"
                )
            out_df = df[shared_cols]
            out_p = output_path(dataset_key, split_name)
            out_df.to_parquet(out_p, index=False)
            log(f"  เซฟ {dataset_key}/{split_name} -> {out_p}  shape={out_df.shape}  (input {in_p} ไม่ถูกแก้ไข)")

    return shared_cols


# ----------------------------------------------------------------------------
# ยืนยันหลังเขียนไฟล์: column names + order ตรงกัน 100% ระหว่างสอง dataset ในไฟล์ crossdataset
# ----------------------------------------------------------------------------
def verify_column_alignment():
    section("STEP 3 — ยืนยัน column names/order ตรงกัน 100% ระหว่าง train ของทั้งสอง dataset (crossdataset)")
    cic_cd = safe_read_parquet(output_path("cicids2017", "train"))
    unsw_cd = safe_read_parquet(output_path("unswnb15", "train"))
    cic_cd_cols = list(cic_cd.columns)
    unsw_cd_cols = list(unsw_cd.columns)

    log(f"  cicids2017_features_train_crossdataset.parquet columns ({len(cic_cd_cols)}):")
    log(f"    {cic_cd_cols}")
    log(f"  unswnb15_features_train_crossdataset.parquet columns ({len(unsw_cd_cols)}):")
    log(f"    {unsw_cd_cols}")

    match = cic_cd_cols == unsw_cd_cols
    log(f"  ชื่อคอลัมน์และลำดับตรงกันทุกตำแหน่งหรือไม่: {match}")
    if not match:
        raise RuntimeError(
            "build_cross_dataset_eval.py: column names/order ไม่ตรงกันระหว่างสอง dataset ใน "
            "ไฟล์ crossdataset — ห้ามใช้ไฟล์นี้เทรนข้าม dataset จนกว่าจะแก้"
        )
    return cic_cd_cols


def main():
    shared_cols = build_cross_dataset_eval_sets()
    final_cols = verify_column_alignment()

    section("สรุป")
    model_input_feature_cols = [c for c in final_cols if c not in NON_FEATURE_COLS]
    reference_cols_present = [c for c in final_cols if c in ("technique_id", "technique_name", "tactic", "mapping_confidence")]
    label_cols_present = [c for c in final_cols if c == "Label"]

    log(f"  รวมคอลัมน์ในไฟล์ crossdataset ทุกไฟล์: {len(final_cols)}")
    log(f"    - model-input feature จริง (network + CTI, ไม่รวม label/reference): {len(model_input_feature_cols)}")
    log(f"    - reference columns (MITRE, เก็บไว้อ้างอิง ไม่ใช่ input ให้โมเดล): {len(reference_cols_present)} {reference_cols_present}")
    log(f"    - label column: {len(label_cols_present)} {label_cols_present}")
    log("  ไฟล์ input เดิมทั้ง 6 ไฟล์ (cicids2017_features_*.parquet 84 คอลัมน์ / "
        "unswnb15_features_*.parquet 80 คอลัมน์) ไม่ถูกแก้ไข — สคริปต์นี้เปิดอ่านอย่างเดียว "
        "แล้วเขียนไฟล์ใหม่แยกต่างหากเท่านั้น")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
