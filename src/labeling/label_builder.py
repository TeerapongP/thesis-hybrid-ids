"""
label_builder.py
==================
เติม context ของ MITRE ATT&CK ให้ข้อมูลที่ clean แล้ว (cicids2017_clean.parquet /
unswnb15_clean.parquet จาก preprocess_pipeline.py) โดย left-join กับตาราง mapping
คงที่ mitre_mapping.csv (dataset, source_label, technique_id, technique_name, tactic,
mapping_confidence) บนคอลัมน์ label ของแต่ละ dataset (Label สำหรับ CICIDS-2017,
attack_cat สำหรับ UNSW-NB15)

หมายเหตุ: นี่คือ mapping ตายตัวจากตาราง MITRE ที่กำหนดไว้ล่วงหน้า ไม่ใช่ weak-supervision
label ที่ได้จาก OpenCTI confidence score ตามที่ระบุไว้ใน diagrams (นั่นเป็นงานของ enrich.py +
OpenCTI connector ในอนาคต ไม่ใช่ไฟล์นี้)

ไม่แตะ/ไม่เรียกใช้ฟังก์ชัน clean/split/LOAO ใดๆ ใน preprocess_pipeline.py — อ่านแค่ผลลัพธ์
*_clean.parquet ของมันเป็น input เท่านั้น

ผลลัพธ์: print รายงานออกจอ, เซฟรายงานเป็น ./reports/label_builder_report.txt, เซฟข้อมูลที่ต่อ
MITRE mapping แล้วเป็น ./data/processed/cicids2017_labeled.parquet และ
./data/processed/unswnb15_labeled.parquet — รันจาก repo root เสมอ เช่น
`python -m src.labeling.label_builder`
"""

import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONFIG — สคริปต์นี้อยู่ใน src/ แต่ยังต้องรันจาก repo root เสมอ path ด้านล่างจึงอ้างอิงจาก
# repo root ไม่ใช่จากตำแหน่งไฟล์ .py เอง
# ----------------------------------------------------------------------------
DATA_DIR = "./data/processed"
MITRE_MAPPING_PATH = "./config/mitre_mapping.csv"

CICIDS_CLEAN_PATH = os.path.join(DATA_DIR, "cicids2017_clean.parquet")
UNSW_CLEAN_PATH = os.path.join(DATA_DIR, "unswnb15_clean.parquet")

CICIDS_LABELED_PATH = os.path.join(DATA_DIR, "cicids2017_labeled.parquet")
UNSW_LABELED_PATH = os.path.join(DATA_DIR, "unswnb15_labeled.parquet")

REPORT_PATH = "./reports/label_builder_report.txt"

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


def safe_read_csv(path, **kwargs):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' กรุณาตรวจสอบว่าตั้งค่า path ถูกต้อง "
            f"และไฟล์ถูกวางไว้ในตำแหน่งนั้นจริง"
        )
    return pd.read_csv(path, **kwargs)


def safe_read_parquet(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' — ต้องรัน preprocess_pipeline.py ก่อนเพื่อสร้างไฟล์นี้"
        )
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------
# apply_mitre_mapping: left-join df (เฉพาะแถวที่ mitre_mapping.csv ระบุ
# dataset == dataset_name) เข้ากับ mitre_mapping.csv บน label_col == source_label
# เพิ่มคอลัมน์ technique_id / technique_name / tactic / mapping_confidence
#
# แถวที่ label_col เป็น NaN/ว่าง (เช่น attack_cat ของ UNSW-NB15 traffic ปกติที่ไม่มี
# หมวดหมู่การโจมตี) จะไม่ match กับแถวไหนใน mapping table เลย -> คอลัมน์ทั้ง 4 ที่เพิ่มเข้ามา
# เป็น NaN ทั้งหมด ซึ่งเป็นพฤติกรรมที่ตั้งใจ (ไม่ error, ไม่ raise) ไม่ใช่ "unmapped" ในความหมาย
# ที่ต้องแก้ตาราง mapping เพราะแถวเหล่านี้ไม่มี label ให้ map ตั้งแต่แรก ("unmapped" ในรายงาน
# หมายถึงเฉพาะแถวที่ "มี" label แต่หาแถวคู่กันใน mitre_mapping.csv ไม่เจอเท่านั้น)
# ----------------------------------------------------------------------------
def apply_mitre_mapping(df, dataset_name, label_col):
    mapping_df = safe_read_csv(MITRE_MAPPING_PATH, dtype=str, keep_default_na=False, na_values=[""])
    subset = mapping_df.loc[mapping_df["dataset"] == dataset_name].drop(columns=["dataset"])

    df = df.copy()
    merge_key = "_label_builder_merge_key"
    df[merge_key] = df[label_col].astype("string")
    subset = subset.copy()
    subset["source_label"] = subset["source_label"].astype("string")

    merged = df.merge(
        subset,
        how="left",
        left_on=merge_key,
        right_on="source_label",
        indicator=True,
    )
    merged = merged.drop(columns=[merge_key, "source_label"])

    has_label = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
    matched = (merged["_merge"] == "both").to_numpy()
    unmapped_with_label = has_label.to_numpy() & ~matched

    n_total = len(df)
    n_no_label = int((~has_label).sum())
    n_mapped = int(matched.sum())
    n_unmapped_with_label = int(unmapped_with_label.sum())

    section(f"apply_mitre_mapping: dataset='{dataset_name}'  label_col='{label_col}'")
    log(f"  แถวทั้งหมด: {n_total}")
    log(f"  แถวที่ไม่มี label ให้ map (เช่น normal traffic ที่ไม่มี attack_cat): {n_no_label}")
    log(f"  แถวที่มี label และ match กับ mitre_mapping.csv สำเร็จ (mapped): {n_mapped}")
    log(f"  แถวที่มี label แต่หา match ใน mitre_mapping.csv ไม่เจอ (unmapped): {n_unmapped_with_label}")

    if n_unmapped_with_label > 0:
        missing_labels = sorted(
            df.loc[unmapped_with_label, label_col].astype(str).unique().tolist()
        )
        log(
            f"  [WARNING] พบ label ที่ไม่มีอยู่ใน mitre_mapping.csv สำหรับ dataset='{dataset_name}': "
            f"{missing_labels} -> mitre_mapping.csv ยังไม่ครอบคลุมครบ ต้องเพิ่มแถวให้ตรงกับ "
            f"label เหล่านี้ก่อนส่งต่อให้ feature_extractor.py"
        )
    else:
        log("  ยืนยัน: ไม่มี label ไหนหลุด mapping (unmapped=0) — mitre_mapping.csv ครอบคลุมครบ")

    log("  --- breakdown mapping_confidence (รวมแถวไม่มี label เป็น NaN) ---")
    conf_counts = merged["mapping_confidence"].value_counts(dropna=False)
    for conf_val, n in conf_counts.items():
        label_display = "NaN (ไม่มี label ให้ map)" if pd.isna(conf_val) else conf_val
        log(f"    {str(label_display):35s} n={n:>9d}")

    merged = merged.drop(columns=["_merge"])
    return merged


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    section("STEP 1: ตรวจสอบ mitre_mapping.csv")
    mapping_df = safe_read_csv(MITRE_MAPPING_PATH, dtype=str, keep_default_na=False, na_values=[""])
    log(f"  โหลด {MITRE_MAPPING_PATH} shape={mapping_df.shape}")
    log(f"  dataset ที่มีในตาราง: {sorted(mapping_df['dataset'].unique().tolist())}")

    section("STEP 2: โหลด cicids2017_clean.parquet")
    cicids_df = safe_read_parquet(CICIDS_CLEAN_PATH)
    log(f"  shape={cicids_df.shape}")

    section("STEP 3: apply_mitre_mapping (CICIDS-2017, label_col='Label')")
    cicids_labeled_df = apply_mitre_mapping(cicids_df, "cicids2017", "Label")

    os.makedirs(DATA_DIR, exist_ok=True)
    cicids_labeled_df.to_parquet(CICIDS_LABELED_PATH, index=False)
    log(f"เซฟ {CICIDS_LABELED_PATH}")

    section("STEP 4: โหลด unswnb15_clean.parquet")
    unsw_df = safe_read_parquet(UNSW_CLEAN_PATH)
    log(f"  shape={unsw_df.shape}")

    section("STEP 5: apply_mitre_mapping (UNSW-NB15, label_col='attack_cat')")
    unsw_labeled_df = apply_mitre_mapping(unsw_df, "unswnb15", "attack_cat")

    unsw_labeled_df.to_parquet(UNSW_LABELED_PATH, index=False)
    log(f"เซฟ {UNSW_LABELED_PATH}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
