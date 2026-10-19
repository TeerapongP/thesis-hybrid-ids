
import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = "./data/processed"
MITRE_MAPPING_PATH = "./config/mitre_mapping.csv"

CICIDS_CLEAN_PATH = os.path.join(DATA_DIR, "cicids2017_clean.parquet")
UNSW_CLEAN_PATH = os.path.join(DATA_DIR, "unswnb15_clean.parquet")

CICIDS_LABELED_PATH = os.path.join(DATA_DIR, "cicids2017_labeled.parquet")
UNSW_LABELED_PATH = os.path.join(DATA_DIR, "unswnb15_labeled.parquet")

REPORT_PATH = "./reports/label_builder_report.txt"

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
