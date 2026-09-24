"""
validate_behavioral_join.py
============================
สคริปต์แยกต่างหาก (ไม่แก้ preprocess_pipeline.py / label_builder.py / feature_extractor.py /
behavioral_feature_extractor.py และไม่แตะ output เดิมของทุกไฟล์เลย) สำหรับ validate การ join
คอลัมน์ behavioral (จาก cicids2017_behavioral_labeled.parquet, สร้างจาก GeneratedLabelledFlows/
TrafficLabelling distribution) กลับเข้าไปใน cicids2017_{train,val,test}.parquet เดิม (สร้างจาก
MachineLearningCSV distribution) — นี่คือ Phase 1 ของ Option A เท่านั้น (การ join + validate)
ไม่รวม Phase 2 (ย้าย impute/CTI-encode/scale/SMOTE logic ของ feature_extractor.py มาใช้กับข้อมูล
ที่ join แล้ว หรือ retrain IF/AE) — Phase 2 เป็นแผนแยกต่างหากที่ต้องขออนุมัติเพิ่ม

ทำไมต้อง validate ก่อน join จริง: ทั้งสอง distribution เป็น CICFlowMeter export คนละรอบกันของ
capture เดียวกัน ไม่มี shared key column ใดๆ เลย (MachineLearningCSV ไม่มี Source IP/Timestamp,
ถูกตัดออกเพื่อความเป็นส่วนตัว) จึงต้อง join ด้วย row-hash บน 79 คอลัมน์ร่วม (78 raw CICFlowMeter
features + Label) เหมือนที่ feature_extractor.py ทำกับ join ของตัวเอง (คอลัมน์ MITRE กลับเข้า
split) แต่ต่างกันตรงที่ feature_extractor.py join ข้อมูลจาก "รอบ process เดียวกัน" (labeled_df
สร้างจาก clean_df ตัวเดียวกับ split_df) รับประกัน 100% match อยู่แล้วโดยไม่ต้องคิดอะไรเพิ่ม
แต่ที่นี่ทั้งสอง distribution ผลิตแยกกันอิสระ (คนละไฟล์ CSV ต้นทาง) การ match จึงไม่รับประกัน —
recon ก่อนหน้านี้เจอ "~163-row duplicate-count discrepancy" ระหว่างสอง distribution ซึ่งตอนแรก
ดูเหมือนเป็นแค่ edge case เล็กๆ

สิ่งที่ script นี้ยืนยัน (จาก diagnostic ที่รันไว้แล้วนอก script นี้ ก่อนเขียนโค้ดนี้):
  1) hash key 79 คอลัมน์ (list(cicids2017_clean-derived split.columns)) unique ในตัวเองทั้งสองฝั่ง
     (ไม่มี hash ซ้ำเลยแม้แต่ตัวเดียว ทั้งฝั่ง split และฝั่ง behavioral) -> validate="1:1" ใช้ได้จริง
     ไม่ใช่แค่ assumption
  2) hash ตรงๆ โดยไม่ normalize dtype ก่อน จะได้ 0% match (ไม่ใช่ ~163 แถวต่าง แต่พังทั้งหมด) เพราะ
     51/79 คอลัมน์มี dtype ไม่ตรงกัน (int64 ใน cicids2017_clean.parquet vs float64 ใน
     cicids2017_behavioral_labeled.parquet) เป็น artifact จากการที่แต่ละ script ผลิตไฟล์แยกกัน
     ไม่ใช่ข้อมูลต่างกันจริง -> ต้อง cast เป็น float64 เหมือนกันทั้งสองฝั่งก่อน hash เสมอ
  3) หลัง normalize dtype แล้ว: ทุกแถวที่เป็น attack (ไม่ใช่ BENIGN) match 100% ทั้ง 425,741 แถว
     ทุก 14 category — ส่วนต่างทั้งหมดกระจุกอยู่ที่ BENIGN เท่านั้น (~23.6% ของ BENIGN แต่ละฝั่ง
     ไม่ match กัน ปัดเศษหักลบกันเหลือ ~163 ตามที่ recon เจอ แต่ตัวเลขจริงที่ไม่ match กันใหญ่กว่านั้น
     มาก ไม่ใช่ edge case เล็กๆ)
  4) กระจายสม่ำเสมอทั้ง train/val/test (~23.6% ของ BENIGN ทุก split ใกล้เคียงกัน) ไม่กระจุกที่ split ใด
  5) นโยบายที่ตัดสินใจร่วมกับผู้ใช้: DROP แถว BENIGN ที่ไม่ match ออกจาก behavioral-augmented split
     แล้วบันทึกจำนวน/สัดส่วนไว้เป็น known limitation อย่างชัดเจน (ไม่ investigate root cause เพิ่ม)

Output:
  - data/processed/behavioral/cicids2017_behavioral_{train,val,test}.parquet (join แล้ว, ยังไม่ scale/
    SMOTE — ตำแหน่งเดียวกับ preprocess_pipeline.py's cicids2017_{train,val,test}.parquet, ก่อนขั้น
    impute/CTI-encode/scale/SMOTE ของ feature_extractor.py ซึ่งเป็น Phase 2 แยกต่างหาก)
  - reports/behavioral_join_validation_report.txt

รันจาก repo root เสมอ: `python -m src.features.validate_behavioral_join`
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
PROCESSED_DATA_DIR = "./data/processed"
BEHAVIORAL_DATA_DIR = "./data/processed/behavioral"
SPLIT_NAMES = ["train", "val", "test"]

SPLIT_PATHS = {s: os.path.join(PROCESSED_DATA_DIR, f"cicids2017_{s}.parquet") for s in SPLIT_NAMES}
BEHAVIORAL_LABELED_PATH = os.path.join(BEHAVIORAL_DATA_DIR, "cicids2017_behavioral_labeled.parquet")
OUTPUT_PATHS = {s: os.path.join(BEHAVIORAL_DATA_DIR, f"cicids2017_behavioral_{s}.parquet") for s in SPLIT_NAMES}
REPORT_PATH = "./reports/behavioral_join_validation_report.txt"

# คอลัมน์ behavioral 12 ตัวที่จะพกไปกับ output ด้วย (นอกเหนือจาก 79 key columns เดิม) — ไม่รวม
# technique_id/technique_name/tactic/mapping_confidence (มาจาก label_builder.py join แยกต่างหาก
# อยู่แล้วใน feature_extractor.py เดิม การพกมาด้วยที่นี่จะซ้ำซ้อน/เสี่ยงชนคอลัมน์) และไม่รวม
# Flow ID/Source IP/Destination IP/Timestamp/day_group (ไม่ใช่ส่วนหนึ่งของ schema pipeline หลัก)
BEHAVIORAL_FEATURE_COLS = [
    f"{metric}_{w}m"
    for w in (1, 3, 5)
    for metric in ("conn_count", "uniq_dport_count", "fail_ratio", "repeat_ratio")
]

# baseline ที่ review แล้วจาก diagnostic ที่รันไว้ก่อนเขียน script นี้ (ดู docstring ข้อ 3-4) —
# ใช้เทียบ drift หากรันใหม่ในอนาคต (เช่น หลัง raw data หรือ upstream script เปลี่ยน)
BASELINE_TOTAL_UNMATCHED_PCT = {"train": 19.59, "val": 19.65, "test": 19.64}
BASELINE_BENIGN_UNMATCHED_PCT = {"train": 23.57, "val": 23.64, "test": 23.63}
DRIFT_TOLERANCE_PP = 2.0  # percentage points

# ----------------------------------------------------------------------------
# ตัวช่วยเก็บ output (รูปแบบเดียวกับ script อื่นในโปรเจกต์)
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
            f"ไม่พบไฟล์: '{path}' -> ต้องรัน preprocess_pipeline.py และ behavioral_feature_extractor.py ก่อน"
        )
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------
# STEP 1: hash key -- normalize dtype ก่อน hash เสมอ (ดูเหตุผลใน docstring ข้อ 2)
# ----------------------------------------------------------------------------
def normalize_key_dtypes(df, key_cols, label_col="Label"):
    df = df[key_cols].copy()
    numeric_cols = [c for c in key_cols if c != label_col]
    for c in numeric_cols:
        df[c] = df[c].astype("float64")
    return df


def compute_row_hash(df, key_cols):
    return pd.util.hash_pandas_object(df[key_cols], index=False)


def assert_hash_unique(hash_series, side_name, split_name):
    dupe_counts = hash_series.value_counts()
    n_dupe_hashes = int((dupe_counts > 1).sum())
    if n_dupe_hashes > 0:
        n_rows_involved = int(dupe_counts[dupe_counts > 1].sum())
        raise RuntimeError(
            f"assert_hash_unique: {split_name}/{side_name} มี {n_dupe_hashes} hash ที่ซ้ำกัน "
            f"({n_rows_involved} แถวเกี่ยวข้อง) -> validate='1:1' จะไม่ผ่าน ต้องตรวจ key columns ใหม่"
        )


# ----------------------------------------------------------------------------
# STEP 2: diagnostic outer join ก่อน join จริงเสมอ (per split) -- ดูสัดส่วน unmatched + label breakdown
# ----------------------------------------------------------------------------
def diagnostic_outer_join(split_df, behavioral_df, key_cols, split_name):
    left_k = normalize_key_dtypes(split_df, key_cols)
    right_k = normalize_key_dtypes(behavioral_df, key_cols)

    assert left_k.dtypes.equals(right_k.dtypes), (
        f"diagnostic_outer_join: {split_name} dtype ของ key columns สองฝั่งไม่ตรงกันแม้ cast แล้ว "
        f"-> ตรวจ normalize_key_dtypes"
    )

    left_hash = compute_row_hash(left_k, key_cols)
    right_hash = compute_row_hash(right_k, key_cols)
    assert_hash_unique(left_hash, "split (clean-derived)", split_name)
    assert_hash_unique(right_hash, "behavioral", split_name)

    left = pd.DataFrame({"_h": left_hash, "Label": split_df["Label"].to_numpy()})
    right = pd.DataFrame({"_h": right_hash, "Label": behavioral_df["Label"].to_numpy()})

    merged = left.merge(right, on="_h", how="outer", suffixes=("_split", "_behav"), indicator=True)

    n_total = len(split_df)
    n_left_only = int((merged["_merge"] == "left_only").sum())
    n_matched = int((merged["_merge"] == "both").sum())
    pct_total_unmatched = 100 * n_left_only / n_total

    benign_total = int((split_df["Label"] == "BENIGN").sum())
    left_only_labels = merged.loc[merged["_merge"] == "left_only", "Label_split"]
    n_benign_unmatched = int((left_only_labels == "BENIGN").sum())
    n_nonbenign_unmatched = int((left_only_labels != "BENIGN").sum())
    pct_benign_unmatched = 100 * n_benign_unmatched / benign_total if benign_total else 0.0

    return {
        "split": split_name,
        "n_total": n_total,
        "n_matched": n_matched,
        "n_left_only": n_left_only,
        "pct_total_unmatched": pct_total_unmatched,
        "benign_total": benign_total,
        "n_benign_unmatched": n_benign_unmatched,
        "n_nonbenign_unmatched": n_nonbenign_unmatched,
        "pct_benign_unmatched": pct_benign_unmatched,
        "left_hash": left_hash,
        "right_hash": right_hash,
    }


def check_drift(diag, split_name):
    baseline_total = BASELINE_TOTAL_UNMATCHED_PCT[split_name]
    baseline_benign = BASELINE_BENIGN_UNMATCHED_PCT[split_name]
    drift_total = abs(diag["pct_total_unmatched"] - baseline_total)
    drift_benign = abs(diag["pct_benign_unmatched"] - baseline_benign)
    log(
        f"  drift check: total_unmatched {diag['pct_total_unmatched']:.2f}% "
        f"(baseline {baseline_total:.2f}%, drift {drift_total:.2f}pp)  |  "
        f"benign_unmatched {diag['pct_benign_unmatched']:.2f}% "
        f"(baseline {baseline_benign:.2f}%, drift {drift_benign:.2f}pp)"
    )
    if drift_total > DRIFT_TOLERANCE_PP or drift_benign > DRIFT_TOLERANCE_PP:
        raise RuntimeError(
            f"check_drift: {split_name} unmatched % เบี่ยงจาก baseline เกิน {DRIFT_TOLERANCE_PP}pp "
            f"-> upstream data/script อาจเปลี่ยนไป ต้อง review นโยบาย drop-BENIGN-unmatched ใหม่ "
            f"ก่อนใช้ผลลัพธ์นี้ต่อ"
        )
    if diag["n_nonbenign_unmatched"] > 0:
        raise RuntimeError(
            f"check_drift: {split_name} มีแถว non-BENIGN ที่ไม่ match {diag['n_nonbenign_unmatched']} แถว "
            f"-> ขัดกับ finding เดิมที่ยืนยันว่า attack ทุก category match 100% ต้องตรวจสอบก่อน drop ต่อ"
        )


# ----------------------------------------------------------------------------
# STEP 3: join จริง (validate='1:1') -- ใช้ left join แล้ว drop แถว unmatched (BENIGN) แบบ explicit
# ----------------------------------------------------------------------------
def apply_join(split_df, behavioral_df, key_cols, split_name):
    left_k = normalize_key_dtypes(split_df, key_cols)
    right_k = normalize_key_dtypes(behavioral_df, key_cols)

    left = split_df.copy()
    left["_h"] = compute_row_hash(left_k, key_cols)
    # ตำแหน่งเดิมใน cicids2017_{split}.parquet ก่อน drop -- คือ space เดียวกับที่ row_index ของ
    # cicids2017_features_{split}.parquet (ฝั่ง Random Forest เดิม) ใช้อยู่ (feature_extractor.py
    # ไม่ drop/reorder แถวเลย) เก็บไว้เป็น orig_row_index เพื่อให้ Fusion Layer ในอนาคต join คะแนน
    # IF/AE (behavioral, ประชากรลดแล้ว) กับคะแนน RF (ประชากรเต็ม) ด้วย key นี้แทนการสมมติว่า
    # row_index ตรงกันทางตำแหน่ง (position) เฉยๆ ซึ่งจะพังทันทีเพราะแถวถูก drop ไปแล้ว
    left["orig_row_index"] = np.arange(len(left))

    right = behavioral_df[key_cols + BEHAVIORAL_FEATURE_COLS].copy()
    right["_h"] = compute_row_hash(right_k, key_cols)
    right = right[["_h"] + BEHAVIORAL_FEATURE_COLS]

    merged = left.merge(right, how="left", on="_h", validate="1:1", indicator=True)

    unmatched_mask = merged["_merge"] != "both"
    n_unmatched = int(unmatched_mask.sum())
    n_nonbenign_unmatched = int((unmatched_mask & (merged["Label"] != "BENIGN")).sum())
    if n_nonbenign_unmatched > 0:
        raise RuntimeError(
            f"apply_join: {split_name} มีแถว non-BENIGN ที่ไม่ match {n_nonbenign_unmatched} แถว "
            f"-> หยุดก่อน drop เพราะขัดกับนโยบายที่ตกลงไว้ (drop เฉพาะ BENIGN unmatched เท่านั้น)"
        )

    n_before = len(merged)
    merged = merged.loc[~unmatched_mask].drop(columns=["_h", "_merge"]).reset_index(drop=True)
    n_after = len(merged)
    log(
        f"  {split_name}: join แล้ว drop แถว BENIGN ที่ไม่ match {n_unmatched:,} แถว "
        f"({n_before:,} -> {n_after:,} แถว)"
    )
    return merged, n_unmatched


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    section("PHASE 1: โหลดไฟล์ split เดิม + behavioral labeled")
    behavioral_df = safe_read_parquet(BEHAVIORAL_LABELED_PATH)
    log(f"behavioral_labeled shape = {behavioral_df.shape}")

    split_dfs = {}
    for split_name in SPLIT_NAMES:
        split_dfs[split_name] = safe_read_parquet(SPLIT_PATHS[split_name])
        log(f"cicids2017_{split_name}.parquet shape = {split_dfs[split_name].shape}")

    key_cols = list(split_dfs["train"].columns)
    log(f"\njoin key: {len(key_cols)} คอลัมน์ (คาดว่า 79)")
    assert len(key_cols) == 79, f"คาดว่า key columns 79 คอลัมน์ แต่ได้ {len(key_cols)}"
    missing_in_behavioral = [c for c in key_cols if c not in behavioral_df.columns]
    assert not missing_in_behavioral, f"behavioral_labeled ขาดคอลัมน์ key: {missing_in_behavioral}"
    missing_behavioral_feats = [c for c in BEHAVIORAL_FEATURE_COLS if c not in behavioral_df.columns]
    assert not missing_behavioral_feats, f"behavioral_labeled ขาดคอลัมน์ behavioral feature: {missing_behavioral_feats}"

    os.makedirs(BEHAVIORAL_DATA_DIR, exist_ok=True)

    for split_name in SPLIT_NAMES:
        section(f"PHASE 2 ({split_name}): diagnostic outer join -- ตรวจก่อน join จริงเสมอ")
        split_df = split_dfs[split_name]
        diag = diagnostic_outer_join(split_df, behavioral_df, key_cols, split_name)
        log(f"  total={diag['n_total']:,}  matched={diag['n_matched']:,}  unmatched={diag['n_left_only']:,} "
            f"({diag['pct_total_unmatched']:.2f}%)")
        log(f"  BENIGN: total={diag['benign_total']:,}  unmatched={diag['n_benign_unmatched']:,} "
            f"({diag['pct_benign_unmatched']:.2f}%)  |  non-BENIGN unmatched={diag['n_nonbenign_unmatched']}")
        check_drift(diag, split_name)

        section(f"PHASE 3 ({split_name}): join จริง (validate='1:1') + drop แถว BENIGN ที่ไม่ match")
        joined_df, n_dropped = apply_join(split_df, behavioral_df, key_cols, split_name)

        output_path = OUTPUT_PATHS[split_name]
        joined_df.to_parquet(output_path, index=False)
        log(f"  เซฟไว้ที่ {output_path} ({joined_df.shape[0]:,} แถว, {joined_df.shape[1]} คอลัมน์)")

    section("สรุปผลลัพธ์")
    log("Phase 1 ของ Option A (row-hash join validation) เสร็จสมบูรณ์:")
    log("  - ทุก attack category match 100% ระหว่างสอง distribution (ไม่มีแถว attack ถูก drop เลย)")
    log("  - แถว BENIGN ที่ไม่ match (~23.6%) ถูก drop ออกตามนโยบายที่ตกลงไว้ ดูตัวเลขจริงต่อ split ด้านบน")
    log("  - ผลลัพธ์: data/processed/behavioral/cicids2017_behavioral_{train,val,test}.parquet")
    log("  - ยังไม่ scale/SMOTE/impute -- นั่นคือ Phase 2 (แผนแยกต่างหาก ยังไม่เริ่ม)")
    log("  - ข้อจำกัดที่ต้อง carry ไป Phase 2: Autoencoder BENIGN training population ลดลง ~23.6%;")
    log("    04_fusion_layer.ipynb row_index positional-alignment assumption ต้อง re-derive สำหรับ")
    log("    การเทียบ fusion ในอนาคตกับ Random Forest (ซึ่งยังใช้ population เดิมไม่ลด)")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
