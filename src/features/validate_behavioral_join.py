
import os
import sys
import numpy as np
import pandas as pd

from src.features import behavioral_feature_extractor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED_DATA_DIR = "./data/processed"
BEHAVIORAL_DATA_DIR = "./data/processed/behavioral"
SPLIT_NAMES = ["train", "val", "test"]

SPLIT_PATHS = {s: os.path.join(PROCESSED_DATA_DIR, f"cicids2017_{s}.parquet") for s in SPLIT_NAMES}
BEHAVIORAL_LABELED_PATH = os.path.join(BEHAVIORAL_DATA_DIR, "cicids2017_behavioral_labeled.parquet")
OUTPUT_PATHS = {s: os.path.join(BEHAVIORAL_DATA_DIR, f"cicids2017_behavioral_{s}.parquet") for s in SPLIT_NAMES}
REPORT_PATH = "./reports/behavioral_join_validation_report.txt"

BEHAVIORAL_CONTEXT_COLS = [
    "Source IP",
    "Timestamp",
    "Destination Port",
    "Protocol",
    "RST Flag Count",
    "SYN Flag Count",
    "ACK Flag Count",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "day_group",
]

BASELINE_TOTAL_UNMATCHED_PCT = {"train": 19.59, "val": 19.65, "test": 19.64}
BASELINE_BENIGN_UNMATCHED_PCT = {"train": 23.57, "val": 23.64, "test": 23.63}
DRIFT_TOLERANCE_PP = 2.0

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


def apply_join(split_df, behavioral_df, key_cols, split_name):
    left_k = normalize_key_dtypes(split_df, key_cols)
    right_k = normalize_key_dtypes(behavioral_df, key_cols)
    left = split_df.copy()
    left["_h"] = compute_row_hash(left_k, key_cols)
    left["orig_row_index"] = np.arange(len(left))
    left["_split_row_order"] = np.arange(len(left))
    # 8 of BEHAVIORAL_CONTEXT_COLS are also join keys, so pulling them from the right
    # makes pandas suffix both sides _x/_y and add_behavioral_features then cannot find
    # e.g. "RST Flag Count". They are join keys, so the left copy is identical by
    # construction: take only the columns the split genuinely lacks.
    added_context_cols = [c for c in BEHAVIORAL_CONTEXT_COLS if c not in key_cols]
    right_context = behavioral_df[added_context_cols].copy()
    right = behavioral_df[key_cols].copy()
    right["_h"] = compute_row_hash(right_k, key_cols)
    right = pd.concat([right["_h"], right_context], axis=1)
    assert_hash_unique(left["_h"], "split", split_name)
    assert_hash_unique(right["_h"], "behavioral", split_name)
    merged = left.merge(right, how="left", on="_h", validate="1:1", indicator=True)
    unmatched_mask = merged["_merge"] != "both"
    n_unmatched = int(unmatched_mask.sum())
    n_nonbenign_unmatched = int((unmatched_mask & (merged["Label"] != "BENIGN")).sum())
    if n_nonbenign_unmatched > 0:
        raise RuntimeError(
            f"apply_join: {split_name} มีแถว non-BENIGN ที่ไม่ match {n_nonbenign_unmatched} แถว "
            f"-> หยุดก่อน drop เพราะขัดกับนโยบายที่ตกลงไว้ (drop เฉพาะ BENIGN unmatched เท่านั้น)"
        )

    matched = merged.loc[~unmatched_mask].drop(columns=["_h", "_merge"]).copy()
    n_after = len(matched)
    day_frames = []
    for _, day_df in matched.groupby("day_group", sort=False):
        ordered_day = day_df.sort_values(["Source IP", "Timestamp"], kind="stable").reset_index(drop=True)
        day_frames.append(behavioral_feature_extractor.add_behavioral_features(ordered_day))
    if day_frames:
        matched = pd.concat(day_frames, ignore_index=True)
        matched = matched.sort_values("_split_row_order", kind="stable").reset_index(drop=True)
    matched = matched.drop(columns=["_split_row_order", *added_context_cols])
    log(
        f"  {split_name}: join แล้ว drop แถว BENIGN ที่ไม่ match {n_unmatched:,} แถว "
        f"({len(split_df):,} -> {n_after:,} แถว), rolling windows rebuilt within this split"
    )
    return matched, n_unmatched


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
    missing_behavioral_context = [c for c in BEHAVIORAL_CONTEXT_COLS if c not in behavioral_df.columns]
    assert not missing_behavioral_context, f"behavioral_labeled ขาดคอลัมน์สำหรับคำนวณ split-aware features: {missing_behavioral_context}"

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
    log("  - rolling behavioral features คำนวณใหม่ภายในแต่ละ split เพื่อไม่ใช้ event จาก split อื่น")
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
