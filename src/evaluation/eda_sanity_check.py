"""
EDA sanity check สำหรับสองชุดข้อมูลที่ pipeline ใช้งานจริง:

1. CICIDS-2017 — การแจกจ่ายแบบ MachineLearningCVE (8 ไฟล์ `*_ISCX.csv`,
   78 flow features + Label, ไม่มี Timestamp/IP/Port)
2. CIC-UNSW-NB15 — `data/raw/cic-unsw-nb15/Data.csv` + `Label.csv`
   (positional join, 76 CICFlowMeter features, label เป็นรหัส 0-9)
   ชุดนี้คือแหล่งข้อมูลของ "unswnb15 slot" ใน
   `src/ingestion/preprocess_pipeline.py` ตั้งแต่การย้ายฐานข้อมูลปี 2026-07-02

ของเดิมสคริปต์นี้อ่าน UNSW-NB15 รูปแบบดิบ 4 ไฟล์ (`UNSW-NB15_1..4.csv`, headerless,
ต้องประกอบชื่อคอลัมน์จาก `NUSW-NB15_features.csv` ที่เป็น cp1252) ซึ่ง **ไม่ได้ป้อน
อะไรใน pipeline อีกแล้ว** รายงาน UNSW ที่ออกมาจึงเป็นสถิติของไฟล์ที่ไม่มีใครใช้
ตอนนี้สคริปต์เรียก `load_unsw()` ของ `preprocess_pipeline` โดยตรง (import ไม่ copy)
เพื่อให้ EDA และ pipeline อ่านไฟล์เดียวกันเสมอและไม่มีโอกาส drift ออกจากกัน
"""

import os
import sys
import numpy as np
import pandas as pd

from src.ingestion import preprocess_pipeline as pp
from src.ingestion.preprocess_pipeline import (
    CIC_UNSW_DATA_FILE,
    CIC_UNSW_DIR,
    CIC_UNSW_LABEL_CODE_MAP,
    CIC_UNSW_LABEL_FILE,
    INIT_WIN_BYTES_COLS,
    load_unsw,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = "./data/raw"

CICIDS_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

UNSW_DATASET_NAME = "CIC-UNSW-NB15"

REPORT_PATH = "./reports/eda_report.txt"

_REPORT_LINES = []


def log(msg=""):
    msg = str(msg)
    print(msg)
    _REPORT_LINES.append(msg)


# `preprocess_pipeline.load_unsw()` เขียน log ผ่าน `log()` ของโมดูลตัวเอง ซึ่งจะไปลง
# _REPORT_LINES ของ preprocess_report.txt แทน เปลี่ยน hook ให้ชี้มาที่ log() ตัวนี้
# เพื่อให้บรรทัดโหลดไฟล์ปรากฏใน eda_report.txt (ไม่ใช่การ copy ตัวโหลดข้อมูลมาเขียนใหม่)
pp.log = log


def section(title):
    log("=" * 70)
    log(title)
    log("=" * 70)


def safe_read_csv(path, **kwargs):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' กรุณาตรวจสอบว่าตั้งค่า DATA_DIR ถูกต้อง "
            f"และไฟล์ถูกวางไว้ในตำแหน่งนั้นจริง"
        )
    return pd.read_csv(path, **kwargs)


def load_cicids():
    frames = []
    for fname in CICIDS_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = safe_read_csv(path, low_memory=False, encoding="utf-8-sig")
        log(f"  โหลด {fname:55s} shape={df.shape}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


#
# โหมดโหลด UNSW ของสคริปต์นี้ถูกลบทิ้งทั้งก้อน แทนที่ด้วย `load_unsw()` ที่ import มาจาก
# `preprocess_pipeline` ตรง ๆ — pipeline อ่าน Data.csv + Label.csv (positional join),
# rename คอลัมน์ CICFlowMeter ที่ชื่อต่างกันผ่าน CIC_UNSW_COLUMN_RENAME และ derive
# attack_cat + Label (binary) จากรหัส 0-9 ตาม CIC_UNSW_LABEL_CODE_MAP
#
# เช็คที่ถูกลบไปพร้อมกัน เพราะไม่มีอยู่ในรูปแบบข้อมูลใหม่:
#   * การอ่าน NUSW-NB15_features.csv ด้วย cp1252 เพื่อประกอบชื่อคอลัมน์ — Data.csv
#     มี header อยู่แล้ว ไม่ต้องประกอบชื่อ ไม่มีไฟล์ cp1252 ให้อ่าน
#   * การอ่านทุกไฟล์เป็น utf-8-sig เพื่อกัน BOM ของ UNSW-NB15_1.csv — คงไว้ใน
#     preprocess_pipeline.safe_read_csv อยู่แล้ว แต่ไม่ใช่ประเด็นของชุดใหม่


def read_unsw_label_codes():
    """อ่านคอลัมน์รหัส 0-9 ดิบจาก Label.csv (ใช้รายงาน distribution ของรหัสเท่านั้น)."""
    label_path = os.path.join(CIC_UNSW_DIR, CIC_UNSW_LABEL_FILE)
    label_df = safe_read_csv(label_path, low_memory=False, encoding="utf-8-sig")
    return label_df["Label"]


def check_structure(df, name):
    log(f"--- {name}: columns ---")
    log(df.columns.tolist())
    log(f"--- {name}: dtypes ---")
    log(df.dtypes.to_string())

    if name == "CICIDS-2017":
        log(f"--- {name}: ตรวจสอบ whitespace นำหน้า/ตามหลังชื่อคอลัมน์ (repr) ---")
        stripped_needed = []
        for col in df.columns:
            if col != col.strip():
                stripped_needed.append(col)
            log(f"  {col!r}")
        if stripped_needed:
            log(f"  พบคอลัมน์ที่มี whitespace เกิน {len(stripped_needed)} คอลัมน์ -> ทำการ .str.strip() ให้อัตโนมัติ")
            df.columns = df.columns.str.strip()
            log("  ผลลัพธ์หลัง strip:")
            log(df.columns.tolist())
        else:
            log("  ไม่พบ whitespace เกินในชื่อคอลัมน์")
    return df


NEGATIVE_CHECK_KEYWORDS = [
    "duration", "dur", "byte", "packet", "pkt", "length", "len",
    "count", "loss", "win", "ttl", "jit", "size", "meansz",
]


def check_missing(df, name):
    log(f"--- {name}: จำนวน missing values (เฉพาะคอลัมน์ที่ NaN > 0) ---")
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    if na_counts.empty:
        log("  ไม่พบ missing values ในทั้งชุดข้อมูล")
    else:
        log(na_counts.to_string())


def check_inf(df, name):
    log(f"--- {name}: จำนวน inf / -inf ในคอลัมน์ตัวเลข ---")
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = numeric_df.apply(lambda s: np.isinf(s).sum())
    inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)
    if inf_counts.empty:
        log("  ไม่พบค่า inf/-inf ในคอลัมน์ตัวเลข")
    else:
        log(inf_counts.to_string())


def check_duplicates(df, name):
    log(f"--- {name}: จำนวนแถวซ้ำ (duplicate rows) ---")
    dup_count = df.duplicated().sum()
    pct = dup_count / len(df) * 100
    log(f"  duplicate rows = {dup_count} ({pct:.4f}% ของทั้งหมด {len(df)} แถว)")


def check_negative_values(df, name):
    log(f"--- {name}: ค่าติดลบในคอลัมน์ที่ไม่ควรติดลบ (duration/bytes/packet/count ฯลฯ) ---")
    numeric_df = df.select_dtypes(include=[np.number])
    candidate_cols = [
        c for c in numeric_df.columns
        if any(kw in c.lower() for kw in NEGATIVE_CHECK_KEYWORDS)
    ]
    found_any = False
    for col in candidate_cols:
        neg_count = (numeric_df[col] < 0).sum()
        if neg_count > 0:
            found_any = True
            log(f"  {col:45s} : {neg_count} ค่าติดลบ")
    if not found_any:
        log(f"  ตรวจสอบ {len(candidate_cols)} คอลัมน์ที่เข้าเงื่อนไข ไม่พบค่าติดลบ")


def check_class_distribution_cicids(df):
    log("--- CICIDS-2017: distribution ของคอลัมน์ Label ---")
    log(df["Label"].value_counts().to_string())
    log("")
    log("--- CICIDS-2017: สัดส่วน (%) ---")
    log(df["Label"].value_counts(normalize=True).mul(100).round(4).to_string())


def check_class_distribution_unsw(df, label_codes):
    log(f"--- {UNSW_DATASET_NAME}: distribution ของรหัส label ดิบใน Label.csv (0-9) ---")
    code_counts = label_codes.value_counts().sort_index()
    total_codes = int(code_counts.sum())
    for code, n in code_counts.items():
        name = CIC_UNSW_LABEL_CODE_MAP.get(code, "<ไม่มีใน mapping>")
        pct = 100 * n / total_codes if total_codes else 0.0
        log(f"  {code} = {name:16s} n={n:>8d}  ({pct:7.4f}%)")
    log(f"  รวม {total_codes} แถว, พบรหัสทั้งหมด {len(code_counts)} ค่า")
    log("")
    log(f"--- {UNSW_DATASET_NAME}: distribution ของคอลัมน์ Label (binary, derive จากรหัสเดียวกัน) ---")
    log(df["Label"].value_counts().to_string())
    log("")
    log(f"--- {UNSW_DATASET_NAME}: สัดส่วน (%) Label ---")
    log(df["Label"].value_counts(normalize=True).mul(100).round(4).to_string())
    log("")
    log(f"--- {UNSW_DATASET_NAME}: distribution ของคอลัมน์ attack_cat (multi-class, NaN = Benign) ---")
    log(df["attack_cat"].value_counts(dropna=False).to_string())
    log("")
    log(f"--- {UNSW_DATASET_NAME}: สัดส่วน (%) attack_cat ---")
    log(df["attack_cat"].value_counts(normalize=True, dropna=False).mul(100).round(4).to_string())

    # ลบเช็ค consistency ระหว่าง attack_cat กับ Label ออก: ในรูปแบบ UNSW-NB15 ดิบเดิม
    # สองคอลัมน์นี้มาจากสองแหล่งอิสระ จึงต้อง cross-check ว่าตรงกัน แต่ใน CIC-UNSW-NB15
    # ทั้งคู่ derive จากรหัส Label.csv ตัวเดียวกันใน load_unsw() จึงตรงกันโดยโครงสร้าง
    # (by construction) การเช็คซ้ำที่นี่ไม่ได้วัดคุณภาพข้อมูลต้นทางอะไรเลย — จะวัดก็แค่
    # บั๊กของโค้ดเราเอง ซึ่ง preprocess_pipeline.clean_unsw() assert ไว้ให้แล้ว


def check_init_win_sentinel_unsw(df):
    """CICIDS-2017 ใช้ -1 เป็น sentinel 'not measured' ใน Init_Win_bytes_* — ยืนยันว่า
    คอลัมน์ที่ rename มาแล้วของ CIC-UNSW-NB15 ไม่มี sentinel นี้ (จึงไม่ต้องจัดการพิเศษ)."""
    log(f"--- {UNSW_DATASET_NAME}: เช็ค sentinel -1 ในคอลัมน์ Init-Win-Bytes ที่ rename แล้ว ---")
    for c in INIT_WIN_BYTES_COLS:
        if c not in df.columns:
            log(f"  [WARNING] ไม่พบคอลัมน์ {c!r} หลัง rename — ตรวจ CIC_UNSW_COLUMN_RENAME")
            continue
        n_sentinel = int((df[c] == -1).sum())
        log(f"  {c:28s} จำนวน -1 = {n_sentinel:>8d}  min = {df[c].min()}  max = {df[c].max()}")


def check_feature_column_count_unsw(df):
    log(f"--- {UNSW_DATASET_NAME}: จำนวนคอลัมน์ feature (ไม่รวม label ที่ derive เพิ่ม) ---")
    derived_cols = ["Label", "attack_cat"]
    feature_cols = [c for c in df.columns if c not in derived_cols]
    numeric_feature_cols = (
        df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    non_numeric = [c for c in feature_cols if c not in numeric_feature_cols]
    log(f"  คอลัมน์ทั้งหมดหลังโหลด = {df.shape[1]} (feature {len(feature_cols)} + derived {derived_cols})")
    log(f"  feature ที่เป็นตัวเลข = {len(numeric_feature_cols)}")
    log(f"  feature ที่ไม่ใช่ตัวเลข = {len(non_numeric)} {non_numeric if non_numeric else ''}")


def check_constant_columns(df, name):
    log(f"--- {name}: คอลัมน์ที่มีค่าเดียวกันทั้งหมด (nunique == 1) ---")
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols:
        log(f"  พบ {len(constant_cols)} คอลัมน์ (ตัวเลือกให้พิจารณา drop ในขั้น preprocessing):")
        for c in constant_cols:
            log(f"    {c!r} -> ค่าเดียว = {df[c].iloc[0]!r}")
    else:
        log("  ไม่พบคอลัมน์ที่มีค่าคงที่ตลอดทั้งชุดข้อมูล")


def check_timestamp_cicids(df):
    log("--- CICIDS-2017: ตรวจสอบคอลัมน์เกี่ยวกับเวลา ---")
    time_cols = [c for c in df.columns if ("time" in c.lower() or "stamp" in c.lower())]
    if not time_cols:
        log(
            "  ไม่พบคอลัมน์ Timestamp ในชุดไฟล์นี้ "
            "(ไฟล์ที่ใช้เป็นชุด MachineLearningCVE ซึ่งตัดคอลัมน์ Flow ID / Source IP / "
            "Source Port / Destination IP / Timestamp ออกจากการแจกจ่ายสาธารณะแล้ว "
            "เหลือเพียง 78 flow-based features + Label). "
            "หากต้องการ timestamp จริงสำหรับทำ simulated replay feed "
            "จะต้องใช้ไฟล์ต้นทาง GeneratedLabelledFlows (มี Timestamp column) แทน"
        )
    else:
        for c in time_cols:
            log(f"  พบคอลัมน์ที่เกี่ยวกับเวลา: {c!r}")
            log(f"  ตัวอย่างค่า: {df[c].head(5).tolist()}")


def check_timestamp_unsw(df):
    # เช็ค Stime/Ltime (unix epoch) ของรูปแบบ UNSW-NB15 ดิบเดิมถูกลบออก: CIC-UNSW-NB15
    # Data.csv ไม่มีสองคอลัมน์นั้น และไม่มีคอลัมน์เวลาใด ๆ เลย การปล่อยให้เช็คเดิมรันต่อ
    # จะได้แค่ "ไม่พบคอลัมน์" สองบรรทัดโดยไม่อธิบายว่าทำไม จึงแทนด้วยการเช็คแบบ generic
    # เหมือนฝั่ง CICIDS-2017 พร้อมระบุว่าถ้าต้องการเวลา/IP จริงต้องไปใช้ไฟล์ไหน
    log(f"--- {UNSW_DATASET_NAME}: ตรวจสอบคอลัมน์เกี่ยวกับเวลา ---")
    time_cols = [c for c in df.columns if ("time" in c.lower() or "stamp" in c.lower())]
    if not time_cols:
        log(
            "  ไม่พบคอลัมน์ Timestamp ในชุดนี้ — Data.csv มีแต่ CICFlowMeter features "
            "ล้วน ๆ (ตัด 7 identifier columns ออก: Flow ID / Src IP / Src Port / "
            "Dst IP / Dst Port / Protocol / Timestamp) "
            "หากต้องการ timestamp หรือ IP จริงต้องใช้ "
            "data/raw/cic-unsw-nb15/CICFlowMeter_out.csv ซึ่งเป็น population "
            "แบบไม่กรอง (3,540,241 แถว) ไม่ใช่คู่ Data.csv/Label.csv ที่ pipeline ใช้"
        )
    else:
        for c in time_cols:
            log(f"  พบคอลัมน์ที่เกี่ยวกับเวลา: {c!r}")
            log(f"  ตัวอย่างค่า: {df[c].head(5).tolist()}")


def main():
    section("SECTION 1: โหลดข้อมูล")
    log(">> CICIDS-2017")
    try:
        cicids_df = load_cicids()
    except FileNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    log(f"CICIDS-2017 รวมทั้งหมด shape = {cicids_df.shape}")
    log("")
    log(f">> {UNSW_DATASET_NAME} (แหล่งข้อมูลจริงของ unswnb15 slot: "
        f"{CIC_UNSW_DATA_FILE} + {CIC_UNSW_LABEL_FILE}, positional join)")
    log(f"   โหลดผ่าน preprocess_pipeline.load_unsw() ตัวเดียวกับที่ pipeline ใช้ "
        f"(import ไม่ reimplement)")
    try:
        unsw_df = load_unsw()
        unsw_label_codes = read_unsw_label_codes()
    except (FileNotFoundError, ValueError) as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    log(f"{UNSW_DATASET_NAME} รวมทั้งหมด shape = {unsw_df.shape} (ยังไม่ dedup)")

    section("SECTION 2: เช็คโครงสร้างข้อมูล")
    cicids_df = check_structure(cicids_df, "CICIDS-2017")
    unsw_df = check_structure(unsw_df, UNSW_DATASET_NAME)
    check_feature_column_count_unsw(unsw_df)

    section("SECTION 3: เช็คคุณภาพข้อมูล - CICIDS-2017")
    check_missing(cicids_df, "CICIDS-2017")
    check_inf(cicids_df, "CICIDS-2017")
    check_duplicates(cicids_df, "CICIDS-2017")
    check_negative_values(cicids_df, "CICIDS-2017")

    section(f"SECTION 3: เช็คคุณภาพข้อมูล - {UNSW_DATASET_NAME}")
    check_missing(unsw_df, UNSW_DATASET_NAME)
    check_inf(unsw_df, UNSW_DATASET_NAME)
    check_duplicates(unsw_df, UNSW_DATASET_NAME)
    check_negative_values(unsw_df, UNSW_DATASET_NAME)
    check_init_win_sentinel_unsw(unsw_df)

    section("SECTION 4: Class Distribution")
    check_class_distribution_cicids(cicids_df)
    log("")
    check_class_distribution_unsw(unsw_df, unsw_label_codes)

    section("SECTION 5: Constant / Low-variance Columns")
    check_constant_columns(cicids_df, "CICIDS-2017")
    log("")
    check_constant_columns(unsw_df, UNSW_DATASET_NAME)

    section("SECTION 6: Timestamp")
    check_timestamp_cicids(cicids_df)
    log("")
    check_timestamp_unsw(unsw_df)

    section("SECTION 7: สรุปผลลัพธ์")
    log(f"CICIDS-2017 : {cicids_df.shape[0]} แถว, {cicids_df.shape[1]} คอลัมน์, "
        f"duplicate rows = {cicids_df.duplicated().sum()}, "
        f"missing cells = {int(cicids_df.isna().sum().sum())}")
    log(f"{UNSW_DATASET_NAME} : {unsw_df.shape[0]} แถว, {unsw_df.shape[1]} คอลัมน์, "
        f"duplicate rows = {unsw_df.duplicated().sum()}, "
        f"missing cells = {int(unsw_df.isna().sum().sum())} "
        f"(missing ส่วนใหญ่คือ attack_cat = NaN ของแถว Benign ตามการ derive)")
    log("รายละเอียดทั้งหมดตามหัวข้อ 1-6 ด้านบน ถูกบันทึกไว้ใน eda_report.txt เรียบร้อยแล้ว")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
