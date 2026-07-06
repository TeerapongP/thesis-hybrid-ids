

import os
import sys
import numpy as np
import pandas as pd

# คอนโซลของ Windows มักใช้ cp1252 เป็นค่า default ซึ่งพิมพ์ข้อความภาษาไทยไม่ได้
# บังคับให้ stdout เป็น utf-8 เพื่อไม่ให้ print() ล้มกลางทาง
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONFIG: แก้ path ตรงนี้ให้ตรงกับตำแหน่งไฟล์จริงบนเครื่อง — สคริปต์นี้อยู่ใน src/ แต่ยังต้อง
# รันจาก repo root เสมอ (เช่น `python src/eda_sanity_check.py`) path ด้านล่างจึงอ้างอิงจาก
# repo root ไม่ใช่จากตำแหน่งไฟล์ .py เอง
# ----------------------------------------------------------------------------
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

UNSW_FILES = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
UNSW_FEATURES_FILE = "NUSW-NB15_features.csv"   # เป็น .csv (encoding cp1252 ไม่ใช่ utf-8)
UNSW_GT_FILE = "NUSW-NB15_GT.csv"               # ไม่ได้ใช้ใน sanity check นี้ แต่เก็บ path ไว้อ้างอิง

REPORT_PATH = "./reports/eda_report.txt"

# ----------------------------------------------------------------------------
# ตัวช่วยเก็บ output: print ออกจอ และเก็บบรรทัดไว้เขียนไฟล์ report พร้อมกัน
# ----------------------------------------------------------------------------
_REPORT_LINES = []


def log(msg=""):
    """print ข้อความ และเก็บไว้เขียนลง eda_report.txt ทีหลัง"""
    msg = str(msg)
    print(msg)
    _REPORT_LINES.append(msg)


def section(title):
    """พิมพ์หัวข้อคั่นแต่ละ section ให้อ่านง่าย"""
    log("=" * 70)
    log(title)
    log("=" * 70)


def safe_read_csv(path, **kwargs):
    """wrap pd.read_csv ให้ error message ชัดเจนว่าไฟล์ไหนหาไม่เจอ"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์: '{path}' กรุณาตรวจสอบว่าตั้งค่า DATA_DIR ถูกต้อง "
            f"และไฟล์ถูกวางไว้ในตำแหน่งนั้นจริง"
        )
    return pd.read_csv(path, **kwargs)


# ----------------------------------------------------------------------------
# SECTION 1: โหลดข้อมูล
# ----------------------------------------------------------------------------
def load_cicids():
    """โหลดไฟล์ CICIDS-2017 ทั้ง 8 ไฟล์ (มี header อยู่แล้ว) แล้ว concat เป็นก้อนเดียว"""
    frames = []
    for fname in CICIDS_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = safe_read_csv(path, low_memory=False, encoding="utf-8-sig")
        log(f"  โหลด {fname:55s} shape={df.shape}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_unsw():
    """
    โหลด UNSW-NB15:
      1) เปิดไฟล์ features เพื่อดึงชื่อคอลัมน์ (คอลัมน์ 'Name' ในไฟล์ features)
         ไฟล์นี้เข้ารหัสแบบ cp1252 (มีอักขระ smart-quote ที่ไม่ใช่ UTF-8)
      2) ใช้ชื่อคอลัมน์ที่ได้ assign ให้กับข้อมูลดิบ 4 ไฟล์ (ไม่มี header ในตัวข้อมูล)
      3) concat รวมเป็นก้อนเดียว
    """
    features_path = os.path.join(DATA_DIR, UNSW_FEATURES_FILE)
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ features: '{features_path}'")

    features_df = pd.read_csv(features_path, encoding="cp1252")
    log(f"  ไฟล์ features: {UNSW_FEATURES_FILE}  columns={features_df.columns.tolist()}")
    col_names = features_df["Name"].str.strip().tolist()
    log(f"  จำนวนชื่อคอลัมน์ที่ดึงมาได้: {len(col_names)}")
    log(f"  รายชื่อคอลัมน์: {col_names}")

    frames = []
    for fname in UNSW_FILES:
        path = os.path.join(DATA_DIR, fname)
        # UNSW-NB15_1.csv มี BOM (utf-8-sig) ส่วนไฟล์อื่นเป็น ASCII ธรรมดา
        # ใช้ utf-8-sig กับทุกไฟล์ได้อย่างปลอดภัย (ไม่กระทบไฟล์ที่ไม่มี BOM)
        df = safe_read_csv(
            path, header=None, names=col_names, low_memory=False, encoding="utf-8-sig"
        )
        log(f"  โหลด {fname:25s} shape={df.shape}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ----------------------------------------------------------------------------
# SECTION 2: เช็คโครงสร้างข้อมูล
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# SECTION 3: เช็คคุณภาพข้อมูล (missing / inf / duplicate / negative)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# SECTION 4: Class distribution
# ----------------------------------------------------------------------------
def check_class_distribution_cicids(df):
    log("--- CICIDS-2017: distribution ของคอลัมน์ Label ---")
    log(df["Label"].value_counts().to_string())
    log("")
    log("--- CICIDS-2017: สัดส่วน (%) ---")
    log(df["Label"].value_counts(normalize=True).mul(100).round(4).to_string())


def check_class_distribution_unsw(df):
    log("--- UNSW-NB15: distribution ของคอลัมน์ Label (binary) ---")
    log(df["Label"].value_counts().to_string())
    log("")
    log("--- UNSW-NB15: สัดส่วน (%) Label ---")
    log(df["Label"].value_counts(normalize=True).mul(100).round(4).to_string())
    log("")
    log("--- UNSW-NB15: distribution ของคอลัมน์ attack_cat (multi-class) ---")
    log(df["attack_cat"].value_counts(dropna=False).to_string())
    log("")
    log("--- UNSW-NB15: สัดส่วน (%) attack_cat ---")
    log(df["attack_cat"].value_counts(normalize=True, dropna=False).mul(100).round(4).to_string())

    log("")
    log("--- UNSW-NB15: ตรวจสอบ consistency ระหว่าง attack_cat กับ Label ---")
    # attack_cat มีค่า (ไม่ใช่ NaN/blank) แปลว่าเป็น traffic โจมตี -> ควรมี Label == 1
    attack_cat_clean = df["attack_cat"].astype("string").str.strip()
    is_attack_row = attack_cat_clean.notna() & (attack_cat_clean != "")
    mismatch_should_be_1 = df.loc[is_attack_row & (df["Label"] != 1)]
    mismatch_should_be_0 = df.loc[~is_attack_row & (df["Label"] != 0)]
    log(f"  แถวที่ attack_cat ระบุว่าเป็นการโจมตี แต่ Label != 1 : {len(mismatch_should_be_1)} แถว")
    log(f"  แถวที่ attack_cat ว่าง/NaN (ควรเป็น normal) แต่ Label != 0 : {len(mismatch_should_be_0)} แถว")
    if len(mismatch_should_be_1) > 0:
        log(f"  ตัวอย่าง attack_cat ที่ mismatch: {mismatch_should_be_1['attack_cat'].unique()[:10].tolist()}")


# ----------------------------------------------------------------------------
# SECTION 5: Constant / low-variance columns
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# SECTION 6: Timestamp
# ----------------------------------------------------------------------------
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
    log("--- UNSW-NB15: ตรวจสอบคอลัมน์เกี่ยวกับเวลา (Stime / Ltime) ---")
    for c in ["Stime", "Ltime"]:
        if c not in df.columns:
            log(f"  ไม่พบคอลัมน์ {c!r} ในข้อมูล")
            continue
        log(f"  คอลัมน์ {c!r} (unix epoch seconds)")
        log(f"    ตัวอย่างค่าดิบ: {df[c].head(5).tolist()}")
        log(f"    min = {df[c].min()}  max = {df[c].max()}")
        parsed_min = pd.to_datetime(df[c].min(), unit="s")
        parsed_max = pd.to_datetime(df[c].max(), unit="s")
        log(f"    ช่วงเวลาจริง: {parsed_min} ถึง {parsed_max}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
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
    log(">> UNSW-NB15")
    try:
        unsw_df = load_unsw()
    except FileNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    log(f"UNSW-NB15 รวมทั้งหมด shape = {unsw_df.shape}")

    section("SECTION 2: เช็คโครงสร้างข้อมูล")
    cicids_df = check_structure(cicids_df, "CICIDS-2017")
    unsw_df = check_structure(unsw_df, "UNSW-NB15")

    section("SECTION 3: เช็คคุณภาพข้อมูล - CICIDS-2017")
    check_missing(cicids_df, "CICIDS-2017")
    check_inf(cicids_df, "CICIDS-2017")
    check_duplicates(cicids_df, "CICIDS-2017")
    check_negative_values(cicids_df, "CICIDS-2017")

    section("SECTION 3: เช็คคุณภาพข้อมูล - UNSW-NB15")
    check_missing(unsw_df, "UNSW-NB15")
    check_inf(unsw_df, "UNSW-NB15")
    check_duplicates(unsw_df, "UNSW-NB15")
    check_negative_values(unsw_df, "UNSW-NB15")

    section("SECTION 4: Class Distribution")
    check_class_distribution_cicids(cicids_df)
    log("")
    check_class_distribution_unsw(unsw_df)

    section("SECTION 5: Constant / Low-variance Columns")
    check_constant_columns(cicids_df, "CICIDS-2017")
    log("")
    check_constant_columns(unsw_df, "UNSW-NB15")

    section("SECTION 6: Timestamp")
    check_timestamp_cicids(cicids_df)
    log("")
    check_timestamp_unsw(unsw_df)

    section("SECTION 7: สรุปผลลัพธ์")
    log(f"CICIDS-2017 : {cicids_df.shape[0]} แถว, {cicids_df.shape[1]} คอลัมน์, "
        f"duplicate rows = {cicids_df.duplicated().sum()}, "
        f"missing cells = {int(cicids_df.isna().sum().sum())}")
    log(f"UNSW-NB15   : {unsw_df.shape[0]} แถว, {unsw_df.shape[1]} คอลัมน์, "
        f"duplicate rows = {unsw_df.duplicated().sum()}, "
        f"missing cells = {int(unsw_df.isna().sum().sum())}")
    log("รายละเอียดทั้งหมดตามหัวข้อ 1-6 ด้านบน ถูกบันทึกไว้ใน eda_report.txt เรียบร้อยแล้ว")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
