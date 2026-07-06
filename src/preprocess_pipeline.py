"""
preprocess_pipeline.py
========================
สคริปต์ preprocessing จริง (ต่างจาก eda_sanity_check.py ที่ทำแค่ตรวจสอบ ไม่แก้ข้อมูล)
สำหรับ CICIDS-2017 เท่านั้น แก้ปัญหา 4 ข้อที่เจอจาก eda_report.txt ก่อนนำข้อมูลไปใช้งานจริง:

  1) Label เพี้ยน (mojibake): "Web Attack � Brute Force" — ตรวจ raw bytes ในไฟล์ต้นฉบับแล้วพบว่า
     ไม่ใช่ปัญหาการอ่านผิด encoding ฝั่งเรา ตำแหน่งตัวขีดเป็น byte 0xEF 0xBF 0xBD ซึ่งคือ utf-8
     encoding ของอักขระ U+FFFD (REPLACEMENT CHARACTER) ที่ถูกฝังลงไปในไฟล์แล้วตั้งแต่ต้นทาง
     (en-dash ตัวจริงถูกทำลาย/แทนที่ไปแล้วก่อนไฟล์นี้จะถูกแจกจ่าย กู้คืนด้วยการเปลี่ยน encoding
     ไม่ได้ — ลอง cp1252 แล้วกลับยิ่งแย่ลง เพราะไปตีความ byte utf-8 เหล่านั้นผิดซ้ำอีกชั้น
     กลายเป็น "ï¿½" 3 ตัวอักษร) -> อ่านด้วย utf-8 (ค่า default/utf-8-sig เดิม) ตามปกติ ซึ่งจะ decode
     กลับมาเป็น U+FFFD ตัวเดียวถูกต้อง แล้ว normalize อักขระนี้ (รวมถึง en-dash/em-dash เผื่อกรณีอื่น)
     ใน Label ให้เป็น "-" เดียวกันหมด เพื่อให้ label อย่างน้อย map เข้ากับตาราง MITRE ได้ (ตัวขีดเดิม
     กู้คืนไม่ได้จริง แต่ทำให้ label สม่ำเสมอพอจะใช้งานได้)
  2) Init_Win_bytes_forward / Init_Win_bytes_backward: ค่า -1 เป็น sentinel ของ CICFlowMeter
     ที่แปลว่า "ไม่ได้วัดค่า/ไม่เกี่ยวข้อง" เฉพาะ 2 คอลัมน์นี้ ไม่ใช่ข้อมูลผิดพลาด -> แปลงเป็น NaN
     แยกต่างหากจาก negative-value cleaning ทั่วไป (ห้าม drop แถวเหล่านี้ จะเสียข้อมูลไป >1/3)
  3) ชุดไฟล์ MachineLearningCSV ที่ใช้ไม่มีคอลัมน์ Timestamp เลย (ถูกตัดออกจากการแจกจ่ายสาธารณะ)
     ส่งผลต่อขั้นตอน simulated replay feed ใน enrich.py -> เตือนชัดเจนใน log แทนที่จะเงียบไป
  4) รายงานจำนวนแถวต่อ attack category พร้อม flag ว่าใหญ่พอจะใช้เป็น LOAO
     (Leave-One-Attack-Out) zero-day holdout ได้อย่างมีนัยสำคัญทางสถิติหรือไม่

ผลลัพธ์: print รายงานออกจอ, เซฟรายงานเป็น ./reports/preprocess_report.txt, และเซฟ CICIDS-2017
ที่แก้แล้วเป็น ./data/processed/cicids2017_clean.parquet ให้ขั้นตอนถัดไป (label_builder.py /
feature_extractor.py) เอาไปใช้ต่อ — รันจาก repo root เสมอ เช่น `python src/preprocess_pipeline.py`
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONFIG — สคริปต์นี้อยู่ใน src/ แต่ยังต้องรันจาก repo root เสมอ (เช่น
# `python src/preprocess_pipeline.py`) path ด้านล่างจึงอ้างอิงจาก repo root
# RAW_DATA_DIR = ต้นฉบับ CSV ที่ไม่แตะเลย, PROCESSED_DATA_DIR = ผลลัพธ์ clean/split/LOAO
# ----------------------------------------------------------------------------
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"

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

# แหล่งข้อมูล "unswnb15" เปลี่ยนจาก UNSW-NB15 raw 4 ไฟล์เดิม (UNSW-NB15_1..4.csv, ยังอยู่ใน
# data/raw/ เผื่ออ้างอิง แต่ไม่ได้ใช้แล้ว) มาเป็น CIC-UNSW-NB15 (data/raw/cic-unsw-nb15/)
# — dataset คนละชุดที่ชื่อคล้ายกัน (UNSW-NB15 raw capture ที่ถูกรันผ่าน CICFlowMeter ใหม่
# อีกที ยืนยันจาก IP ที่ตรงกับ UNSW-NB15 testbed) เหตุผลที่สลับมาใช้: Data.csv มีคอลัมน์
# feature ชุดเดียวกับ CICFlowMeter ของ CICIDS-2017 (แค่ชื่อคอลัมน์ต่างกันบางส่วน ดู
# CIC_UNSW_COLUMN_RENAME) ทำให้เทียบข้าม dataset ได้ตรงกว่า raw UNSW-NB15 เดิม
CIC_UNSW_DIR = os.path.join(RAW_DATA_DIR, "cic-unsw-nb15")
CIC_UNSW_DATA_FILE = "Data.csv"
CIC_UNSW_LABEL_FILE = "Label.csv"

# จาก Readme.txt ของ CIC-UNSW-NB15 (ไม่มีคำอธิบายอื่นนอกจากตารางนี้)
CIC_UNSW_LABEL_CODE_MAP = {
    0: "Benign",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}

# Data.csv ใช้ CICFlowMeter เวอร์ชันใหม่กว่าที่สร้าง cicids2017 (ชื่อคอลัมน์เปลี่ยนไปบางส่วน
# แต่เป็น feature เดียวกันทางความหมาย) — ตรวจสอบแล้ว (2026-07-02): 57/76 คอลัมน์ตรงชื่อกับ
# cicids2017_clean.parquet พอดี, อีก 19 คอลัมน์ต้อง rename ตามตารางนี้ให้ตรงกับชื่อฝั่ง
# CICIDS-2017 เพื่อให้เทียบ/ใช้ข้าม dataset ได้ — คอลัมน์ที่ CICIDS-2017 มีแต่ Data.csv ไม่มีเลย
# (ไม่ใช่แค่ชื่อต่าง): 'Destination Port' (Data.csv ไม่มีคอลัมน์ port/IP ใดๆ เลย เพราะ
# CICFlowMeter_out.csv ตัดคอลัมน์ identifier ออกไปแล้วตอนสร้าง Data.csv), 'Label' (อยู่แยกใน
# Label.csv), 'Fwd Header Length.1' (คอลัมน์ซ้ำที่เป็น quirk ของไฟล์ cicids2017 เอง ไม่ใช่
# feature จริงที่ขาดหายไปจากฝั่งนี้)
CIC_UNSW_COLUMN_RENAME = {
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",  # cicids2017 สืบทอด typo ของ CICFlowMeter เก่า (CWR ที่ถูกต้อง แต่ไฟล์เดิมพิมพ์ CWE) — คอลัมน์เดียวกัน
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

REPORT_PATH = "./reports/preprocess_report.txt"
CLEAN_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "cicids2017_clean.parquet")
OUTPUT_DIR = os.path.dirname(CLEAN_OUTPUT_PATH)
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cicids2017_train.parquet")
VAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cicids2017_val.parquet")
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cicids2017_test.parquet")

UNSW_CLEAN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "unswnb15_clean.parquet")
UNSW_TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "unswnb15_train.parquet")
UNSW_VAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "unswnb15_val.parquet")
UNSW_TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "unswnb15_test.parquet")

# LOAO parquet pairs แยกไปอยู่ subfolder ของตัวเอง (ไม่ปนกับ clean/train/val/test หลัก)
LOAO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "loao")

# หมายเหตุ: 3 ค่าคงที่ของ raw UNSW-NB15 เดิม (UNSW_ATTACK_CAT_RENAME สำหรับแก้ 'Backdoors'
# vs 'Backdoor', UNSW_MIXED_TYPE_COLS สำหรับ sport/dsport/ct_ftp_cmd, UNSW_SENTINEL_COLS
# สำหรับ is_ftp_login/ct_flw_http_mthd) ถูกลบออกแล้ว — ไม่มีความหมายกับ CIC-UNSW-NB15
# (attack_cat มาจากการแปลงรหัสตัวเลขของเราเองใน load_unsw() เลยสะกดถูกอยู่แล้วเสมอ, Data.csv
# ไม่มีคอลัมน์ sport/dsport/ct_ftp_cmd/is_ftp_login/ct_flw_http_mthd เลย — ตรวจสอบแล้วว่า
# Data.csv เป็นตัวเลขล้วนทั้ง 76 คอลัมน์ ไม่มี NaN/Inf และ Init_Win_bytes_forward/backward
# (หลัง rename) ไม่มีค่า -1 sentinel แบบ CICIDS-2017 เลย) ถ้าจะย้อนกลับไปใช้ raw UNSW-NB15
# เดิม ต้องกู้ค่าคงที่ 3 ตัวนี้กลับมาด้วย

# สัดส่วน train/val/test สำหรับ main train/val/test cycle (M1a metrics: precision/recall/F1/AUC-ROC)
# นี่คือ split มาตรฐาน ไม่ใช่ LOAO (Leave-One-Attack-Out) holdout ซึ่งเป็นการทดลองแยกต่างหาก
# ที่ใช้ข้อมูลชุดเดียวกันนี้คนละแบบ (ดู LOAO_MIN_SAMPLES / report_attack_categories ด้านบน)
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42

# LOAO (Leave-One-Attack-Out): สัดส่วน BENIGN ที่กันไว้ทำ zero-day test ต่อ category
# (คนละก้อนกับ BENIGN ที่โมเดลเห็นตอน train) ใช้สัดส่วนเดียวกับ TEST_SIZE ของ split ปกติ
# เพื่อความสอดคล้อง ไม่ใช่ค่าที่ผูกกับ split_train_val_test() แต่อย่างใด
LOAO_BENIGN_HOLDOUT_SIZE = TEST_SIZE

INIT_WIN_BYTES_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# ตัวขีดที่พบในต้นฉบับ (en-dash / em-dash cp1252) รวมถึง replacement char เผื่อไฟล์ไหนเพี้ยนไปแล้ว
DASH_CHARS = ["–", "—", "�"]

# threshold ขั้นต่ำสำหรับถือว่า attack category มีขนาดใหญ่พอจะใช้ทำ LOAO
# (Leave-One-Attack-Out) แล้วได้ recall/FPR ที่น่าเชื่อถือ; ต่ำกว่านี้ใช้ได้แค่เชิงคุณภาพ
LOAO_MIN_SAMPLES = 1000

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
            f"ไม่พบไฟล์: '{path}' กรุณาตรวจสอบว่าตั้งค่า RAW_DATA_DIR ถูกต้อง "
            f"และไฟล์ถูกวางไว้ในตำแหน่งนั้นจริง"
        )
    return pd.read_csv(path, **kwargs)


# ----------------------------------------------------------------------------
# STEP 1: โหลด CICIDS-2017 ด้วย utf-8-sig (byte เพี้ยนเป็น U+FFFD มาจากต้นทางอยู่แล้ว
# ไม่ใช่ปัญหาการเลือก encoding ผิด — อ่านด้วย cp1252 จะยิ่งตีความ byte utf-8 ผิดซ้ำ)
# ----------------------------------------------------------------------------
def load_cicids():
    frames = []
    for fname in CICIDS_FILES:
        path = os.path.join(RAW_DATA_DIR, fname)
        df = safe_read_csv(path, low_memory=False, encoding="utf-8-sig")
        log(f"  โหลด {fname:55s} shape={df.shape}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.str.strip()
    return combined


# ----------------------------------------------------------------------------
# STEP 1b: โหลด CIC-UNSW-NB15 (Data.csv = feature columns ล้วน, ไม่มี identifier/Label;
# Label.csv = รหัสตัวเลข 0-9 ตัวเดียว) — ไม่มี key column ร่วมกันเลยในทั้งสองไฟล์ ต้อง join
# แบบ positional (แถวที่ i ของ Data.csv คู่กับแถวที่ i ของ Label.csv) ยืนยันแล้วว่าถูกต้อง
# (2026-07-02): ลำดับ label ที่ไม่ใช่ Benign ตรงกัน byte-ต่อ-byte กับลำดับใน
# CICFlowMeter_out.csv ทั้ง 89,583 แถว + สุ่มตรวจ feature vector 1 แถวเทียบกับ
# CICFlowMeter_out.csv ตรงกันเป๊ะ (ดู CLAUDE.md) — Data.csv/Label.csv มีจำนวนแถวเท่ากัน
# เสมอ (447,915 ทั้งคู่) ถ้าไม่เท่ากันคือไฟล์ผิดชุด/เสียหาย ต้อง fail ดังๆ ไม่ join ต่อ
# ----------------------------------------------------------------------------
def load_unsw():
    data_path = os.path.join(CIC_UNSW_DIR, CIC_UNSW_DATA_FILE)
    label_path = os.path.join(CIC_UNSW_DIR, CIC_UNSW_LABEL_FILE)

    data_df = safe_read_csv(data_path, low_memory=False, encoding="utf-8-sig")
    log(f"  โหลด {CIC_UNSW_DATA_FILE:25s} shape={data_df.shape}")
    label_df = safe_read_csv(label_path, low_memory=False, encoding="utf-8-sig")
    log(f"  โหลด {CIC_UNSW_LABEL_FILE:25s} shape={label_df.shape}")

    if len(data_df) != len(label_df):
        raise ValueError(
            f"Data.csv ({len(data_df)} แถว) กับ Label.csv ({len(label_df)} แถว) จำนวนแถวไม่เท่ากัน "
            f"-> positional join ไม่ปลอดภัย (ไฟล์อาจผิดชุด/เสียหาย) หยุดทันทีแทนที่จะเดา"
        )

    data_df = data_df.reset_index(drop=True)
    label_codes = label_df["Label"].reset_index(drop=True)

    unknown_codes = sorted(set(label_codes.unique().tolist()) - set(CIC_UNSW_LABEL_CODE_MAP.keys()))
    if unknown_codes:
        raise ValueError(
            f"พบรหัส label ใน Label.csv ที่ไม่มีอยู่ใน CIC_UNSW_LABEL_CODE_MAP (จาก Readme.txt): "
            f"{unknown_codes} -> ต้องเพิ่ม mapping ให้ครบก่อน"
        )

    combined = data_df.rename(columns=CIC_UNSW_COLUMN_RENAME)

    # สร้าง attack_cat (string, NaN=Benign) และ Label (binary 0/1) จากรหัสตัวเลขเดียวกัน ทำให้
    # สอง column นี้ consistent กันเองโดยอัตโนมัติ (คนละสถานการณ์กับ raw UNSW-NB15 เดิมที่
    # attack_cat/Label เป็นสอง column อิสระจากต้นทาง เลยต้อง cross-check ว่าตรงกันจริงไหม)
    category_names = label_codes.map(CIC_UNSW_LABEL_CODE_MAP)
    combined["attack_cat"] = category_names.where(category_names != "Benign", other=pd.NA).astype("string")
    combined["Label"] = (label_codes != 0).astype(int)

    return combined


# ----------------------------------------------------------------------------
# STEP 2: แก้ mojibake ในคอลัมน์ Label ด้วยการ normalize ตัวขีด
# ----------------------------------------------------------------------------
def normalize_label_dashes(df):
    labels = df["Label"].astype("string")
    for ch in DASH_CHARS:
        labels = labels.str.replace(ch, "-", regex=False)
    df["Label"] = labels
    return df


# ----------------------------------------------------------------------------
# STEP 3: Init_Win_bytes_forward/backward: -1 คือ sentinel -> แปลงเป็น NaN
# (แยกจาก negative-value cleaning ทั่วไป ห้ามใช้ logic เดียวกันกับคอลัมน์อื่น)
# ----------------------------------------------------------------------------
def fix_init_win_bytes_sentinel(df):
    missing_cols = [c for c in INIT_WIN_BYTES_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"ไม่พบคอลัมน์ที่คาดไว้: {missing_cols}")

    before_nan = {c: int(df[c].isna().sum()) for c in INIT_WIN_BYTES_COLS}
    sentinel_counts = {c: int((df[c] == -1).sum()) for c in INIT_WIN_BYTES_COLS}
    for c in INIT_WIN_BYTES_COLS:
        df[c] = df[c].replace(-1, np.nan)
    after_nan = {c: int(df[c].isna().sum()) for c in INIT_WIN_BYTES_COLS}

    log("--- CICIDS-2017: แก้ sentinel -1 ใน Init_Win_bytes_forward/backward ---")
    for c in INIT_WIN_BYTES_COLS:
        log(
            f"  {c:28s} NaN ก่อนแก้={before_nan[c]:>8d}  "
            f"พบ -1 (sentinel)={sentinel_counts[c]:>8d}  "
            f"NaN หลังแก้={after_nan[c]:>8d}"
        )
    return df, before_nan, after_nan


# ----------------------------------------------------------------------------
# STEP 3b: ทำความสะอาด UNSW-NB15 เฉพาะจุด — เดิมทำ whitespace-strip + แก้สะกด
# 'Backdoors'->'Backdoor' + cast คอลัมน์ type ปนกัน (sport/dsport/ct_ftp_cmd) เป็น str
# แต่ทั้งหมดนั้นเป็น quirk ของ raw UNSW-NB15 4 ไฟล์เดิมเท่านั้น ไม่มีผลกับ CIC-UNSW-NB15:
# attack_cat ตอนนี้มาจาก CIC_UNSW_LABEL_CODE_MAP ใน load_unsw() ตรงๆ เลยสะกดถูกและไม่มี
# whitespace อยู่แล้วเสมอ (ไม่ต้องแก้), และ Data.csv ไม่มีคอลัมน์ sport/dsport/ct_ftp_cmd เลย
# (ตรวจสอบแล้วว่าเป็นตัวเลขล้วนทั้ง 76 คอลัมน์) เหลือแค่ log สรุป + sanity check ว่า
# attack_cat/Label ที่ derive มาจากรหัสเดียวกันใน load_unsw() สอดคล้องกันจริง (ควรผ่านเสมอ
# เพราะมาจากแหล่งเดียวกัน แต่เก็บเช็คนี้ไว้เป็นตาข่ายดักบั๊กของโค้ดเราเอง)
# ----------------------------------------------------------------------------
def clean_unsw(df):
    log("--- CIC-UNSW-NB15: attack_cat unique values (ไม่รวม NaN) ---")
    log(f"  {sorted(df['attack_cat'].dropna().unique().tolist())}")

    log("--- CIC-UNSW-NB15: attack_cat value_counts ---")
    log(df["attack_cat"].value_counts(dropna=False).to_string())

    log("--- CIC-UNSW-NB15: ตรวจสอบ consistency ระหว่าง attack_cat กับ Label (derive จากรหัสเดียวกันใน load_unsw()) ---")
    is_attack_row = df["attack_cat"].notna()
    mismatch_should_be_1 = df.loc[is_attack_row & (df["Label"] != 1)]
    mismatch_should_be_0 = df.loc[~is_attack_row & (df["Label"] != 0)]
    log(f"  แถวที่ attack_cat ระบุว่าเป็นการโจมตี แต่ Label != 1 : {len(mismatch_should_be_1)} แถว")
    log(f"  แถวที่ attack_cat ว่าง/NaN (ควรเป็น normal) แต่ Label != 0 : {len(mismatch_should_be_0)} แถว")
    if len(mismatch_should_be_1) > 0 or len(mismatch_should_be_0) > 0:
        raise AssertionError(
            "attack_cat/Label ไม่สอดคล้องกันทั้งที่ derive จากรหัสเดียวกันใน load_unsw() "
            "-> บั๊กในโค้ดเราเอง ไม่ใช่ปัญหาข้อมูลต้นทาง ตรวจ load_unsw() ก่อนดำเนินการต่อ"
        )

    return df


# ----------------------------------------------------------------------------
# STEP 4: ทำความสะอาดข้อมูลทั่วไป (inf -> NaN, drop duplicates, dropna)
#
# สำคัญ: Init_Win_bytes_forward/backward ต้อง "ยกเว้น" จาก dropna ทั่วไปนี้เสมอ
# เพราะ STEP 3 เพิ่งแปลง sentinel -1 ของสองคอลัมน์นี้เป็น NaN ไปแล้ว (นับได้ 1,001,189
# และ 1,441,552 แถวตามลำดับ) ถ้าเอาสองคอลัมน์นี้เข้า dropna(subset=numeric_cols) ทั่วไป
# ด้วย จะเสียข้อมูลไปเกือบครึ่งชุดข้อมูลทันที ซึ่งขัดกับเหตุผลที่ STEP 3 เลือกแปลงเป็น NaN
# (คือ "ไม่ได้วัดค่า" ไม่ใช่ข้อมูลเสีย) แทนที่จะ drop ทิ้งไปตั้งแต่ต้น
# ----------------------------------------------------------------------------
def clean_common(df, sentinel_cols=None):
    if sentinel_cols is None:
        sentinel_cols = INIT_WIN_BYTES_COLS

    rows_before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    inf_counts = {
        c: int(np.isinf(df[c]).sum()) for c in numeric_cols if np.isinf(df[c]).any()
    }
    if inf_counts:
        df[list(inf_counts.keys())] = df[list(inf_counts.keys())].replace(
            [np.inf, -np.inf], np.nan
        )

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # กันไว้เผื่อ dataset นี้ไม่มีคอลัมน์ sentinel ตามที่ระบุมาเลย — ไม่ error, แค่ไม่มีอะไรให้ยกเว้น
    sentinel_cols_present = [c for c in sentinel_cols if c in df.columns]
    dropna_cols = [c for c in numeric_cols if c not in sentinel_cols_present]
    rows_before_dropna = len(df)
    df = df.dropna(subset=dropna_cols).reset_index(drop=True)
    rows_after = len(df)

    log(f"  Inf values พบและแปลงเป็น NaN: {inf_counts if inf_counts else 'ไม่มี'}")
    log(f"  Duplicate rows ที่ลบ: {n_dupes}")
    log(
        f"  คอลัมน์ที่ยกเว้นจาก dropna (sentinel, คง NaN ไว้ ไม่ drop แถว): "
        f"{sentinel_cols_present if sentinel_cols_present else 'ไม่มี (dataset นี้ไม่มีคอลัมน์ sentinel)'}"
    )
    log(f"  แถวก่อน dropna ทั่วไป: {rows_before_dropna}")
    log(f"  แถวหลัง dropna ทั่วไป: {rows_after} (ลดลง {rows_before_dropna - rows_after})")
    for c in sentinel_cols_present:
        remaining_nan = int(df[c].isna().sum())
        log(f"  {c}: NaN ที่ยังคงอยู่ในข้อมูล (ไม่ถูก drop) = {remaining_nan}")

    total_dropped = rows_before - rows_after
    pct = 100 * total_dropped / rows_before if rows_before else 0.0
    log(
        f"  รวมแถวที่หายไปจาก clean_common(): {total_dropped} จาก {rows_before} แถวเดิม "
        f"({pct:.2f}%)"
    )
    if total_dropped >= 500_000:
        log(
            "  [WARNING] จำนวนแถวที่หายไปสูงผิดปกติ (>= 500,000) — ตรวจสอบว่า "
            "Init_Win_bytes_forward/backward ถูกยกเว้นจาก dropna จริงหรือไม่ ก่อนดำเนินการต่อ"
        )

    return df


# ----------------------------------------------------------------------------
# STEP 5: เตือนถ้าไม่มีคอลัมน์ Timestamp (MachineLearningCSV ไม่มีคอลัมน์นี้)
# ----------------------------------------------------------------------------
def warn_missing_timestamp(df):
    time_cols = [c for c in df.columns if ("time" in c.lower() or "stamp" in c.lower())]
    if time_cols:
        log(f"  พบคอลัมน์เกี่ยวกับเวลา: {time_cols}")
        return True

    log(
        "  [WARNING] ไม่พบคอลัมน์ Timestamp ในชุดไฟล์นี้ (MachineLearningCSV ตัดออกจากการ "
        "แจกจ่ายสาธารณะแล้ว) -> ขั้นตอน simulated replay feed ใน enrich.py จะไม่มี timestamp จริงให้ใช้ "
        "ต้องตัดสินใจก่อนว่าจะ (a) ใช้ลำดับแถวเดิมในไฟล์เป็น proxy ของเวลา หรือ "
        "(b) เปลี่ยนไปใช้ชุดไฟล์ GeneratedLabelledFlows ที่มีคอลัมน์ Timestamp จริงแทน"
    )
    return False


# ----------------------------------------------------------------------------
# STEP 6: รายงานจำนวนแถวต่อ attack category พร้อม flag ว่าใช้ทำ LOAO ได้หรือไม่
# ----------------------------------------------------------------------------
def report_attack_categories(df):
    counts = df.loc[df["Label"] != "BENIGN", "Label"].value_counts()
    log(f"--- จำนวนแถวต่อ attack category (threshold LOAO = {LOAO_MIN_SAMPLES} แถว) ---")
    for label, n in counts.items():
        if n >= LOAO_MIN_SAMPLES:
            flag = "OK สำหรับ LOAO holdout"
        else:
            flag = "เล็กเกินไป -> ใช้เชิงคุณภาพเท่านั้น พร้อมระบุ caveat เรื่องขนาด sample"
        log(f"  {label:35s} n={n:>8d}  [{flag}]")
    return counts


# ----------------------------------------------------------------------------
# STEP 8: แบ่ง train/val/test แบบ stratified by Label (60/20/20, random_state=42)
#
# นี่คือ split สำหรับ main train/val/test cycle เท่านั้น (M1a metrics) ไม่เกี่ยวกับ LOAO
# holdout ซึ่งเป็น setup การทดลองแยกต่างหาก (leave-one-attack-out ทีละ category เช่น Bot,
# DDoS) ที่ใช้ข้อมูล cleaned เต็มชุดคนละแบบ — จะทำเป็นฟังก์ชันของตัวเองภายหลัง ไม่ผูกรวมกับ
# ฟังก์ชันนี้
# ----------------------------------------------------------------------------
def split_train_val_test(df, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
                          random_state=SPLIT_RANDOM_STATE):
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-9, (
        "train_size + val_size + test_size ต้องรวมกันเป็น 1.0"
    )

    train_df, rest_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["Label"],
        random_state=random_state,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=relative_test_size,
        stratify=rest_df["Label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    total_rows = len(df)
    for name, split_df in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        pct_of_total = 100 * len(split_df) / total_rows
        log(f"--- {name}: {len(split_df)} แถว ({pct_of_total:.2f}% ของทั้งหมด {total_rows} แถว) ---")
        dist = split_df["Label"].value_counts(normalize=True).mul(100).round(4)
        log(dist.to_string())

    return train_df, val_df, test_df


# ----------------------------------------------------------------------------
# LOAO (Leave-One-Attack-Out): zero-day holdout ต่อ category — แยกต่างหากจาก
# split_train_val_test() ทั้งหมดด้านบน (ไม่แก้/ไม่เรียกใช้ split_train_val_test(),
# clean_common(), หรือฟังก์ชัน load/clean อื่นใด) ทำงานบน df ที่ clean แล้ว
# (เทียบเท่า cicids2017_clean.parquet) โดยตรง เรียกทีละ category ได้ ไม่ hardcode
# ไว้ตัวเดียว เพื่อ loop ทำครบทุก category ที่ผ่าน threshold LOAO_MIN_SAMPLES
#
# Train          = BENIGN ทั้งหมด (ยกเว้นส่วนที่กันไว้ทำ zero-day test)
#                  + ทุก attack category ยกเว้น holdout_category
#                  (โมเดล unsupervised จะไม่เคยเห็นแถวของ category นี้ตอน train เลย)
# Zero-day test  = BENIGN ส่วนที่กันไว้ (คนละก้อนกับที่ train เห็น) + เฉพาะแถวของ
#                  holdout_category เท่านั้น
# ----------------------------------------------------------------------------
def build_loao_split(df, holdout_category, benign_holdout_size=LOAO_BENIGN_HOLDOUT_SIZE,
                      random_state=SPLIT_RANDOM_STATE):
    benign_mask = df["Label"] == "BENIGN"
    holdout_mask = df["Label"] == holdout_category

    if not holdout_mask.any():
        raise ValueError(f"ไม่พบแถวของ holdout_category='{holdout_category}' ใน df ที่ส่งเข้ามา")

    other_attack_df = df.loc[~benign_mask & ~holdout_mask]
    holdout_df = df.loc[holdout_mask]

    benign_train_df, benign_zeroday_df = train_test_split(
        df.loc[benign_mask],
        test_size=benign_holdout_size,
        random_state=random_state,
    )

    train_df = pd.concat([benign_train_df, other_attack_df], ignore_index=True)
    zeroday_test_df = pd.concat([benign_zeroday_df, holdout_df], ignore_index=True)

    return train_df, zeroday_test_df


def _slugify_category(category):
    """แปลงชื่อ attack category ให้เป็นชื่อไฟล์ปลอดภัย: lowercase, เว้นวรรค/ขีดกลาง (รวมกันเป็นก้อน) -> underscore เดียว"""
    slug = category.strip().lower()
    slug = re.sub(r"[\s\-]+", "_", slug)
    return slug


def loao_output_paths(category, output_dir=LOAO_OUTPUT_DIR):
    slug = _slugify_category(category)
    train_path = os.path.join(output_dir, f"loao_{slug}_train.parquet")
    test_path = os.path.join(output_dir, f"loao_{slug}_zeroday_test.parquet")
    return train_path, test_path


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    section("STEP 1: โหลด CICIDS-2017 (encoding=utf-8-sig)")
    try:
        df = load_cicids()
    except FileNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    log(f"CICIDS-2017 รวมทั้งหมด shape = {df.shape}")

    section("STEP 2: แก้ mojibake ใน Label")
    log("  ตัวอย่าง Label ก่อนแก้ (เฉพาะที่มีตัวขีด):")
    before_labels = sorted(
        l for l in df["Label"].unique() if any(ch in l for ch in DASH_CHARS)
    )
    log(f"    {before_labels}")
    df = normalize_label_dashes(df)
    log("  ตัวอย่าง Label หลังแก้:")
    after_labels = sorted(l for l in df["Label"].unique() if "-" in l or "Web Attack" in l)
    log(f"    {after_labels}")

    section("STEP 3: แก้ sentinel -1 ใน Init_Win_bytes_forward/backward")
    df, before_nan, after_nan = fix_init_win_bytes_sentinel(df)

    section("STEP 4: ทำความสะอาดข้อมูลทั่วไป (inf -> NaN, drop duplicates, dropna)")
    rows_before_clean = len(df)
    df = clean_common(df)
    rows_after_clean = len(df)
    for c in INIT_WIN_BYTES_COLS:
        assert after_nan[c] == int(df[c].isna().sum()) or df[c].isna().sum() <= after_nan[c], (
            f"{c}: NaN ถูกลบไปโดยไม่ได้ตั้งใจ ตรวจสอบ dropna_cols ใน clean_common()"
        )
    log(
        f"  ยืนยัน: แถวลดลงจาก {rows_before_clean} เหลือ {rows_after_clean} "
        f"(ลดลง {rows_before_clean - rows_after_clean}) — ไม่ใช่การลดลงจาก sentinel "
        f"Init_Win_bytes_forward/backward เพราะสองคอลัมน์นี้ถูกยกเว้นจาก dropna แล้ว"
    )

    section("STEP 5: ตรวจสอบคอลัมน์ Timestamp")
    warn_missing_timestamp(df)

    section("STEP 6: จำนวนแถวต่อ attack category + LOAO usability")
    counts = report_attack_categories(df)

    section("STEP 7: สรุปผลลัพธ์")
    log(f"CICIDS-2017 หลัง preprocessing: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    log(f"Label ที่มีตัวขีดเหลือ mojibake หลังแก้: "
        f"{sum(any(ch in l for ch in DASH_CHARS) for l in df['Label'].unique())} รายการ")
    for c in INIT_WIN_BYTES_COLS:
        log(f"  {c}: NaN ก่อนแก้={before_nan[c]}  NaN หลังแก้={after_nan[c]}  "
            f"(เพิ่มขึ้น {after_nan[c] - before_nan[c]})")
    usable = counts[counts >= LOAO_MIN_SAMPLES]
    too_small = counts[counts < LOAO_MIN_SAMPLES]
    log(f"Attack categories ที่ใช้ทำ LOAO ได้ ({len(usable)}): {usable.index.tolist()}")
    log(f"Attack categories ที่เล็กเกินไป ({len(too_small)}): {too_small.index.tolist()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(CLEAN_OUTPUT_PATH, index=False)
    log(f"เซฟข้อมูลที่แก้แล้วไว้ที่ {CLEAN_OUTPUT_PATH}")

    section(
        f"STEP 8: แบ่ง train/val/test (stratified by Label, "
        f"{TRAIN_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}, random_state={SPLIT_RANDOM_STATE})"
    )
    train_df, val_df, test_df = split_train_val_test(df)

    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_parquet(VAL_OUTPUT_PATH, index=False)
    test_df.to_parquet(TEST_OUTPUT_PATH, index=False)
    log(f"เซฟ train split ไว้ที่ {TRAIN_OUTPUT_PATH}")
    log(f"เซฟ val split ไว้ที่ {VAL_OUTPUT_PATH}")
    log(f"เซฟ test split ไว้ที่ {TEST_OUTPUT_PATH}")

    section("STEP 9: LOAO (Leave-One-Attack-Out) zero-day holdout ต่อ category")
    log(
        f"  Category ที่ผ่าน threshold LOAO (n>={LOAO_MIN_SAMPLES}), จะสร้าง holdout ให้ "
        f"({len(usable)}): {usable.index.tolist()}"
    )
    os.makedirs(LOAO_OUTPUT_DIR, exist_ok=True)
    loao_summary = []
    for category in usable.index.tolist():
        loao_train_df, loao_zeroday_df = build_loao_split(df, category)
        train_path, test_path = loao_output_paths(category)
        loao_train_df.to_parquet(train_path, index=False)
        loao_zeroday_df.to_parquet(test_path, index=False)

        n_train = len(loao_train_df)
        n_test = len(loao_zeroday_df)
        n_benign_test = int((loao_zeroday_df["Label"] == "BENIGN").sum())
        n_attack_test = n_test - n_benign_test
        pct_benign = 100 * n_benign_test / n_test if n_test else 0.0
        pct_attack = 100 * n_attack_test / n_test if n_test else 0.0

        log(
            f"  [{category}] train={n_train:>8d}  zeroday_test={n_test:>8d}  "
            f"(BENIGN={pct_benign:6.2f}%  {category}={pct_attack:6.2f}%)"
        )
        log(f"    เซฟ train ไว้ที่ {train_path}")
        log(f"    เซฟ zero-day test ไว้ที่ {test_path}")

        loao_summary.append({
            "category": category,
            "train_rows": n_train,
            "zeroday_test_rows": n_test,
            "pct_benign": pct_benign,
            "pct_attack": pct_attack,
        })

    log("--- LOAO: สรุปทุก category ---")
    for row in loao_summary:
        log(
            f"  {row['category']:35s} train={row['train_rows']:>8d}  "
            f"zeroday_test={row['zeroday_test_rows']:>8d}  "
            f"BENIGN={row['pct_benign']:6.2f}%  attack={row['pct_attack']:6.2f}%"
        )

    # ------------------------------------------------------------------------
    # "unswnb15" dataset slot: ตอนนี้ดึงจาก CIC-UNSW-NB15 (data/raw/cic-unsw-nb15/,
    # Data.csv + Label.csv) แทน raw UNSW-NB15 4 ไฟล์เดิม — pipeline แยกต่างหากจาก
    # CICIDS-2017 ทั้งหมดด้านบน (ไม่แตะ/ไม่ใช้ df เดิม) ชื่อไฟล์ output (unswnb15_*.parquet)
    # ไม่เปลี่ยน เพื่อไม่ให้ label_builder.py/feature_extractor.py ต้องแก้โค้ด
    # ------------------------------------------------------------------------
    section("STEP 1 (unswnb15): โหลด CIC-UNSW-NB15 (Data.csv + Label.csv, positional join)")
    try:
        unsw_df = load_unsw()
    except (FileNotFoundError, ValueError) as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    rows_raw_unsw = len(unsw_df)
    log(f"CIC-UNSW-NB15 รวมทั้งหมด shape = {unsw_df.shape}")

    section("STEP 2 (unswnb15): ตรวจสอบ attack_cat/Label ที่ derive จากรหัส Label.csv")
    unsw_df = clean_unsw(unsw_df)

    section("STEP 3 (unswnb15): ทำความสะอาดข้อมูลทั่วไป (inf -> NaN, drop duplicates, dropna)")
    rows_before_clean_unsw = len(unsw_df)
    unsw_df = clean_common(unsw_df, sentinel_cols=[])
    rows_after_clean_unsw = len(unsw_df)
    log(
        f"  unswnb15: แถวดิบ={rows_raw_unsw}  ก่อน clean_common={rows_before_clean_unsw}  "
        f"หลัง clean_common={rows_after_clean_unsw} "
        f"(ลดลงรวม {rows_raw_unsw - rows_after_clean_unsw})"
    )

    section("STEP 4 (unswnb15): สรุปผลลัพธ์")
    log(f"unswnb15 หลัง preprocessing: {unsw_df.shape[0]} แถว, {unsw_df.shape[1]} คอลัมน์")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unsw_df.to_parquet(UNSW_CLEAN_OUTPUT_PATH, index=False)
    log(f"เซฟข้อมูล unswnb15 ที่แก้แล้วไว้ที่ {UNSW_CLEAN_OUTPUT_PATH}")

    section(
        f"STEP 5 (unswnb15): แบ่ง train/val/test (stratified by Label, "
        f"{TRAIN_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}, random_state={SPLIT_RANDOM_STATE})"
    )
    unsw_train_df, unsw_val_df, unsw_test_df = split_train_val_test(unsw_df)

    unsw_train_df.to_parquet(UNSW_TRAIN_OUTPUT_PATH, index=False)
    unsw_val_df.to_parquet(UNSW_VAL_OUTPUT_PATH, index=False)
    unsw_test_df.to_parquet(UNSW_TEST_OUTPUT_PATH, index=False)
    log(f"เซฟ unswnb15 train split ไว้ที่ {UNSW_TRAIN_OUTPUT_PATH}")
    log(f"เซฟ unswnb15 val split ไว้ที่ {UNSW_VAL_OUTPUT_PATH}")
    log(f"เซฟ unswnb15 test split ไว้ที่ {UNSW_TEST_OUTPUT_PATH}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
