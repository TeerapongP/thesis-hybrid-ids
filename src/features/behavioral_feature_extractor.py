
import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.labeling.label_builder import apply_mitre_mapping

RAW_DIR = "./data/raw/GeneratedLabelledFlows/TrafficLabelling "
OUTPUT_DIR = "./data/processed/behavioral"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cicids2017_behavioral_labeled.parquet")
REPORT_PATH = "./reports/behavioral_feature_extractor_report.txt"

FILE_DAY_MAP = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

CP1252_FILES = {"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"}
INT_COLS_TO_RESTORE = ["Source Port", "Destination Port", "Protocol"]

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp"]
DAY_GROUP_COL = "day_group"

DASH_CHARS = ["–", "—", "�"]

INIT_WIN_BYTES_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

WINDOW_MINUTES = [1, 3, 5]

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
            f"ไม่พบไฟล์: '{path}' กรุณาตรวจสอบว่าตั้งค่า RAW_DIR ถูกต้อง "
            f"และไฟล์ถูกวางไว้ในตำแหน่งนั้นจริง"
        )
    return pd.read_csv(path, **kwargs)


def normalize_label_dashes(df):
    labels = df["Label"].astype("string")
    for ch in DASH_CHARS:
        labels = labels.str.replace(ch, "-", regex=False)
    df["Label"] = labels
    return df


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


def load_file(fname):
    path = os.path.join(RAW_DIR, fname)
    encoding = "cp1252" if fname in CP1252_FILES else "utf-8-sig"
    df = safe_read_csv(path, low_memory=False, encoding=encoding)
    df.columns = df.columns.str.strip()
    log(f"  โหลด {fname:60s} encoding={encoding:10s} shape={df.shape}")

    if fname in CP1252_FILES:
        rows_before = len(df)
        df = df.dropna(subset=["Label"]).reset_index(drop=True)
        n_dropped = rows_before - len(df)
        present_int_cols = [c for c in INT_COLS_TO_RESTORE if c in df.columns]
        df[present_int_cols] = df[present_int_cols].astype("int64")
        log(f"    ตัด blank filler rows (Label เป็น NaN): {n_dropped} แถว -> เหลือ {len(df)} แถว")

    parsed_ts = pd.to_datetime(df["Timestamp"], dayfirst=True)
    n_parse_fail = int(parsed_ts.isna().sum())
    if n_parse_fail:
        log(f"    [WARNING] Timestamp parse ไม่สำเร็จ {n_parse_fail} แถว ในไฟล์นี้")
    df["Timestamp"] = parsed_ts

    df[DAY_GROUP_COL] = FILE_DAY_MAP[fname]
    return df


def load_all_files():
    frames = []
    for fname in FILE_DAY_MAP:
        frames.append(load_file(fname))
    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_day_groups(combined):
    day_frames = []
    for day in DAY_ORDER:
        day_df = (
            combined.loc[combined[DAY_GROUP_COL] == day]
            .sort_values("Timestamp", kind="stable")
            .reset_index(drop=True)
        )
        t_min, t_max = day_df["Timestamp"].min(), day_df["Timestamp"].max()
        log(f"  {day:10s} rows={len(day_df):>8d}  range=[{t_min} .. {t_max}]")
        day_frames.append((day, day_df))
    return day_frames


def compute_failed_flag(protocol, rst, syn, ack, fwd_len, bwd_len):
    is_tcp = protocol == 6
    tcp_failed = (rst > 0) | ((syn > 0) & (ack == 0))
    non_tcp_failed = (fwd_len == 0) & (bwd_len == 0)
    return np.where(is_tcp, tcp_failed, non_tcp_failed)


def _sliding_window_features(ts_sec, dport, failed, window_seconds):
    n = len(ts_sec)
    conn_count = np.empty(n, dtype=np.int32)
    uniq_dport = np.empty(n, dtype=np.int32)
    fail_ratio = np.empty(n, dtype=np.float64)

    port_counts = {}
    distinct = 0
    failed_sum = 0
    lo = 0
    for i in range(n):
        p = dport[i]
        c = port_counts.get(p, 0)
        if c == 0:
            distinct += 1
        port_counts[p] = c + 1
        if failed[i]:
            failed_sum += 1

        threshold = ts_sec[i] - window_seconds
        while ts_sec[lo] < threshold:
            p_lo = dport[lo]
            port_counts[p_lo] -= 1
            if port_counts[p_lo] == 0:
                distinct -= 1
                del port_counts[p_lo]
            if failed[lo]:
                failed_sum -= 1
            lo += 1

        cnt = i - lo + 1
        conn_count[i] = cnt
        uniq_dport[i] = distinct
        fail_ratio[i] = failed_sum / cnt if cnt > 0 else 0.0

    return conn_count, uniq_dport, fail_ratio


def add_behavioral_features(day_df, window_minutes_list=WINDOW_MINUTES):
    n = len(day_df)
    new_cols = {}
    for w in window_minutes_list:
        new_cols[f"conn_count_{w}m"] = np.zeros(n, dtype=np.int32)
        new_cols[f"uniq_dport_count_{w}m"] = np.zeros(n, dtype=np.int32)
        new_cols[f"fail_ratio_{w}m"] = np.zeros(n, dtype=np.float64)
        new_cols[f"repeat_ratio_{w}m"] = np.zeros(n, dtype=np.float64)

    protocol = day_df["Protocol"].to_numpy()
    rst = day_df["RST Flag Count"].to_numpy()
    syn = day_df["SYN Flag Count"].to_numpy()
    ack = day_df["ACK Flag Count"].to_numpy()
    fwd_len = day_df["Total Length of Fwd Packets"].to_numpy()
    bwd_len = day_df["Total Length of Bwd Packets"].to_numpy()
    failed_all = compute_failed_flag(protocol, rst, syn, ack, fwd_len, bwd_len)

    dport_all = day_df["Destination Port"].to_numpy()
    ts_sec_all = day_df["Timestamp"].values.astype("datetime64[s]").astype(np.int64)

    for _, idx in day_df.groupby("Source IP", sort=False).indices.items():
        idx = np.asarray(idx)
        ts_sec = ts_sec_all[idx]
        dport = dport_all[idx]
        failed = failed_all[idx]
        for w in window_minutes_list:
            cc, ud, fr = _sliding_window_features(ts_sec, dport, failed, w * 60)
            new_cols[f"conn_count_{w}m"][idx] = cc
            new_cols[f"uniq_dport_count_{w}m"][idx] = ud
            new_cols[f"fail_ratio_{w}m"][idx] = fr
            new_cols[f"repeat_ratio_{w}m"][idx] = cc / ud

    day_df = day_df.copy()
    for col_name, arr in new_cols.items():
        day_df[col_name] = arr
    return day_df


def clean_common_behavioral(df, dedup_subset_cols):
    rows_before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    inf_counts = {c: int(np.isinf(df[c]).sum()) for c in numeric_cols if np.isinf(df[c]).any()}
    if inf_counts:
        df[list(inf_counts.keys())] = df[list(inf_counts.keys())].replace([np.inf, -np.inf], np.nan)

    n_dupes = int(df.duplicated(subset=dedup_subset_cols).sum())
    df = df.drop_duplicates(subset=dedup_subset_cols, keep="first").reset_index(drop=True)

    sentinel_cols_present = [c for c in INIT_WIN_BYTES_COLS if c in df.columns]
    dropna_cols = [c for c in numeric_cols if c not in sentinel_cols_present]
    rows_before_dropna = len(df)
    df = df.dropna(subset=dropna_cols).reset_index(drop=True)
    rows_after = len(df)

    log(f"  Inf values พบและแปลงเป็น NaN: {inf_counts if inf_counts else 'ไม่มี'}")
    log(f"  Duplicate rows ที่ลบ (subset={len(dedup_subset_cols)} คอลัมน์เดิม ไม่รวม ID/behavioral ใหม่): {n_dupes}")
    log(
        f"  คอลัมน์ที่ยกเว้นจาก dropna (sentinel, คง NaN ไว้ ไม่ drop แถว): "
        f"{sentinel_cols_present if sentinel_cols_present else 'ไม่มี'}"
    )
    log(f"  แถวก่อน dropna ทั่วไป: {rows_before_dropna}")
    log(f"  แถวหลัง dropna ทั่วไป: {rows_after} (ลดลง {rows_before_dropna - rows_after})")

    total_dropped = rows_before - rows_after
    pct = 100 * total_dropped / rows_before if rows_before else 0.0
    log(f"  รวมแถวที่หายไปจาก clean_common_behavioral(): {total_dropped} จาก {rows_before} แถวเดิม ({pct:.2f}%)")

    stats = {
        "rows_before": rows_before,
        "n_dupes": n_dupes,
        "rows_before_dropna": rows_before_dropna,
        "rows_after": rows_after,
    }
    return df, stats


def spot_check_signal(df):
    behavioral_cols = [
        f"{m}_{w}m" for w in WINDOW_MINUTES
        for m in ("conn_count", "uniq_dport_count", "fail_ratio", "repeat_ratio")
    ]
    benign_sub = df.loc[df["Label"] == "BENIGN", behavioral_cols]
    benign_stats = benign_sub.agg(["mean", "median"]).T

    log(f"--- BENIGN (n={len(benign_sub)}) mean/median (baseline) ---")
    log(benign_stats.to_string())

    for label in ["PortScan", "FTP-Patator", "SSH-Patator", "DDoS"]:
        sub = df.loc[df["Label"] == label, behavioral_cols]
        if len(sub) == 0:
            log(f"--- {label}: ไม่พบแถว (ข้าม) ---")
            continue
        n_ips = df.loc[df["Label"] == label, "Source IP"].nunique()
        log(f"--- {label} (n={len(sub)}, unique source IPs={n_ips}) เทียบกับ BENIGN ---")
        label_stats = sub.agg(["mean", "median"]).T
        comparison = pd.DataFrame({
            "benign_median": benign_stats["median"],
            f"{label}_median": label_stats["median"],
            "median_ratio_vs_benign": label_stats["median"] / benign_stats["median"].replace(0, np.nan),
            "benign_mean": benign_stats["mean"],
            f"{label}_mean": label_stats["mean"],
        })
        log(comparison.to_string())


def main():
    section("PHASE 1: โหลดไฟล์ทั้งหมดจาก data/raw/GeneratedLabelledFlows/TrafficLabelling  + per-file fixes")
    try:
        combined = load_all_files()
    except FileNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    log(f"รวมทั้งหมด shape = {combined.shape}")

    original_79_cols = [c for c in combined.columns if c not in ID_COLS and c != DAY_GROUP_COL]
    log(f"คอลัมน์ร่วมกับ cicids2017_clean.parquet (ใช้เป็น dedup subset ใน PHASE 4): {len(original_79_cols)} คอลัมน์")
    assert len(original_79_cols) == 79, (
        f"คาดว่าจะได้ 79 คอลัมน์ร่วมกับ cicids2017_clean.parquet แต่ได้ {len(original_79_cols)} "
        f"-> ตรวจ ID_COLS/DAY_GROUP_COL หรือโครงสร้างไฟล์ raw เปลี่ยนไปหรือไม่"
    )

    section("PHASE 1b: แก้ mojibake ใน Label (normalize_label_dashes)")
    combined = normalize_label_dashes(combined)

    section("PHASE 1c: แก้ sentinel -1 ใน Init_Win_bytes_forward/backward (fix_init_win_bytes_sentinel)")
    combined, before_nan, after_nan = fix_init_win_bytes_sentinel(combined)

    section("PHASE 1d: ตรวจสอบผล Timestamp parse (parse จริงทำต่อไฟล์ใน load_file() ก่อนหน้านี้แล้ว)")
    n_parse_fail_total = int(combined["Timestamp"].isna().sum())
    log(f"  Timestamp parse fail รวมทุกไฟล์: {n_parse_fail_total}")

    section("PHASE 2: Day-grouping — concat ไฟล์ในวันเดียวกัน แล้ว sort ตาม Timestamp (stable)")
    day_frames = build_day_groups(combined)

    section("PHASE 3: Rolling behavioral features ต่อ source IP (per day-group, causal, หน้าต่าง 1/3/5 นาที)")
    processed_day_frames = []
    for day, day_df in day_frames:
        day_df2 = add_behavioral_features(day_df, WINDOW_MINUTES)
        n_unique_ips = day_df2["Source IP"].nunique()
        log(f"  {day:10s} shape={day_df2.shape}  unique source IPs={n_unique_ips}")
        processed_day_frames.append(day_df2)

    full_df = pd.concat(processed_day_frames, ignore_index=True)
    log(f"รวมทุก day-group แล้ว shape = {full_df.shape}")

    section("PHASE 4: ทำความสะอาดทั่วไป + ลบ duplicate (subset=79 คอลัมน์เดิม, หลัง behavioral columns แล้ว)")
    full_df, dedup_stats = clean_common_behavioral(full_df, dedup_subset_cols=original_79_cols)
    log(
        f"  เทียบ ballpark: cicids2017_clean.parquet มี 2,520,798 แถว (ต่างกันได้เล็กน้อยจาก "
        f"~163-row duplicate-count discrepancy ระหว่างสอง distribution ตามที่ recon เจอ) — "
        f"ได้จริง {len(full_df)} แถว"
    )

    section("PHASE 5: MITRE ATT&CK label join (label_builder.apply_mitre_mapping, dataset='cicids2017')")
    full_df = apply_mitre_mapping(full_df, dataset_name="cicids2017", label_col="Label")

    section("PHASE 5b: Spot-check — behavioral feature แยกตาม BENIGN vs attack category ที่คาดว่าจะเห็นสัญญาณชัด")
    spot_check_signal(full_df)

    section("สรุปผลลัพธ์")
    log(f"Output สุดท้าย: {full_df.shape[0]} แถว, {full_df.shape[1]} คอลัมน์")
    log("ข้อจำกัดที่บันทึกไว้ (ดูรายละเอียดเต็มใน docstring ของสคริปต์นี้):")
    log("  1) Timestamp ระดับนาทีเท่านั้นใน 7/8 ไฟล์ -> เลือกหน้าต่าง 1/3/5 นาที ไม่ใช่ 10/30/60 วินาที")
    log("  2) หน้าต่างเวลาไม่ carry ข้ามวัน (แต่ละ day-group เป็น timeline ของตัวเอง)")
    log("  3) fail_ratio เป็น proxy จาก flow-level aggregate ไม่ใช่ ground truth จาก raw packet trace")
    log("     (ยืนยันจากรันจริง: จับ FTP/SSH-Patator ไม่ได้เลย เพราะ brute force ผ่าน TCP handshake สำเร็จปกติ)")
    log("  3b) repeat_ratio_{w}m (=conn_count/uniq_dport_count) เพิ่มมาปิด gap ของข้อ 3 -- ยืนยันจากรันจริง:")
    log("      median สูงกว่า BENIGN ~5-7 เท่าสำหรับ FTP/SSH-Patator (ดูตัวเลขเต็มใน PHASE 5b)")
    log("  4) ยังไม่ยืนยัน row-for-row join กับ cicids2017_clean.parquet เดิม (~163-row discrepancy ที่ recon เจอ)")
    log("  5) สคริปต์นี้หยุดที่ 'engineered feature ครบ ทั้ง population เดียว ยังไม่ split' -- ไม่มี scale/SMOTE/train-val-test/LOAO")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df.to_parquet(OUTPUT_PATH, index=False)
    log(f"เซฟข้อมูลไว้ที่ {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_REPORT_LINES))
    print(f"\n[บันทึกไฟล์] เซฟรายงานฉบับเต็มไว้ที่ {REPORT_PATH}")


if __name__ == "__main__":
    main()
