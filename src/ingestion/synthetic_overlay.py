"""Synthetic attack overlay with ground-truth inject timestamps (MTTD, metric M1b).

Generates at most 5% synthetic attack flows relative to a benign baseline and stamps each
flow with `inject_timestamp` at generation time. That column is the only ground-truth clock
for Mean Time To Detect; nothing downstream may recompute it.

Organizing structure: PROFILES, a registry with one AttackProfile per attack class. Adding
a class means appending a profile; `generate_overlay` never branches on a class name.

Output rows are in the scaled 92-feature space of
data/features/behavioral/cicids2017_behavioral_features_*.parquet, so they can be scored by
the behavioral IF/AE models and the fusion layer. Flows are first drawn in raw CICFlowMeter
units, then scaled with the min/max of the real train split (the MinMaxScaler
feature_extractor_behavioral.py fit), reconstructed from data/processed/behavioral.

Run from the repo root: python -m src.ingestion.synthetic_overlay
"""

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.features.behavioral_feature_extractor import (
    WINDOW_MINUTES,
    _sliding_window_features,
    compute_failed_flag,
)

SEED = 42
MAX_RATIO = 0.05
TARGET_RATIO = 0.03
SPAN_HOURS = 24
ANCHOR = pd.Timestamp("2026-01-01T00:00:00Z")

FEATURES_TEST = "data/features/behavioral/cicids2017_behavioral_features_test.parquet"
PROCESSED_TRAIN = "data/processed/behavioral/cicids2017_behavioral_train.parquet"
PROCESSED_TEST = "data/processed/behavioral/cicids2017_behavioral_test.parquet"
MITRE_CSV = "config/mitre_mapping.csv"
OUT_PARQUET = "data/synthetic/cicids2017_overlay.parquet"
OUT_REPORT = "reports/synthetic_overlay_report.txt"

SENTINEL_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
META_COLS = [
    "inject_id", "inject_timestamp", "synthetic", "attack_class", "campaign_id",
    "src_ip", "dst_ip", "technique_id", "technique_name", "tactic", "Label",
]


@dataclass(frozen=True)
class Dist:
    kind: str
    a: float = 0.0
    b: float = 0.0
    lo: float = 0.0
    hi: float = np.inf
    integer: bool = False
    values: tuple = ()
    probs: tuple = ()

    def draw(self, rng, n):
        if self.kind == "const":
            x = np.full(n, self.a, dtype=float)
        elif self.kind == "uniform":
            x = rng.uniform(self.a, self.b, n)
        elif self.kind == "lognormal":
            x = rng.lognormal(np.log(self.a), self.b, n)
        elif self.kind == "choice":
            x = rng.choice(np.array(self.values, dtype=float), n, p=self.probs or None)
        else:
            raise ValueError(self.kind)
        x = np.clip(x, self.lo, self.hi)
        return np.round(x) if self.integer else x


def const(v):
    return Dist("const", a=v)


def uniform(a, b, integer=False):
    return Dist("uniform", a=a, b=b, integer=integer)


def lognormal(median, sigma, lo=0.0, hi=np.inf, integer=False):
    return Dist("lognormal", a=median, b=sigma, lo=lo, hi=hi, integer=integer)


def choice(values, probs=()):
    return Dist("choice", values=tuple(values), probs=tuple(probs))


@dataclass(frozen=True)
class FlowModel:
    n_fwd: Dist
    n_bwd: Dist
    fwd_len_mean: Dist
    fwd_len_std: Dist
    bwd_len_mean: Dist
    bwd_len_std: Dist
    duration_us: Dist
    iat_cv: Dist
    psh_count: Dist
    init_win_fwd: Dist
    init_win_bwd: Dist
    completes_handshake: bool = True


# Temporal patterns: each factory returns f(rng, n) -> sorted offsets in seconds from campaign start.
def poisson(mean_gap):
    return lambda rng, n: np.cumsum(rng.exponential(mean_gap, n))


def periodic(period_range, jitter_frac):
    def f(rng, n):
        period = rng.uniform(*period_range)
        gaps = period * (1 + rng.normal(0, jitter_frac, n)).clip(0.2)
        return np.cumsum(gaps)
    return f


def on_off_bursts(burst_range, mean_gap, pause_range):
    def f(rng, n):
        gaps = rng.exponential(mean_gap, n)
        i = 0
        while i < n:
            i += int(rng.integers(*burst_range))
            if i < n:
                gaps[i] += rng.uniform(*pause_range)
        return np.cumsum(gaps)
    return f


def spaced(gap_range):
    return lambda rng, n: np.cumsum(rng.uniform(*gap_range, n))


# Destination-port patterns: f(rng, n) -> int array.
def fixed_port(*ports):
    return lambda rng, n: rng.choice(np.array(ports), n)


def sweep_ports(upper=10000):
    return lambda rng, n: rng.permutation(np.arange(1, upper + 1))[:n]


@dataclass(frozen=True)
class AttackProfile:
    name: str
    mitre_source_label: str | None
    share: float
    campaign_size: tuple
    temporal: Callable
    dports: Callable
    n_sources: tuple
    src_zone: str
    dst_zone: str
    flow: FlowModel
    pattern: str


PROFILES = (
    AttackProfile(
        name="brute_force", mitre_source_label="SSH-Patator", share=0.25,
        campaign_size=(150, 500), temporal=poisson(0.8), dports=fixed_port(22),
        n_sources=(1, 1), src_zone="external", dst_zone="internal",
        pattern="one source, one SSH target, steady jittered stream of completed handshakes",
        flow=FlowModel(
            n_fwd=lognormal(14, 0.2, 8, 30, True), n_bwd=lognormal(12, 0.2, 6, 26, True),
            fwd_len_mean=lognormal(60, 0.25, 20, 200), fwd_len_std=lognormal(55, 0.3, 5, 300),
            bwd_len_mean=lognormal(70, 0.25, 20, 250), bwd_len_std=lognormal(70, 0.3, 5, 300),
            duration_us=lognormal(1.6e6, 0.35, 2e5, 2e7), iat_cv=uniform(0.8, 1.6),
            psh_count=lognormal(8, 0.3, 2, 20, True),
            init_win_fwd=choice([29200, 64240], [0.5, 0.5]), init_win_bwd=choice([28960, 65160], [0.5, 0.5]),
        ),
    ),
    AttackProfile(
        name="port_scan", mitre_source_label="PortScan", share=0.20,
        campaign_size=(500, 1500), temporal=poisson(0.004), dports=sweep_ports(),
        n_sources=(1, 1), src_zone="external", dst_zone="internal",
        pattern="single-source burst of SYN probes across many distinct ports; closed ports answer RST",
        flow=FlowModel(
            n_fwd=choice([1, 2], [0.85, 0.15]), n_bwd=choice([0, 1], [0.7, 0.3]),
            fwd_len_mean=const(0), fwd_len_std=const(0), bwd_len_mean=const(0), bwd_len_std=const(0),
            duration_us=lognormal(2e4, 0.9, 1e2, 3e6), iat_cv=uniform(0.0, 0.5),
            psh_count=const(0),
            init_win_fwd=choice([1024, 2048, 4096], [0.4, 0.3, 0.3]), init_win_bwd=const(0),
            completes_handshake=False,
        ),
    ),
    AttackProfile(
        name="data_exfil", mitre_source_label=None, share=0.10,
        campaign_size=(3, 12), temporal=spaced((60, 900)), dports=fixed_port(443, 22, 21),
        n_sources=(1, 1), src_zone="internal", dst_zone="external",
        pattern="few long, upload-heavy flows from an internal host to one external endpoint, minutes apart",
        flow=FlowModel(
            n_fwd=lognormal(6000, 0.6, 800, 40000, True), n_bwd=lognormal(3000, 0.6, 400, 20000, True),
            fwd_len_mean=lognormal(1250, 0.1, 600, 1460), fwd_len_std=lognormal(300, 0.4, 20, 600),
            bwd_len_mean=lognormal(10, 0.8, 0, 120), bwd_len_std=lognormal(15, 0.8, 0, 150),
            duration_us=lognormal(1.2e8, 0.6, 1e7, 6e8), iat_cv=uniform(1.0, 3.0),
            psh_count=lognormal(1500, 0.6, 100, 15000, True),
            init_win_fwd=choice([29200, 64240], [0.5, 0.5]), init_win_bwd=choice([501, 1002, 8192]),
        ),
    ),
    AttackProfile(
        name="c2_beacon", mitre_source_label="Bot", share=0.20,
        campaign_size=(40, 120), temporal=periodic((30, 300), 0.15), dports=fixed_port(443, 8080),
        n_sources=(1, 1), src_zone="internal", dst_zone="external",
        pattern="fixed-period check-ins with roughly 15% gaussian jitter to one external C2 host",
        flow=FlowModel(
            n_fwd=lognormal(5, 0.25, 3, 12, True), n_bwd=lognormal(5, 0.25, 3, 12, True),
            fwd_len_mean=lognormal(150, 0.3, 40, 400), fwd_len_std=lognormal(120, 0.3, 10, 400),
            bwd_len_mean=lognormal(450, 0.4, 60, 1200), bwd_len_std=lognormal(400, 0.4, 20, 900),
            duration_us=lognormal(6e5, 0.6, 5e4, 1e7), iat_cv=uniform(0.8, 2.0),
            psh_count=lognormal(3, 0.3, 1, 8, True),
            init_win_fwd=choice([8192, 29200, 64240]), init_win_bwd=choice([28960, 65535]),
        ),
    ),
    AttackProfile(
        name="credential_stuffing", mitre_source_label="Web Attack - Brute Force", share=0.25,
        campaign_size=(300, 900), temporal=on_off_bursts((20, 60), 0.2, (5, 30)),
        dports=fixed_port(443, 80), n_sources=(3, 8), src_zone="external", dst_zone="internal",
        pattern="bursts of short failed-login HTTP requests spread over a small proxy pool, then pauses",
        flow=FlowModel(
            n_fwd=lognormal(8, 0.2, 5, 14, True), n_bwd=lognormal(6, 0.2, 4, 12, True),
            fwd_len_mean=lognormal(420, 0.25, 150, 900), fwd_len_std=lognormal(450, 0.3, 50, 1000),
            bwd_len_mean=lognormal(260, 0.3, 80, 700), bwd_len_std=lognormal(330, 0.3, 30, 800),
            duration_us=lognormal(3.5e5, 0.4, 5e4, 3e6), iat_cv=uniform(0.8, 1.8),
            psh_count=lognormal(4, 0.3, 2, 10, True),
            init_win_fwd=choice([29200, 64240, 65535]), init_win_bwd=choice([28960, 65160]),
        ),
    ),
)


def load_mitre_lookup(path=MITRE_CSV):
    m = pd.read_csv(path)
    m = m[m["dataset"] == "cicids2017"].set_index("source_label")
    return m[["technique_id", "technique_name", "tactic"]].to_dict("index")


def feature_columns(features_path=FEATURES_TEST):
    cols = list(pd.read_parquet(features_path).columns[:92])
    assert cols[-2:] == ["Init_Win_bytes_forward_missing", "Init_Win_bytes_backward_missing"]
    return cols


@dataclass(frozen=True)
class RealScaler:
    cols: tuple
    lo: np.ndarray
    span: np.ndarray
    medians: dict

    def transform(self, df):
        return (df[list(self.cols)].to_numpy(float) - self.lo) / self.span


def fit_real_scaler(cols, train_path=PROCESSED_TRAIN):
    scaled_cols = [c for c in cols if not c.endswith("_missing")]
    tr = pd.read_parquet(train_path, columns=scaled_cols)
    lo, hi = tr.min().to_numpy(float), tr.max().to_numpy(float)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return RealScaler(tuple(scaled_cols), lo, span, {c: float(tr[c].median()) for c in SENTINEL_COLS})


def verify_scaler(scaler, n=2000, test_path=PROCESSED_TEST, features_path=FEATURES_TEST):
    """Max abs error between our scaling of raw test rows and the pipeline's own scaled rows."""
    raw = pd.read_parquet(test_path)
    feat = pd.read_parquet(features_path).set_index("orig_row_index")
    raw = raw.set_index("orig_row_index")
    ids = raw.index.intersection(feat.index)
    ids = pd.Index(np.random.default_rng(SEED).choice(ids, min(n, len(ids)), replace=False))
    r = raw.loc[ids]
    r = r.copy()
    for c in SENTINEL_COLS:
        r[c] = r[c].fillna(scaler.medians[c])
    mine = scaler.transform(r)
    ref = feat.loc[ids, list(scaler.cols)].to_numpy(float)
    return float(np.abs(mine - ref).max())


def _ips(rng, zone, k):
    if zone == "external":
        return [f"203.0.113.{x}" for x in rng.choice(np.arange(1, 255), k, replace=False)]
    return [f"10.20.{rng.integers(0, 16)}.{rng.integers(1, 255)}" for _ in range(k)]


def raw_flow_columns(p, rng, n, dports):
    """Derive the 78 raw CICFlowMeter columns from the profile's primitives."""
    f = p.flow
    nf, nb = f.n_fwd.draw(rng, n), f.n_bwd.draw(rng, n)
    fm, fs = f.fwd_len_mean.draw(rng, n), f.fwd_len_std.draw(rng, n)
    bm, bs = f.bwd_len_mean.draw(rng, n), f.bwd_len_std.draw(rng, n)
    dur = np.maximum(f.duration_us.draw(rng, n), 1.0)
    cv = f.iat_cv.draw(rng, n)
    psh = f.psh_count.draw(rng, n)
    nt = nf + nb
    fs = np.where(nf > 1, fs, 0.0)
    bs = np.where(nb > 1, bs, 0.0)
    fmax = np.where(nf > 1, np.minimum(fm + 2 * fs, 1460), fm)
    fmin = np.where(nf > 1, np.maximum(fm - 2 * fs, 0), fm)
    bmax = np.where(nb > 1, np.minimum(bm + 2 * bs, 1460), bm)
    bmin = np.where(nb > 1, np.maximum(bm - 2 * bs, 0), bm)
    tf, tb = nf * fm, nb * bm
    sec = dur / 1e6

    if f.completes_handshake:
        syn, fin, rst = np.full(n, 2.0), np.full(n, 2.0), np.zeros(n)
        ack = np.maximum(nt - 1, 0)
    else:
        syn, fin, rst, ack = np.ones(n), np.zeros(n), (nb > 0).astype(float), np.zeros(n)

    gm = dur / np.maximum(nt - 1, 1)
    fwd_gm = dur / np.maximum(nf - 1, 1)
    bwd_gm = dur / np.maximum(nb - 1, 1)

    def iat(mean, cnt):
        ok = cnt > 1
        z = np.zeros(n)
        return (np.where(ok, mean, z), np.where(ok, mean * cv, z),
                np.where(ok, np.minimum(dur, mean * (1 + 3 * cv)), z),
                np.where(ok, np.maximum(mean * (1 - cv), 0), z))

    fi_mean, fi_std, fi_max, fi_min = iat(fwd_gm, nf)
    bi_mean, bi_std, bi_max, bi_min = iat(bwd_gm, nb)
    ai_mean, ai_std, ai_max, ai_min = iat(gm, nt)
    pm = (tf + tb) / np.maximum(nt, 1)
    pvar = (nf * (fs**2 + (fm - pm) ** 2) + nb * (bs**2 + (bm - pm) ** 2)) / np.maximum(nt, 1)
    z = np.zeros(n)
    has_both = nb > 0
    cols = {
        "Destination Port": dports.astype(float), "Flow Duration": dur,
        "Total Fwd Packets": nf, "Total Backward Packets": nb,
        "Total Length of Fwd Packets": tf, "Total Length of Bwd Packets": tb,
        "Fwd Packet Length Max": fmax, "Fwd Packet Length Min": fmin,
        "Fwd Packet Length Mean": fm, "Fwd Packet Length Std": fs,
        "Bwd Packet Length Max": bmax, "Bwd Packet Length Min": bmin,
        "Bwd Packet Length Mean": bm, "Bwd Packet Length Std": bs,
        "Flow Bytes/s": (tf + tb) / sec, "Flow Packets/s": nt / sec,
        "Flow IAT Mean": ai_mean, "Flow IAT Std": ai_std, "Flow IAT Max": ai_max, "Flow IAT Min": ai_min,
        "Fwd IAT Total": np.where(nf > 1, dur, 0.0), "Fwd IAT Mean": fi_mean, "Fwd IAT Std": fi_std,
        "Fwd IAT Max": fi_max, "Fwd IAT Min": fi_min,
        "Bwd IAT Total": np.where(nb > 1, dur, 0.0), "Bwd IAT Mean": bi_mean, "Bwd IAT Std": bi_std,
        "Bwd IAT Max": bi_max, "Bwd IAT Min": bi_min,
        "Fwd PSH Flags": (psh > 0).astype(float), "Bwd PSH Flags": z, "Fwd URG Flags": z, "Bwd URG Flags": z,
        "Fwd Header Length": 20 * nf, "Bwd Header Length": 20 * nb,
        "Fwd Packets/s": nf / sec, "Bwd Packets/s": nb / sec,
        "Min Packet Length": np.where(has_both, np.minimum(fmin, bmin), fmin),
        "Max Packet Length": np.maximum(fmax, bmax),
        "Packet Length Mean": pm, "Packet Length Std": np.sqrt(pvar), "Packet Length Variance": pvar,
        "FIN Flag Count": fin, "SYN Flag Count": syn, "RST Flag Count": rst, "PSH Flag Count": psh,
        "ACK Flag Count": ack, "URG Flag Count": z, "CWE Flag Count": z, "ECE Flag Count": z,
        "Down/Up Ratio": np.floor(nb / np.maximum(nf, 1)), "Average Packet Size": pm,
        "Avg Fwd Segment Size": fm, "Avg Bwd Segment Size": bm, "Fwd Header Length.1": 20 * nf,
        "Fwd Avg Bytes/Bulk": z, "Fwd Avg Packets/Bulk": z, "Fwd Avg Bulk Rate": z,
        "Bwd Avg Bytes/Bulk": z, "Bwd Avg Packets/Bulk": z, "Bwd Avg Bulk Rate": z,
        "Subflow Fwd Packets": nf, "Subflow Fwd Bytes": tf, "Subflow Bwd Packets": nb, "Subflow Bwd Bytes": tb,
        "Init_Win_bytes_forward": f.init_win_fwd.draw(rng, n),
        "Init_Win_bytes_backward": np.where(has_both, f.init_win_bwd.draw(rng, n), np.nan),
        "act_data_pkt_fwd": np.where(tf > 0, np.maximum(nf - 2, 0), 0.0),
        "min_seg_size_forward": np.full(n, 20.0),
        "Active Mean": dur, "Active Std": z, "Active Max": dur, "Active Min": dur,
        "Idle Mean": z, "Idle Std": z, "Idle Max": z, "Idle Min": z,
    }
    return cols, {"rst": rst, "syn": syn, "ack": ack, "fwd_len": tf, "bwd_len": tb}


def add_windowed_features(df, failed):
    """Per-source-IP trailing-window features, reusing the real extractor's sweep and fail proxy."""
    out = {f"{m}_{w}m": np.zeros(len(df)) for w in WINDOW_MINUTES
           for m in ("conn_count", "uniq_dport_count", "fail_ratio", "repeat_ratio")}
    ts = (df["inject_timestamp"] - ANCHOR).dt.total_seconds().to_numpy()
    dport, src = df["Destination Port"].to_numpy(), df["src_ip"].to_numpy()
    for ip in np.unique(src):
        idx = np.flatnonzero(src == ip)
        idx = idx[np.argsort(ts[idx], kind="stable")]
        for w in WINDOW_MINUTES:
            cc, ud, fr = _sliding_window_features(ts[idx], dport[idx], failed[idx], w * 60)
            out[f"conn_count_{w}m"][idx], out[f"uniq_dport_count_{w}m"][idx] = cc, ud
            out[f"fail_ratio_{w}m"][idx], out[f"repeat_ratio_{w}m"][idx] = fr, cc / ud
    return out


def generate_campaigns(p, rng, budget, span_s, mitre):
    """Yield one DataFrame of raw-unit flows per campaign until the class budget is met."""
    tech = mitre.get(p.mitre_source_label, {})
    frames, remaining, k = [], budget, 0
    while remaining > 0:
        size = min(int(rng.integers(p.campaign_size[0], p.campaign_size[1] + 1)), remaining)
        offsets = np.round(p.temporal(rng, size), 6)
        start = rng.uniform(0, max(0.0, span_s - offsets[-1]))
        n_src = int(rng.integers(p.n_sources[0], p.n_sources[1] + 1))
        srcs, dst = _ips(rng, p.src_zone, n_src), _ips(rng, p.dst_zone, 1)[0]
        dports = np.asarray(p.dports(rng, size))
        cols, flag_inputs = raw_flow_columns(p, rng, size, dports)
        df = pd.DataFrame(cols)
        df["inject_timestamp"] = ANCHOR + pd.to_timedelta(np.round((start + offsets) * 1e6), unit="us")
        df["src_ip"] = rng.choice(np.array(srcs), size)
        df["dst_ip"] = dst
        df["campaign_id"] = f"{p.name}-{k:04d}"
        df["_failed"] = compute_failed_flag(
            6, flag_inputs["rst"], flag_inputs["syn"], flag_inputs["ack"],
            flag_inputs["fwd_len"], flag_inputs["bwd_len"])
        df["technique_id"] = tech.get("technique_id")
        df["technique_name"] = tech.get("technique_name")
        df["tactic"] = tech.get("tactic")
        frames.append(df)
        remaining -= size
        k += 1
    return pd.concat(frames, ignore_index=True)


def generate_overlay(n_benign, scaler, cols, seed=SEED, target_ratio=TARGET_RATIO,
                     profiles=PROFILES, mitre=None):
    """Return the overlay DataFrame. Raises if the achieved ratio would exceed MAX_RATIO."""
    mitre = load_mitre_lookup() if mitre is None else mitre
    n_total = int(target_ratio / (1 - target_ratio) * n_benign)
    shares = np.array([p.share for p in profiles])
    budgets = np.floor(n_total * shares / shares.sum()).astype(int)
    parts = []
    for p, budget in zip(profiles, budgets):
        rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(p.name.encode())]))
        d = generate_campaigns(p, rng, int(budget), SPAN_HOURS * 3600.0, mitre)
        d["attack_class"] = p.name
        d["inject_id"] = [f"inj-{p.name}-{i:06d}" for i in range(len(d))]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["inject_timestamp", "inject_id"], kind="stable").reset_index(drop=True)

    df = df.join(pd.DataFrame(add_windowed_features(df, df["_failed"].to_numpy(bool))))
    df["Init_Win_bytes_forward_missing"] = df["Init_Win_bytes_forward"].isna().astype(np.int64)
    df["Init_Win_bytes_backward_missing"] = df["Init_Win_bytes_backward"].isna().astype(np.int64)
    for c in SENTINEL_COLS:
        df[c] = df[c].fillna(scaler.medians[c])
    scaled = pd.DataFrame(scaler.transform(df), columns=list(scaler.cols), index=df.index)

    out = pd.concat([df[["inject_id", "inject_timestamp"]], pd.Series(True, index=df.index, name="synthetic"),
                     df[["attack_class", "campaign_id", "src_ip", "dst_ip",
                         "technique_id", "technique_name", "tactic"]],
                     pd.Series("SYNTHETIC", index=df.index, name="Label"),
                     scaled, df[["Init_Win_bytes_forward_missing", "Init_Win_bytes_backward_missing"]]], axis=1)
    out = out[META_COLS + cols]
    ratio = len(out) / (n_benign + len(out))
    if ratio > MAX_RATIO:
        raise ValueError(f"injection ratio {ratio:.4f} exceeds {MAX_RATIO}")
    return out


def mttd_seconds(overlay, detections, per="campaign"):
    """MTTD in seconds. `detections` has columns inject_id and detected_at (tz-aware).

    per="flow": detected_at - inject_timestamp for each injected flow, first alert per inject_id.
    per="campaign": first alert across a campaign minus the campaign's earliest inject_timestamp.
    Undetected flows/campaigns are omitted from the result (count them as misses separately).
    """
    first = detections.groupby("inject_id")["detected_at"].min().rename("detected_at")
    j = overlay[["inject_id", "inject_timestamp", "campaign_id", "attack_class"]].join(first, on="inject_id")
    j = j.dropna(subset=["detected_at"])
    if per == "flow":
        return (j["detected_at"] - j["inject_timestamp"]).dt.total_seconds().rename("mttd_s")
    g = j.groupby(["attack_class", "campaign_id"])
    onset = overlay.groupby("campaign_id")["inject_timestamp"].min()
    first_det = g["detected_at"].min().reset_index().set_index("campaign_id")["detected_at"]
    return (first_det - onset.loc[first_det.index]).dt.total_seconds().rename("mttd_s")


def write_report(overlay, n_benign, scaler_err, path=OUT_REPORT):
    n = len(overlay)
    ratio = n / (n_benign + n)
    counts = overlay["attack_class"].value_counts()
    camp = overlay.groupby("attack_class")["campaign_id"].nunique()
    t0, t1 = overlay["inject_timestamp"].min(), overlay["inject_timestamp"].max()
    ex = overlay[overlay["attack_class"] == "c2_beacon"].iloc[:3]
    detect = ex["inject_timestamp"].iloc[0] + pd.Timedelta(seconds=7.5)
    lines = []
    w = lines.append
    bar = "=" * 70
    w(bar)
    w("SYNTHETIC ATTACK OVERLAY (inject timestamps for MTTD, metric M1b)")
    w(bar)
    w(f"Seed: {SEED}   Generator: src/ingestion/synthetic_overlay.py   Output: {OUT_PARQUET}")
    w("")
    w(bar)
    w("1. INJECTION RATIO (MEASURED)")
    w(bar)
    w(f"Benign baseline: BENIGN rows of {FEATURES_TEST} = {n_benign:,}")
    w(f"Synthetic flows injected                              = {n:,}")
    w(f"Achieved ratio = {n:,} / ({n_benign:,} + {n:,}) = {ratio:.6f} ({ratio*100:.3f}%)")
    w(f"Ceiling {MAX_RATIO*100:.0f}%: {'HELD' if ratio <= MAX_RATIO else 'VIOLATED'}; "
      f"generate_overlay raises ValueError above the ceiling, so a violating file cannot be written.")
    w("")
    w(bar)
    w("2. PER-CLASS COUNTS")
    w(bar)
    for p in PROFILES:
        tid = overlay.loc[overlay["attack_class"] == p.name, "technique_id"].iloc[0]
        w(f"{p.name:<20} flows={counts[p.name]:>6,}  campaigns={camp[p.name]:>3}  "
          f"MITRE={tid if isinstance(tid, str) else 'none in config/mitre_mapping.csv'}")
        w(f"{'':<20} pattern: {p.pattern}")
    w("")
    w(bar)
    w("3. TEMPORAL SPAN")
    w(bar)
    w(f"Stream window: {ANCHOR.isoformat()} for {SPAN_HOURS} h (fixed anchor, not wall clock, for reproducibility).")
    w(f"First inject: {t0.isoformat()}   Last inject: {t1.isoformat()}   Span: {t1 - t0}")
    w("The real parquet files carry no timestamps (stripped for privacy), so the benign baseline has no")
    w("clock. Whoever merges the overlay into a replay assigns benign timestamps; the overlay's own")
    w("inject_timestamp values are authoritative and must not be recomputed.")
    w("")
    w(bar)
    w("4. SCHEMA AND FIDELITY")
    w(bar)
    w("Rows are in the scaled 92-feature space of cicids2017_behavioral_features_*.parquet. Raw values")
    w("are drawn first, then scaled with the real train-split min/max (the pipeline's MinMaxScaler,")
    w("reconstructed from data/processed/behavioral). Verification on 2,000 real test rows: max abs")
    w(f"difference between this scaling and the pipeline's scaled values = {scaler_err:.2e}.")
    w("Real columns also exceed [0,1] slightly because test rows fall outside the train range; no clipping.")
    w("")
    w("Tier A, MODELLED per attack class (drawn from the profile's distributions):")
    w("  Destination Port, Flow Duration, Total Fwd/Backward Packets, Fwd/Bwd Packet Length Mean/Std,")
    w("  PSH/SYN/FIN/RST/ACK counts (from handshake outcome), Init_Win_bytes_forward/backward,")
    w("  Init_Win_*_missing flags (backward missing when the target never replies, as in real data).")
    w("Tier B, DERIVED deterministically from tier A (exact given the primitives):")
    w("  Total Length Fwd/Bwd, Subflow*, Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s, Down/Up Ratio,")
    w("  Average Packet Size, Avg Fwd/Bwd Segment Size, Packet Length Mean/Std/Variance.")
    w("Tier C, HEURISTIC approximations (plausible, not drawn from real attack captures):")
    w("  Fwd/Bwd Packet Length Max/Min (mean +/- 2 std, capped at 1460), all IAT Mean/Std/Max/Min and")
    w("  Fwd/Bwd IAT Total (mean = duration / gaps, std = cv * mean with cv drawn per profile),")
    w("  Fwd/Bwd Header Length and Fwd Header Length.1 (20 bytes per packet), act_data_pkt_fwd (n_fwd - 2),")
    w("  Fwd PSH Flags (1 when any PSH), min_seg_size_forward (20).")
    w("Tier D, PLACEHOLDER defaults, not modelled:")
    w("  Active Mean/Max/Min = flow duration, Active Std = 0, all Idle* = 0 (each flow is one active period),")
    w("  Bwd PSH Flags, Fwd/Bwd URG Flags, URG/CWE/ECE Flag Count = 0,")
    w("  all six Fwd/Bwd Avg Bytes/Packets/Bulk and Bulk Rate columns = 0.")
    w("Tier E, WINDOWED behavioral columns (12): computed, not drawn, with the real extractor's own")
    w("  _sliding_window_features and compute_failed_flag. Caveat: windows see only overlay flows from the")
    w("  same synthetic source IP, never the benign background. Synthetic sources use TEST-NET-3")
    w("  (203.0.113.0/24) or 10.20.0.0/16, so they do not collide with real CICIDS-2017 hosts.")
    w("Metadata, not model input: inject_id, inject_timestamp, synthetic, attack_class, campaign_id,")
    w("  src_ip, dst_ip, technique_id/name, tactic, Label ('SYNTHETIC'). The four CTI-encoded columns are")
    w("  omitted: they are not detection-model input (CLAUDE.md) and encodings were fit on real labels.")
    w("MITRE notes: data_exfil has no matching row in config/mitre_mapping.csv, so technique is empty.")
    w("  credential_stuffing uses T1110 (parent); T1110.004 is not in the mapping file.")
    w("Fidelity limit: the five profiles are hand-set distributions informed by attack mechanics, not fitted")
    w("  to real captures. Treat detector recall on this overlay as a measure of timing and plumbing, not")
    w("  as evidence of real-world detection performance.")
    w("")
    w(bar)
    w("5. COMPUTING MTTD FROM inject_timestamp")
    w(bar)
    w("inject_timestamp is set once at generation (flow start time, UTC, tz-aware) and never recomputed.")
    w("Per flow:      MTTD_flow = detected_at(inject_id) - inject_timestamp, where detected_at is the time")
    w("               the pipeline first raised an alert for that flow. Join alerts to the overlay on inject_id.")
    w("Per campaign:  MTTD_campaign = min(detected_at over the campaign's flows) - min(inject_timestamp over")
    w("               the campaign). This is the analyst-relevant figure: how long the attack ran unseen.")
    w("Report the mean (and median) across detected campaigns per attack_class. Campaigns with no alert are")
    w("misses: count them separately; do not drop them silently or assign them an MTTD.")
    w("Code: mttd_seconds(overlay, detections, per='flow'|'campaign'), detections = [inject_id, detected_at].")
    w("Worked example (HYPOTHETICAL detection time, to show the arithmetic):")
    w(f"  inject_id {ex['inject_id'].iloc[0]} has inject_timestamp {ex['inject_timestamp'].iloc[0].isoformat()}.")
    w(f"  Suppose the first alert for it is at {detect.isoformat()}.")
    w(f"  MTTD_flow = {detect.isoformat()} - {ex['inject_timestamp'].iloc[0].isoformat()} = 7.5 s.")
    w("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    import os
    cols = feature_columns()
    benign = pd.read_parquet(FEATURES_TEST, columns=["Label"])["Label"]
    n_benign = int((benign == "BENIGN").sum())
    scaler = fit_real_scaler(cols)
    err = verify_scaler(scaler)
    assert err < 1e-6, f"scaler reconstruction does not match pipeline: {err}"
    overlay = generate_overlay(n_benign, scaler, cols)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    overlay.to_parquet(OUT_PARQUET, index=False)
    write_report(overlay, n_benign, err)
    print(f"wrote {len(overlay):,} rows ({len(overlay)/(n_benign+len(overlay)):.4%}) to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
