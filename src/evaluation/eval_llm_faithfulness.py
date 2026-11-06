"""M2 measurement for the Thai LLM explanation layer: entity preservation, technique grounding,
and a mechanical (partial) claim-level support check, on alerts sampled from real test data.

Run from the repo root:  uv run python -m src.evaluation.eval_llm_faithfulness
Flags: --limit N (smoke run), --rescore (recompute checks and reports from the raw JSONL, no LLM),
       --rebuild-alerts (regenerate the sampled alert set).

Two interpreters are needed. Scoring the alerts (RF, AE, IF, fusion) and RAG retrieval need
pandas/torch/sentence-transformers, which only the Anaconda interpreter has. The LLM call needs
langchain, which only the uv .venv has. The uv entry point therefore shells out to Anaconda once for
`--stage alerts`, then runs the LLM stage itself. Everything the scoring functions need is stdlib, so
tests import this module under uv without pandas.

POPULATION: the unified one. This used to sample from the behavioral test matrix, score with
random_forest.pkl (80 features) plus the behavioral_v2 IF/AE (92 features), and reconcile the
two through a row-hash join -- the arrangement the unified population exists to remove. It also
recorded `model_used` as "autoencoder_v2", a value fusion_layer no longer emits, so every M2
figure produced that way described a pipeline that is not deployed.

Identifiers are the one thing the unified feature matrix does not carry: build_unified_population
drops Source IP, Destination IP and Timestamp from model input, because Source IP in particular
would let a model memorise which hosts attacked in this capture. They are recovered by
re-deriving the same split from the same source with the same seed and indexing it POSITIONALLY,
which is valid because the unified val/test splits get no SMOTE and are written in split order
with the index reset. The alignment is asserted on Label, not assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "data" / "knowledge_base" / "mitre_techniques.json"
ALERTS_PATH = ROOT / "reports" / "llm_faithfulness_alerts.jsonl"
RAW_PATH = ROOT / "reports" / "llm_faithfulness_raw.jsonl"
METRICS_PATH = ROOT / "reports" / "llm_faithfulness_metrics.csv"
REPORT_PATH = ROOT / "reports" / "llm_faithfulness_report.txt"
ML_PYTHON = os.environ.get("ML_PYTHON", "C:/Users/devMrX/anaconda3/python.exe")

SEED = 42
PER_CATEGORY_CAP = 7
BENIGN_N = 10
UNIFIED_TEST = ROOT / "data/features/unified/cicids2017_unified_test.parquet"

# ----------------------------------------------------------------------------
# Pure scoring functions (stdlib only)
# ----------------------------------------------------------------------------
IPV4_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
TECH_RE = re.compile(r"(?<![A-Za-z0-9])T[0-9]{4}(?:\.[0-9]{3})?(?![A-Za-z0-9])")
HOST_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9]*-[A-Z0-9][A-Z0-9-]*|[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]+)+)(?![A-Za-z0-9])"
)
PORT_MENTION_RE = re.compile(r"(?:พอร์ต|port)\s*[:#]?\s*([0-9]{1,5})(?![0-9.])", re.IGNORECASE)
IP_PORT_RE = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}:([0-9]{1,5})(?![0-9])")
THAI_DIGIT_RE = re.compile(r"[๐-๙]")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&+_./-]*")
CONF_RE = re.compile(r"ความมั่นใจ[^0-9\n]{0,15}([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)(%?)")
SCI_NUM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+")
PRIORITY_RE = re.compile(r"(?<![A-Za-z])(HIGH|MEDIUM|LOW)(?![A-Za-z])")
CLAIM_SPLIT_RE = re.compile(r"(?<=[฀-๿])\s+(?=[฀-๿])|[\n;]|(?<=[A-Za-z0-9)])\.(?:\s+|$)")

# Not product/service claims; everything else Latin must appear in the alert or retrieved chunks.
GENERIC_LATIN = {"log", "logs", "ip", "soc", "rag", "mitre", "att&ck", "tcp", "udp", "icmp"}
PORT_SERVICES = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 80: "http", 110: "pop3", 139: "smb",
    143: "imap", 389: "ldap", 443: "https", 445: "smb", 636: "ldaps", 1433: "mssql", 3306: "mysql",
    3389: "rdp", 8080: "http", 8443: "https",
}
SERVICE_ALIASES = {
    "waf": ["web application firewall"], "smb": ["server message block"], "ssh": ["secure shell"],
    "rdp": ["remote desktop"], "ftp": ["file transfer protocol"], "dns": ["domain name system"],
    "mfa": ["multi-factor", "multifactor", "two-factor"], "2fa": ["multi-factor", "two-factor"],
    "ids": ["intrusion detection"], "ips": ["intrusion prevention"], "vpn": ["virtual private network"],
    "acl": ["access control list"], "ddos": ["denial of service"], "dos": ["denial of service"],
    "http": ["https"], "https": ["http"],
}
WEB_PORTS = {80, 443, 8000, 8008, 8080, 8443, 8888}

ATTACK_GROUPS = {
    "scan": re.compile(r"scan|สแกน|สำรวจ|reconnaissance", re.I),
    "bruteforce": re.compile(r"brute|บรูท|เดารหัส|เดาสุ่ม|patator|credential stuffing|password guess|ล็อกอินซ้ำ", re.I),
    "dos": re.compile(r"(?<![a-z])d?dos(?![a-z])|flood|ฟลัด|ปฏิเสธการให้บริการ|denial of service|slowloris|slowhttp", re.I),
    "webattack": re.compile(r"sql injection|(?<![a-z])xss(?![a-z])|cross-site|web attack|เว็บแอป", re.I),
    "botc2": re.compile(r"botnet|บอตเน็ต|command and control|(?<![a-z])c2(?![a-z0-9])|(?<![a-z])bot(?![a-z])", re.I),
    "heartbleed": re.compile(r"heartbleed", re.I),
    "infiltration": re.compile(r"infiltration|ฝังตัว", re.I),
}
LABEL_GROUPS = {
    "PortScan": {"scan"}, "FTP-Patator": {"bruteforce"}, "SSH-Patator": {"bruteforce"},
    "Web Attack - Brute Force": {"bruteforce", "webattack"}, "Web Attack - XSS": {"webattack"},
    "Web Attack - Sql Injection": {"webattack"}, "Bot": {"botc2"}, "Heartbleed": {"heartbleed"},
    "Infiltration": {"infiltration"}, "DDoS": {"dos"}, "BENIGN": set(),
}
INSUFFICIENT_RE = re.compile(
    r"(หลักฐาน|RAG)[^\n]{0,40}(ไม่พอ|ไม่เพียงพอ|ไม่สอดคล้อง|ไม่ชัดเจน|ไม่ครบ)|ไม่สามารถระบุ(?:เทคนิค|technique)",
    re.IGNORECASE,
)
SEVERE_RE = re.compile(r"วิกฤต|ฉุกเฉิน|รุนแรงมาก")
# The few-shot examples are now RENDERED with each alert's own values (see
# llm_explain.render_few_shots), so the old check for hardcoded example entities
# (172.16.33.9, LMS-WEB03, ...) can only ever report zero -- those strings are gone from the
# prompt. What replaced that leak is a different risk: prose that names the right host but
# reuses the example's sentence. EXAMPLE_ECHO_THRESHOLD is the similarity above which an
# explanation counts as echoing rather than describing. 0.80 is a judgement call, so the raw
# ratio is recorded per alert and the rate can be recomputed at any cutoff from the JSONL.
EXAMPLE_ECHO_THRESHOLD = 0.80
WAF_RE = re.compile(r"\bWAF\b|web application firewall", re.IGNORECASE)


def label_groups(label: str) -> set[str]:
    if label in LABEL_GROUPS:
        return LABEL_GROUPS[label]
    if label.startswith("DoS"):
        return {"dos"}
    return set()


def load_kb(path: Path = KB_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["techniques"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def split_claims(text: str) -> list[str]:
    parts = [p.strip(" -•*\t") for p in CLAIM_SPLIT_RE.split(text or "")]
    return [p for p in parts if len(p) >= 4]


def context_entities(ctx: dict[str, Any], technique_id: str | None) -> list[dict[str, str]]:
    entities = []
    for key in ("source_ip", "dst_ip"):
        if ctx.get(key):
            entities.append({"type": "ip", "value": str(ctx[key])})
    if ctx.get("dst_port") is not None:
        entities.append({"type": "port", "value": str(ctx["dst_port"])})
    asset = ctx.get("affected_asset")
    if asset and not IPV4_RE.fullmatch(str(asset)):
        entities.append({"type": "hostname", "value": str(asset)})
    if technique_id:
        entities.append({"type": "technique_id", "value": technique_id})
    return entities


def _present(value: str, text: str, etype: str) -> bool:
    if etype == "port":
        return re.search(r"(?<![0-9.])" + re.escape(value) + r"(?![0-9])", text) is not None
    if etype == "ip":
        return value in IPV4_RE.findall(text)
    if etype == "technique_id":
        return value in TECH_RE.findall(text)
    return value in text


def check_entities(rec: dict[str, Any]) -> dict[str, Any]:
    ctx = rec["alert_context"]
    out = rec["output"]
    text = out["explanation_th"] + "\n" + "\n".join(out["remediation"])
    required = context_entities(ctx, rec.get("ctx_technique_id"))
    for ent in required:
        ent["present"] = _present(ent["value"], text, ent["type"])
    missing = [f'{e["type"]}:{e["value"]}' for e in required if not e["present"]]

    known_ips = {str(ctx[k]) for k in ("source_ip", "dst_ip") if ctx.get(k)}
    known_hosts = {e["value"] for e in required if e["type"] == "hostname"}
    hit_text = _hit_text(rec)
    foreign = []
    for ip in IPV4_RE.findall(text):
        if ip not in known_ips and ip not in hit_text:
            foreign.append(f"ip:{ip}")
    dst_port = str(ctx.get("dst_port"))
    for port in PORT_MENTION_RE.findall(text) + IP_PORT_RE.findall(text):
        if port != dst_port:
            foreign.append(f"port:{port}")
    for tid in TECH_RE.findall(text):
        if tid != rec.get("ctx_technique_id"):
            foreign.append(f"technique_id:{tid}")
    stripped = IPV4_RE.sub(" ", text)
    for host in HOST_RE.findall(stripped):
        if TECH_RE.fullmatch(host):
            continue
        if host in known_hosts or any(host in h for h in known_hosts) or host.lower() in hit_text:
            continue
        foreign.append(f"hostname:{host}")
    thai_digits = THAI_DIGIT_RE.search(text) is not None
    return {
        "required": required,
        "missing": missing,
        "foreign": sorted(set(foreign)),
        "thai_digits": thai_digits,
        "pass": not missing and not foreign and not thai_digits,
        "corruption_free": not foreign and not thai_digits,
    }


def _hit_text(rec: dict[str, Any]) -> str:
    parts = []
    for hit in rec.get("rag_hits", []):
        meta = hit.get("metadata", {})
        parts += [hit.get("document", ""), meta.get("technique_name", ""), meta.get("mitigation_name", "")]
    return "\n".join(parts).lower()


def _hit_technique_ids(rec: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for hit in rec.get("rag_hits", []):
        meta = hit.get("metadata", {})
        if meta.get("technique_id"):
            ids.add(meta["technique_id"])
        ids.update(t for t in str(meta.get("technique_ids", "")).split(",") if t)
    return ids


def check_grounding(rec: dict[str, Any], kb: dict[str, Any]) -> dict[str, Any]:
    out = rec["output"]
    hit_ids = _hit_technique_ids(rec)
    tid, tactic = out.get("technique_id"), out.get("mitre_tactic")
    in_hits = tactic_ok = None
    if tid:
        in_hits = tid in hit_ids
        kb_tactics = (kb.get(tid) or {}).get("tactics") or []
        tactic_ok = bool(kb_tactics) and tactic == kb_tactics[0] and tactic == tactic.lower()
    prose = out["explanation_th"] + "\n" + "\n".join(out["remediation"])
    prose_ids = sorted(set(TECH_RE.findall(prose)))
    ungrounded = [t for t in prose_ids if t not in hit_ids]
    mapped = rec.get("label_mapping_technique_id")
    return {
        "result_technique_id": tid,
        "result_tactic": tactic,
        "technique_in_hits": in_hits,
        "tactic_matches_kb": tactic_ok,
        "prose_technique_ids": prose_ids,
        "prose_ungrounded": ungrounded,
        "matches_label_mapping": (tid == mapped) if (tid and mapped) else None,
    }


def _context_hay(rec: dict[str, Any]) -> str:
    ctx = rec["alert_context"]
    services = [PORT_SERVICES[int(ctx["dst_port"])]] if str(ctx.get("dst_port", "")).isdigit() and int(ctx["dst_port"]) in PORT_SERVICES else []
    return (json.dumps(ctx, ensure_ascii=False) + " " + " ".join(services)).lower() + "\n" + _hit_text(rec)


def _latin_supported(token: str, hay: str) -> bool:
    candidates = [token.lower()] + SERVICE_ALIASES.get(token.lower(), [])
    return any(re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", hay) for c in candidates)


def classify_claim(text: str, source: str, rec: dict[str, Any]) -> dict[str, Any]:
    """supported / unsupported / contradicted / undecidable. 'supported' means every mechanically
    checkable atom in the claim is grounded in the alert or retrieved chunks. It does NOT mean the
    claim as a whole is true; a claim with no checkable atom is 'undecidable', never 'supported'."""
    ctx = rec["alert_context"]
    label = ctx.get("rf_predicted_label", "")
    is_expl = source == "explanation"
    hay = _context_hay(rec)
    hit_ids = _hit_technique_ids(rec)
    chosen = rec.get("ctx_technique_id")
    contradicted: list[str] = []
    unsupported: list[str] = []
    atoms = 0

    known_ips = {str(ctx[k]) for k in ("source_ip", "dst_ip") if ctx.get(k)}
    for ip in IPV4_RE.findall(text):
        atoms += 1
        if ip not in known_ips and ip not in hay:
            (contradicted if is_expl else unsupported).append(f"foreign_ip:{ip}")
    dst_port = str(ctx.get("dst_port"))
    for port in PORT_MENTION_RE.findall(text) + IP_PORT_RE.findall(text):
        atoms += 1
        if port != dst_port:
            (contradicted if is_expl else unsupported).append(f"foreign_port:{port}")
    for tid in TECH_RE.findall(text):
        atoms += 1
        if tid == chosen:
            continue
        if tid not in hit_ids:
            unsupported.append(f"technique_not_retrieved:{tid}")
        elif chosen:
            contradicted.append(f"technique_differs_from_consensus:{tid}")
        else:
            unsupported.append(f"technique_named_despite_abstention:{tid}")
    for m in CONF_RE.finditer(text):
        atoms += 1
        value = float(m.group(1)) / (100 if m.group(2) else 1)
        if abs(value - float(rec["confidence"])) > 0.005:
            contradicted.append(f"confidence:{m.group(1)}{m.group(2)}")
    for lvl in PRIORITY_RE.findall(text):
        atoms += 1
        if lvl != rec["priority"]:
            contradicted.append(f"priority:{lvl}")
    if rec["priority"] == "LOW" and SEVERE_RE.search(text):
        atoms += 1
        contradicted.append("severity_vs_low_priority")
    allowed = label_groups(label)
    for group, pattern in ATTACK_GROUPS.items():
        if pattern.search(text):
            atoms += 1
            if group not in allowed:
                (contradicted if (is_expl and label != "BENIGN") else unsupported).append(f"attack_type:{group}")
    bare = SCI_NUM_RE.sub(" ", THAI_DIGIT_RE.sub(" ", TECH_RE.sub(" ", IPV4_RE.sub(" ", text))))
    for host in HOST_RE.findall(bare):
        bare = bare.replace(host, " ")
        atoms += 1
        if host.lower() not in hay:
            unsupported.append(f"foreign_hostname:{host}")
    for token in LATIN_RE.findall(bare):
        token = token.strip(".-/")
        low = token.lower()
        if len(low) < 2 or low in GENERIC_LATIN or low in {"high", "medium", "low"}:
            continue
        if re.fullmatch(r"[0-9.]+", low):
            continue
        atoms += 1
        if not _latin_supported(token, hay):
            unsupported.append(f"ungrounded_term:{token}")
    if contradicted:
        verdict = "contradicted"
    elif unsupported:
        verdict = "unsupported"
    elif atoms:
        verdict = "supported"
    else:
        verdict = "undecidable"
    return {"source": source, "text": text, "verdict": verdict, "reasons": contradicted + unsupported, "atoms": atoms}


def check_claims(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out = rec["output"]
    claims = [classify_claim(c, "explanation", rec) for c in split_claims(out["explanation_th"])]
    claims += [classify_claim(r, f"remediation[{i}]", rec) for i, r in enumerate(out["remediation"])]
    return claims


def _prose(out):
    return "\n".join([out["explanation_th"], *out["remediation"]])


def example_echo_ratio(rec: dict[str, Any]) -> float:
    """Highest similarity between the output's explanation and any rendered example's.

    difflib, not an embedding model: this runs in the stdlib-only scoring path that tests
    import under uv without pandas, and verbatim reuse is a surface-form question anyway.
    Returns 0.0 when the rendered examples are unavailable, which is reported as unmeasured
    rather than as a pass.
    """
    import difflib

    shots = rec.get("rendered_example_explanations") or []
    text = (rec.get("output") or {}).get("explanation_th") or ""
    if not shots or not text:
        return 0.0
    return max(difflib.SequenceMatcher(None, text, shot).ratio() for shot in shots)


def check_defects(rec: dict[str, Any]) -> dict[str, Any]:
    out = rec["output"]
    port = rec["alert_context"].get("dst_port")
    waf = any(WAF_RE.search(r) for r in out["remediation"])
    waf_nonweb = waf and port not in WEB_PORTS and not rec["alert_context"].get("rf_predicted_label", "").startswith("Web Attack")
    return {
        "waf_in_remediation": waf,
        "waf_nonweb_context": waf_nonweb,
        "technique_null": out.get("technique_id") is None,
        "example_echo_ratio": round(example_echo_ratio(rec), 4),
        "example_echo": example_echo_ratio(rec) >= EXAMPLE_ECHO_THRESHOLD,
        "states_insufficient_evidence": INSUFFICIENT_RE.search(out["explanation_th"]) is not None,
    }


def score_record(rec: dict[str, Any], kb: dict[str, Any] | None = None) -> dict[str, Any]:
    kb = kb if kb is not None else load_kb()
    return {
        "entity": check_entities(rec),
        "grounding": check_grounding(rec, kb),
        "claims": check_claims(rec),
        "defects": check_defects(rec),
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"num": k, "den": n, "rate": (k / n) if n else float("nan"), "ci_lo": lo, "ci_hi": hi}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    s: dict[str, Any] = {"n_total": len(records), "n_ok": len(ok), "n_error": len(records) - len(ok)}

    def count(pred, population=None):
        pop = ok if population is None else population
        return sum(1 for r in pop if pred(r))

    s["entity_pass"] = _rate(count(lambda r: r["checks"]["entity"]["pass"]), len(ok))
    s["entity_corruption_free"] = _rate(count(lambda r: r["checks"]["entity"]["corruption_free"]), len(ok))
    s["entity_complete"] = _rate(count(lambda r: not r["checks"]["entity"]["missing"]), len(ok))
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    foreign_by_type: Counter = Counter()
    for r in ok:
        for ent in r["checks"]["entity"]["required"]:
            by_type[ent["type"]][1] += 1
            by_type[ent["type"]][0] += int(ent["present"])
        for f in r["checks"]["entity"]["foreign"]:
            foreign_by_type[f.split(":")[0]] += 1
    s["entity_presence_by_type"] = {t: _rate(*v) for t, v in sorted(by_type.items())}
    s["foreign_entity_mentions_by_type"] = dict(foreign_by_type)
    s["thai_digit_alerts"] = count(lambda r: r["checks"]["entity"]["thai_digits"])

    nonnull = [r for r in ok if r["checks"]["grounding"]["result_technique_id"]]
    s["abstained"] = _rate(len(ok) - len(nonnull), len(ok))
    s["result_technique_in_hits"] = _rate(count(lambda r: r["checks"]["grounding"]["technique_in_hits"], nonnull), len(nonnull))
    s["result_tactic_matches_kb"] = _rate(count(lambda r: r["checks"]["grounding"]["tactic_matches_kb"], nonnull), len(nonnull))
    s["prose_ungrounded_alerts"] = _rate(count(lambda r: r["checks"]["grounding"]["prose_ungrounded"]), len(ok))
    s["prose_technique_mention_alerts"] = _rate(count(lambda r: r["checks"]["grounding"]["prose_technique_ids"]), len(ok))
    attack_nonnull = [r for r in nonnull if r["true_label"] != "BENIGN" and r["checks"]["grounding"]["matches_label_mapping"] is not None]
    s["consensus_matches_label_mapping"] = _rate(count(lambda r: r["checks"]["grounding"]["matches_label_mapping"], attack_nonnull), len(attack_nonnull))

    verdicts: Counter = Counter()
    by_src: dict[str, Counter] = {"explanation": Counter(), "remediation": Counter()}
    reasons: Counter = Counter()
    for r in ok:
        for c in r["checks"]["claims"]:
            verdicts[c["verdict"]] += 1
            by_src[c["source"].split("[")[0]][c["verdict"]] += 1
            for reason in c["reasons"]:
                reasons[reason.split(":")[0]] += 1
    total = sum(verdicts.values())
    decidable = total - verdicts["undecidable"]
    bad = verdicts["unsupported"] + verdicts["contradicted"]
    s["claims"] = {
        "total": total, "verdicts": dict(verdicts),
        "hallucination_lower_bound": _rate(bad, total),
        "hallucination_among_decidable": _rate(bad, decidable),
        "undecidable_share": _rate(verdicts["undecidable"], total),
        "by_source": {k: dict(v) for k, v in by_src.items()},
        "reason_counts": dict(reasons),
    }
    s["alerts_with_bad_claim"] = _rate(count(lambda r: any(c["verdict"] in ("unsupported", "contradicted") for c in r["checks"]["claims"])), len(ok))

    ps = [r for r in ok if r["alert_context"]["rf_predicted_label"] == "PortScan"]
    ps_true = [r for r in ok if r["true_label"] == "PortScan"]
    ssh_true = [r for r in ok if r["true_label"] == "SSH-Patator"]
    null_tech = [r for r in ok if r["checks"]["defects"]["technique_null"]]
    ssh_null = [r for r in ssh_true if r["checks"]["defects"]["technique_null"]]
    s["defect_waf_portscan_pred"] = _rate(count(lambda r: r["checks"]["defects"]["waf_in_remediation"], ps), len(ps))
    s["defect_waf_portscan_true"] = _rate(count(lambda r: r["checks"]["defects"]["waf_in_remediation"], ps_true), len(ps_true))
    s["defect_waf_portscan_true_smb"] = _rate(count(lambda r: r["checks"]["defects"]["waf_in_remediation"] and r["alert_context"]["dst_port"] in (139, 445), ps_true), count(lambda r: r["alert_context"]["dst_port"] in (139, 445), ps_true))
    s["example_echo_alerts"] = _rate(count(lambda r: r["checks"]["defects"].get("example_echo")), len(ok))
    _ratios = [r["checks"]["defects"].get("example_echo_ratio", 0.0) for r in ok]
    s["example_echo_ratio_max"] = round(max(_ratios), 4) if _ratios else 0.0
    s["example_echo_ratio_mean"] = round(sum(_ratios) / len(_ratios), 4) if _ratios else 0.0
    s["waf_nonweb_all"] = _rate(count(lambda r: r["checks"]["defects"]["waf_nonweb_context"]), len(ok))
    s["defect_insufficient_null_all"] = _rate(count(lambda r: r["checks"]["defects"]["states_insufficient_evidence"], null_tech), len(null_tech))
    s["defect_insufficient_ssh_true"] = _rate(count(lambda r: r["checks"]["defects"]["states_insufficient_evidence"], ssh_true), len(ssh_true))
    s["defect_insufficient_ssh_null"] = _rate(count(lambda r: r["checks"]["defects"]["states_insufficient_evidence"], ssh_null), len(ssh_null))
    s["ssh_true_technique_null"] = _rate(len(ssh_null), len(ssh_true))

    cats = sorted({r["true_label"] for r in records})
    per_cat = {}
    for cat in cats:
        sub = [r for r in ok if r["true_label"] == cat]
        cl = [c for r in sub for c in r["checks"]["claims"]]
        per_cat[cat] = {
            "n_sampled": sum(1 for r in records if r["true_label"] == cat),
            "n_ok": len(sub),
            "entity_pass": sum(r["checks"]["entity"]["pass"] for r in sub),
            "entity_corruption_free": sum(r["checks"]["entity"]["corruption_free"] for r in sub),
            "abstained": sum(r["checks"]["defects"]["technique_null"] for r in sub),
            "prose_ungrounded": sum(bool(r["checks"]["grounding"]["prose_ungrounded"]) for r in sub),
            "claims": len(cl),
            "claims_unsupported": sum(c["verdict"] == "unsupported" for c in cl),
            "claims_contradicted": sum(c["verdict"] == "contradicted" for c in cl),
            "claims_undecidable": sum(c["verdict"] == "undecidable" for c in cl),
        }
    s["per_category"] = per_cat
    s["priority_counts"] = dict(Counter(r["priority"] for r in ok))

    m = [r["llm_metrics"] for r in ok if r.get("llm_metrics", {}).get("gen_tok_s")]
    walls = [r["llm_metrics"]["wall_s"] for r in ok if r.get("llm_metrics")]
    s["throughput"] = {
        "n_with_metrics": len(m),
        "gen_tokens_total": sum(x["gen_tokens"] for x in m),
        "gen_seconds_total": sum(x["gen_s"] for x in m),
        "gen_tok_s_pooled": (sum(x["gen_tokens"] for x in m) / sum(x["gen_s"] for x in m)) if m else float("nan"),
        "gen_tok_s_median": sorted(x["gen_tok_s"] for x in m)[len(m) // 2] if m else float("nan"),
        "gen_tok_s_min": min((x["gen_tok_s"] for x in m), default=float("nan")),
        "gen_tok_s_max": max((x["gen_tok_s"] for x in m), default=float("nan")),
        "prompt_tokens_mean": (sum(x["prompt_tokens"] for x in m) / len(m)) if m else float("nan"),
        "wall_s_total_llm": sum(walls),
        "wall_s_mean": (sum(walls) / len(walls)) if walls else float("nan"),
        "wall_s_first": walls[0] if walls else float("nan"),
    }
    return s


def _fmt(r: dict[str, Any]) -> str:
    return f'{r["num"]}/{r["den"]} = {r["rate"]*100:.1f}% (95% Wilson CI {r["ci_lo"]*100:.1f}-{r["ci_hi"]*100:.1f}%)'


def write_metrics_csv(s: dict[str, Any], path: Path = METRICS_PATH) -> None:
    rows: list[list[Any]] = []
    rows.append(["# aggregate rates (each row recomputable from llm_faithfulness_raw.jsonl; see report section 8)"])
    rows.append(["metric", "numerator", "denominator", "rate", "ci95_lo", "ci95_hi", "basis"])
    basis = {
        "entity_pass": "measured: no omitted and no foreign/altered entity (strict)",
        "entity_corruption_free": "measured: no foreign/altered IP, port, technique id, hostname; no Thai digits",
        "entity_complete": "measured: every context IP/port/hostname/technique id appears verbatim",
        "abstained": "measured: technique_id null (fewer than 2 of top-3 RAG hits agree)",
        "result_technique_in_hits": "measured (by construction near-certain: technique_id is code-derived from hits)",
        "result_tactic_matches_kb": "measured (by construction near-certain: tactic is code-derived from KB)",
        "prose_ungrounded_alerts": "measured: alert prose names a technique id absent from retrieved hits",
        "prose_technique_mention_alerts": "measured: alert prose names any technique id",
        "consensus_matches_label_mapping": "measured, informational: RAG consensus technique equals MITRE mapping of true label (retrieval accuracy, not hallucination)",
        "alerts_with_bad_claim": "approximated: alert has >=1 unsupported/contradicted claim by mechanical rules",
        "defect_waf_portscan_pred": "measured: WAF in remediation, alerts RF-predicted PortScan",
        "defect_waf_portscan_true": "measured: WAF in remediation, alerts truly PortScan",
        "defect_waf_portscan_true_smb": "measured: WAF in remediation, truly PortScan with dst_port 139/445",
        "example_echo_alerts": f"measured: explanation is >= {EXAMPLE_ECHO_THRESHOLD} similar to a rendered few-shot example (difflib ratio)",
        "waf_nonweb_all": "measured: WAF in remediation when dst_port is not a web port and label not Web Attack",
        "defect_insufficient_null_all": "approximated (keyword regex): abstained alerts whose explanation states RAG evidence insufficient",
        "defect_insufficient_ssh_true": "approximated (keyword regex): truly SSH-Patator alerts stating evidence insufficient",
        "defect_insufficient_ssh_null": "approximated (keyword regex): truly SSH-Patator alerts that abstained and state insufficiency",
        "ssh_true_technique_null": "measured: truly SSH-Patator alerts where the technique abstained",
    }
    for key, why in basis.items():
        r = s[key]
        rows.append([key, r["num"], r["den"], f'{r["rate"]:.6f}', f'{r["ci_lo"]:.6f}', f'{r["ci_hi"]:.6f}', why])
    # Plain scalars, not proportions -- they have no numerator/denominator or CI, so they get
    # their own rows rather than being forced into the rate columns.
    for key, why in (
        ("example_echo_ratio_max", "measured scalar: highest single-alert difflib similarity to a rendered example"),
        ("example_echo_ratio_mean", "measured scalar: mean difflib similarity to a rendered example"),
    ):
        rows.append([key, "", "", f"{s[key]:.6f}", "", "", why])
    c = s["claims"]
    for key, why in (
        ("hallucination_lower_bound", "approximated: (unsupported+contradicted)/all claims; undecidable claims counted as not hallucinated, so this is a LOWER bound"),
        ("hallucination_among_decidable", "approximated: (unsupported+contradicted)/(supported+unsupported+contradicted)"),
        ("undecidable_share", "unmeasured share: claims with no mechanically checkable atom; need human raters"),
    ):
        r = c[key]
        rows.append([key, r["num"], r["den"], f'{r["rate"]:.6f}', f'{r["ci_lo"]:.6f}', f'{r["ci_hi"]:.6f}', why])
    rows.append([])
    rows.append(["# entity presence by type (measured)"])
    rows.append(["entity_type", "present", "required", "rate"])
    for t, r in s["entity_presence_by_type"].items():
        rows.append([t, r["num"], r["den"], f'{r["rate"]:.6f}'])
    rows.append([])
    rows.append(["# claim verdict counts (approximated; claim split heuristic documented in report section 4)"])
    rows.append(["source", "supported", "unsupported", "contradicted", "undecidable"])
    for src, v in c["by_source"].items():
        rows.append([src, v.get("supported", 0), v.get("unsupported", 0), v.get("contradicted", 0), v.get("undecidable", 0)])
    rows.append([])
    rows.append(["# claim reason counts (rule that fired; one claim may fire several)"])
    rows.append(["reason", "count"])
    for k, v in sorted(c["reason_counts"].items()):
        rows.append([k, v])
    rows.append([])
    rows.append(["# per true category (alerts sampled from the CICIDS-2017 behavioral test split)"])
    rows.append(["category", "n_sampled", "n_ok", "entity_pass", "entity_corruption_free", "abstained", "prose_ungrounded_alerts", "claims", "claims_unsupported", "claims_contradicted", "claims_undecidable"])
    for cat, v in s["per_category"].items():
        rows.append([cat, v["n_sampled"], v["n_ok"], v["entity_pass"], v["entity_corruption_free"], v["abstained"], v["prose_ungrounded"], v["claims"], v["claims_unsupported"], v["claims_contradicted"], v["claims_undecidable"]])
    rows.append([])
    rows.append(["# throughput (measured from Ollama eval_count/eval_duration; first call may include model load)"])
    rows.append(["metric", "value"])
    for k, v in s["throughput"].items():
        rows.append([k, f"{v:.4f}" if isinstance(v, float) else v])
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def write_report(s: dict[str, Any], records: list[dict[str, Any]], path: Path = REPORT_PATH, wall_total_s: float | None = None) -> None:
    ok = [r for r in records if r.get("status") == "ok"]
    L: list[str] = []
    bar = "=" * 70
    L += [bar, "LLM LAYER -- M2 FAITHFULNESS / HALLUCINATION MEASUREMENT", bar,
          "Model: scb10x/llama3.1-typhoon2-8b-instruct via Ollama, deployed path src/llm/llm_explain.explain_alert.",
          "Module: src/evaluation/eval_llm_faithfulness.py. Evidence: reports/llm_faithfulness_raw.jsonl.", ""]
    L += [bar, "1. SAMPLE", bar]
    L += [f"Alerts sampled: {s['n_total']}; LLM calls that returned a valid explanation: {s['n_ok']}; failed: {s['n_error']}.",
          f"Source: data/features/behavioral/cicids2017_behavioral_features_test.parquet (405,142 rows), seed {SEED}.",
          f"Stratification: up to {PER_CATEGORY_CAP} random alerts per attack category (all rows if fewer exist: Infiltration 7, "
          f"Sql Injection 4, Heartbleed 2), plus {BENIGN_N} random BENIGN rows. Counts per true category:"]
    for cat, v in s["per_category"].items():
        L.append(f"  {cat:<28} sampled={v['n_sampled']:>3} ok={v['n_ok']:>3}")
    L += ["Why about 100: the budget is 10-20 minutes of generation at about 65 tok/s, and a flat 100 with a cap of 7 per",
          "category keeps DoS Hulk and DDoS (60% of the test attacks) at 7% each. The cost is that per-category rates rest on 2-7",
          "alerts and are descriptive only. Only aggregate rates carry usable confidence intervals, and even those treat alerts as",
          "independent draws from a category-balanced mix, not from the natural traffic mix. Rates are therefore NOT the rates a",
          "campus SOC would see; they describe model behavior across attack types.",
          f"Priority mix of the explained alerts: {s['priority_counts']}.",
          "Alert context is real: RF/AE/IF scores from the deployed models, fusion_layer.fuse_with_metadata for fused_score and",
          "model_used, IPs/ports/timestamps recovered for each sampled row by row-hash join to the raw behavioral file,",
          "RagRetriever.retrieve (top_k=3) for the hits. Limits of that context: CICIDS-2017 has no hostnames, so affected_asset",
          "falls back to dst_ip and the hostname check never fires on real data (it is unit-tested only). The RAG query is the RF",
          "predicted label (the same query the RAG precision evaluation uses); no production query builder exists in the repo.",
          "For rows RF predicts BENIGN the query is a generic anomaly sentence. The true label is never shown to the LLM.", ""]
    L += [bar, "2. ENTITY PRESERVATION (measured, exact)", bar]
    L += ["Required entities: source IP, destination IP, destination port, hostname (none in this data), and the RAG-consensus",
          "technique_id when one exists. 'Pass' needs every required entity verbatim in explanation_th + remediation AND no",
          "foreign or altered IP, port ('พอร์ต N'/'port N'/'ip:N'), technique id or hostname, AND no Thai-numeral digits.",
          f"Strict pass:            {_fmt(s['entity_pass'])}",
          f"Complete (no omission): {_fmt(s['entity_complete'])}",
          f"Corruption-free (no foreign/altered entity, no Thai digits): {_fmt(s['entity_corruption_free'])}",
          "Presence by entity type:"]
    for t, r in s["entity_presence_by_type"].items():
        L.append(f"  {t:<13} {_fmt(r)}")
    L.append(f"Foreign entity mentions by type (count of mentions): {s['foreign_entity_mentions_by_type']}; alerts with Thai digits: {s['thai_digit_alerts']}.")
    L.append("Violating alerts (omitted = required but absent; foreign = present but not in the alert context):")
    for r in ok:
        e = r["checks"]["entity"]
        if not e["pass"]:
            L.append(f"  {r['alert_id']} [{r['true_label']}] omitted={e['missing']} foreign={e['foreign']} thai_digits={e['thai_digits']}")
    L.append("")
    L += [bar, "3. TECHNIQUE GROUNDING (measured)", bar]
    L += ["The Typhoon draft schema has no technique_id or tactic field. explain_alert fills both from the RAG top-3 consensus and the",
          "KB, so the result-level fields cannot be hallucinated by the model by construction. Those rates are reported as a",
          "regression guard, not a model measurement. The model-dependent question is whether its PROSE names technique ids.",
          f"Abstained (technique_id null):                 {_fmt(s['abstained'])}",
          f"Result technique_id present in retrieved hits: {_fmt(s['result_technique_in_hits'])}  (guard)",
          f"Result mitre_tactic equals KB lowercase tactic: {_fmt(s['result_tactic_matches_kb'])}  (guard)",
          f"Alerts whose prose names any technique id:     {_fmt(s['prose_technique_mention_alerts'])}",
          f"Alerts whose prose names an id NOT retrieved:   {_fmt(s['prose_ungrounded_alerts'])}   <- technique hallucination rate",
          f"Informational, not hallucination: RAG consensus technique equals MITRE mapping of the true label: {_fmt(s['consensus_matches_label_mapping'])}",
          ""]
    L += [bar, "4. CLAIM-LEVEL SUPPORT (approximated, partial)", bar]
    c = s["claims"]
    L += ["Segmentation: explanation_th is split at newlines, ';', a full stop after a Latin letter/digit/')' followed by space or end,",
          "and whitespace between two Thai characters. Each remediation item is one claim. This is a heuristic; Thai has no sentence",
          "punctuation, so some claims are over- or under-split.",
          "Classification is rule-based on checkable atoms only. contradicted: stated port, IP, confidence number, priority level or",
          "attack type conflicts with the alert (explanation), or a different retrieved technique than the consensus. unsupported: names",
          "an IP, port, technique, hostname or Latin-script service/product term found nowhere in the alert or retrieved chunk text",
          "(alias and port-to-service tables in the source; 'SMB' is grounded by dst_port 445/139). supported: it has >=1 checkable",
          "atom and all are grounded. undecidable: no checkable atom (most Thai prose and generic advice).",
          f"Claims total: {c['total']}; verdicts: {c['verdicts']}",
          f"Hallucination LOWER bound (unsupported+contradicted)/all claims: {_fmt(c['hallucination_lower_bound'])}",
          f"Among mechanically decidable claims only:                       {_fmt(c['hallucination_among_decidable'])}",
          f"Undecidable share (not judged either way):                       {_fmt(c['undecidable_share'])}",
          f"Alerts with at least one unsupported/contradicted claim:         {_fmt(s['alerts_with_bad_claim'])}",
          f"By source: {c['by_source']}",
          f"Rules that fired (claims may fire several): {c['reason_counts']}",
          "Reading: a claim marked supported means its checkable atoms are grounded, not that the sentence is true. No LLM judge was",
          "used. Typhoon grading Typhoon would not be independent, and no second model was available that would not need its own",
          "validation.", ""]
    L += [bar, "5. KNOWN DEFECTS FROM models/llm_layer_config.json (measured on this set)", bar]
    L += ["Defect C (new, not in quality_issues): few-shot entity copying, which quality_issues_fixed says was fixed.",
          f"  Alerts echoing a rendered few-shot example (difflib ratio >= {EXAMPLE_ECHO_THRESHOLD}): {_fmt(s['example_echo_alerts'])}",
          f"    highest similarity seen: {s['example_echo_ratio_max']}   mean: {s['example_echo_ratio_mean']}",
          "    The examples now carry this alert's own values, so a copied entity is the CORRECT",
          "    entity and the old few-shot-leak check cannot fire. Echoing the example's sentence",
          "    is the risk that replaced it, which is what this measures.",
          "Defect A: PortScan remediation suggesting a WAF block.",
          f"  RF-predicted PortScan alerts with WAF in remediation:   {_fmt(s['defect_waf_portscan_pred'])}",
          f"  Truly PortScan alerts with WAF in remediation:          {_fmt(s['defect_waf_portscan_true'])}",
          f"  ... of those with dst_port 139/445 (SMB):               {_fmt(s['defect_waf_portscan_true_smb'])}",
          f"  Any alert with WAF while dst_port is non-web and label not Web Attack: {_fmt(s['waf_nonweb_all'])}",
          "Defect B: explanation does not state the RAG evidence is insufficient (keyword regex, approximated).",
          f"  Truly SSH-Patator alerts where technique abstained:      {_fmt(s['ssh_true_technique_null'])}",
          f"  Truly SSH-Patator alerts stating insufficiency:          {_fmt(s['defect_insufficient_ssh_true'])}",
          f"  ... among SSH-Patator alerts that abstained:             {_fmt(s['defect_insufficient_ssh_null'])}",
          f"  All abstained alerts (any category) stating insufficiency: {_fmt(s['defect_insufficient_null_all'])}",
          "The prompt only asks for the insufficiency statement when technique_id is null, so the abstained denominators are the fair ones.", ""]
    t = s["throughput"]
    L += [bar, "6. THROUGHPUT (measured)", bar]
    L += [f"Alerts with Ollama timing metadata: {t['n_with_metrics']}.",
          f"Generation: {t['gen_tokens_total']} tokens in {t['gen_seconds_total']:.1f}s of decode = {t['gen_tok_s_pooled']:.1f} tok/s pooled "
          f"(per-alert median {t['gen_tok_s_median']:.1f}, min {t['gen_tok_s_min']:.1f}, max {t['gen_tok_s_max']:.1f}).",
          f"Mean prompt tokens: {t['prompt_tokens_mean']:.0f}. Mean wall time per alert: {t['wall_s_mean']:.2f}s; first alert {t['wall_s_first']:.2f}s (may include model load).",
          f"Total LLM wall time: {t['wall_s_total_llm']:.0f}s." + (f" Whole run including alert build and retrieval: {wall_total_s:.0f}s." if wall_total_s else ""), ""]
    L += [bar, "7. WHAT IS NOT MEASURED", bar]
    L += [f"- Whether undecidable claims ({c['verdicts'].get('undecidable', 0)} of {c['total']}) are true. These are the Thai prose statements and generic remediation",
          "  advice with no checkable entity. Judging them needs human raters (SOC analysts) against the retrieved chunks.",
          "- Whether 'supported' claims are true as sentences; only their entities and terms are grounded.",
          "- Semantic contradictions with no lexical trigger, for example a remediation that is wrong for the attack but uses only",
          "  grounded words, or severity language beyond the three English priority words and three Thai severity words.",
          "- Remediation quality, safety and operational fit (is blocking this IP appropriate for this campus). Not mechanically decidable.",
          "- Hostname preservation on real data: CICIDS-2017 has no hostnames. The check exists and is unit-tested only.",
          "- Any live (non-CICIDS) alert. PA-5220 firewall logs were not available to this run.",
          "- Run-to-run variance: one sample per alert at temperature 0.1; no repeated draws.",
          "- Inter-rater reliability or agreement between these rules and human judgment: none, because no human labels exist.",
          "- Priority correctness: priority is a deterministic rule (derive_priority), not model output.", ""]
    L += [bar, "8. HOW TO RECHECK A NUMBER", bar]
    L += ["Each line of reports/llm_faithfulness_raw.jsonl holds the alert context, the retrieved hits (id, metadata, text), the raw",
          "model output, and every per-alert check. Entity pass = checks.entity.pass. Claim verdicts = checks.claims[*].verdict.",
          "Prose grounding = checks.grounding.prose_ungrounded non-empty. Defects = checks.defects.*. The aggregate rows in",
          "llm_faithfulness_metrics.csv are counts over those fields. `--rescore` recomputes all checks and both reports from the raw",
          "outputs without calling the LLM, so scoring-rule changes never require regenerating text.", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Stage 1 (Anaconda): sample real alerts, score them, retrieve RAG hits
# ----------------------------------------------------------------------------
NON_FEATURE_COLS = [
    "Label", "attack_cat", "technique_id", "technique_name", "tactic", "mapping_confidence",
    "technique_id_encoded", "tactic_encoded", "mapping_confidence_encoded", "is_known_technique",
    "orig_row_index",
]
EXPECTED_FEATURES = 94
GENERIC_ANOMALY_QUERY = "anomalous network traffic flagged by anomaly detector"


def build_alerts() -> None:
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(ROOT))
    from src.features import build_unified_population as BUP
    from src.modeling import fusion_layer
    from src.rag.rag_retrieval import RagRetriever

    rng = np.random.default_rng(SEED)
    test = pd.read_parquet(UNIFIED_TEST)
    plan: list[tuple[str, int]] = []
    for label, n_avail in test["Label"].value_counts().items():
        plan.append((label, BENIGN_N if label == "BENIGN" else min(PER_CATEGORY_CAP, int(n_avail))))
    picks = []
    for label, n in sorted(plan):
        idx = test.index[test["Label"] == label].to_numpy()
        picks.append(rng.choice(idx, size=n, replace=False))
    pos_idx = np.concatenate(picks)
    sample = test.loc[pos_idx].reset_index(drop=True)
    print(f"[alerts] sampled {len(sample)} rows: {sample['Label'].value_counts().to_dict()}")

    # Recover the identifiers the feature matrix deliberately drops, by re-deriving the same
    # split from the same source with the same seed. split_population is imported rather than
    # reimplemented so it cannot drift from the split the deployed models were trained on.
    print("[alerts] re-deriving the unified test split to recover IP/timestamp ...")
    _, _, raw_test = BUP.split_population(pd.read_parquet(BUP.SOURCE))
    assert len(raw_test) == len(test), f"re-derived test split has {len(raw_test)} rows, parquet has {len(test)}"
    ctx_rows = raw_test.iloc[pos_idx].reset_index(drop=True)
    assert (ctx_rows["Label"].to_numpy() == sample["Label"].to_numpy()).all(), \
        "positional alignment between the feature matrix and the re-derived split is wrong"

    # One feature space for all three members -- no join, no second column list.
    feat_cols = [c for c in test.columns
                 if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(test[c])]
    assert len(feat_cols) == EXPECTED_FEATURES, f"expected {EXPECTED_FEATURES}, got {len(feat_cols)}"
    x = sample[feat_cols].to_numpy(dtype=np.float64)

    rf = joblib.load(ROOT / "models/random_forest_unified.pkl")
    proba = rf.predict_proba(x)
    classes = list(rf.classes_)
    rf_attack = 1.0 - proba[:, classes.index("BENIGN")]
    rf_label = np.array(classes)[proba.argmax(1)]

    iso = joblib.load(ROOT / "models/isolation_forest_unified.pkl")
    if_score = -iso.score_samples(x)
    xb = x.astype(np.float32)
    cfg = json.loads((ROOT / "models/autoencoder_unified_config.json").read_text(encoding="utf-8"))

    def stack(dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class AE(nn.Module):
        def __init__(self, d, hidden, bottleneck):
            super().__init__()
            self.encoder = stack([d] + list(hidden) + [bottleneck])
            self.decoder = stack([bottleneck] + list(hidden)[::-1] + [d])

        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = AE(cfg["input_dim"], cfg["hidden_dims"], cfg["bottleneck_dim"])
    ae.load_state_dict(torch.load(ROOT / "models/autoencoder_unified.pt", map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        t = torch.tensor(xb)
        ae_err = torch.mean((ae(t) - t) ** 2, dim=1).numpy()
    fused, pred, margin, model_used = fusion_layer.fuse_with_metadata(rf_attack, if_score, ae_err)

    # The row-hash join that used to live here is gone: there is only one population now,
    # so there is nothing to reconcile. ctx_rows came from the positional re-derivation above.
    retriever = RagRetriever()
    alerts = []
    for i in range(len(sample)):
        label = str(sample.loc[i, "Label"])
        pred_label = str(rf_label[i])
        query = pred_label if pred_label != "BENIGN" else GENERIC_ANOMALY_QUERY
        hits = retriever.retrieve(query)
        row = ctx_rows.iloc[i]
        confidence = float(abs(rf_attack[i] - 0.5) * 2)
        alerts.append({
            "alert_id": f"A{i:03d}",
            "true_label": label,
            "test_row_index": int(pos_idx[i]),
            "label_mapping_technique_id": (None if pd.isna(sample.loc[i, "technique_id"]) else str(sample.loc[i, "technique_id"])),
            "alert_context": {
                "source_ip": str(row["Source IP"]),
                "dst_ip": str(row["Destination IP"]),
                "dst_port": int(row["Destination Port"]),
                "timestamp": str(row["Timestamp"]),
                "rf_predicted_label": pred_label,
                "fused_score": float(fused[i]),
                "fused_prediction": int(pred[i]),
                "confidence": confidence,
                "model_used": str(model_used[i]),
            },
            "scores": {"rf_attack_probability": float(rf_attack[i]), "if_anomaly_score": float(if_score[i]), "ae_reconstruction_error": float(ae_err[i])},
            "rag_query": query,
            "rag_hits": [{"rank": h["rank"], "id": h["id"], "metadata": h["metadata"], "document": h["document"], "combined_score": float(h["combined_score"])} for h in hits],
        })
    with ALERTS_PATH.open("w", encoding="utf-8") as fh:
        for a in alerts:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[alerts] wrote {len(alerts)} alerts to {ALERTS_PATH}")


# ----------------------------------------------------------------------------
# Stage 2 (uv): LLM calls with per-alert checkpointing
# ----------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class _RecordingModel:
    """Wraps the chat model so explain_alert's own call path runs unchanged while the Ollama timing
    metadata (eval_count, eval_duration, ...) is captured; langchain_client.invoke_structured drops it."""

    def __init__(self, inner: Any, sink: dict[str, Any]):
        self.inner, self.sink = inner, sink

    def with_structured_output(self, schema: Any, method: str):
        runnable = self.inner.with_structured_output(schema, method=method, include_raw=True)
        sink = self.sink

        class _Call:
            def invoke(self, prompt: str):
                result = runnable.invoke(prompt)
                sink["meta"] = dict(getattr(result["raw"], "response_metadata", {}) or {})
                if result.get("parsing_error") is not None:
                    raise result["parsing_error"]
                return result["parsed"]

        return _Call()


def run_llm(limit: int | None) -> float:
    from src.llm import langchain_client, llm_explain

    config = llm_explain.load_env_config()
    kb = load_kb()
    alerts = _read_jsonl(ALERTS_PATH)
    if limit:
        alerts = alerts[:limit]
    done = {r["alert_id"] for r in _read_jsonl(RAW_PATH)}
    sink: dict[str, Any] = {}
    original = llm_explain.invoke_structured

    def recording_invoke(prompt: str, cfg: dict[str, Any], model: Any = None):
        return langchain_client.invoke_structured(prompt, cfg, _RecordingModel(langchain_client.build_chat_model(cfg), sink))

    llm_explain.invoke_structured = recording_invoke
    t_run = time.perf_counter()
    try:
        for n, alert in enumerate(alerts, start=1):
            if alert["alert_id"] in done:
                print(f"[{n}/{len(alerts)}] {alert['alert_id']} already done, skipping", flush=True)
                continue
            ctx = alert["alert_context"]
            sink.clear()
            t0 = time.perf_counter()
            rec = {k: v for k, v in alert.items()}
            tech_id, tactic = llm_explain._technique_context(alert["rag_hits"])
            rec["ctx_technique_id"], rec["ctx_tactic"] = tech_id, tactic
            rec["confidence"] = ctx["confidence"]
            rec["priority"] = llm_explain.derive_priority(
                ctx["fused_score"], ctx["fused_prediction"], ctx.get("model_used"))
            rec["rag_hit_ids"] = [h["id"] for h in alert["rag_hits"]]
            try:
                facts_for_shots = {
                    "alert_context": ctx, "confidence": ctx["confidence"],
                    "technique_id": tech_id, "mitre_tactic": tactic,
                }
                rec["rendered_example_explanations"] = [
                    sh["output"]["explanation_th"] for sh in llm_explain.render_few_shots(facts_for_shots)
                ]
                output = llm_explain.explain_alert(ctx, alert["rag_hits"], ctx["fused_score"], ctx["fused_prediction"], ctx["confidence"], config)
                rec.update(status="ok", error=None, output=output)
            except Exception as exc:
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}", output=None)
            wall = time.perf_counter() - t0
            meta = sink.get("meta", {})
            if meta.get("eval_count") and meta.get("eval_duration"):
                gen_s = meta["eval_duration"] / 1e9
                rec["llm_metrics"] = {
                    "prompt_tokens": meta.get("prompt_eval_count"), "gen_tokens": meta["eval_count"], "gen_s": gen_s,
                    "gen_tok_s": meta["eval_count"] / gen_s, "load_s": (meta.get("load_duration") or 0) / 1e9, "wall_s": wall,
                }
            else:
                rec["llm_metrics"] = {"wall_s": wall}
            rec["checks"] = score_record(rec, kb) if rec["status"] == "ok" else None
            with RAW_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            tag = rec["status"] if rec["status"] != "ok" else f"ok entity_pass={rec['checks']['entity']['pass']} tok/s={rec['llm_metrics'].get('gen_tok_s', 0):.1f}"
            print(f"[{n}/{len(alerts)}] {alert['alert_id']} {alert['true_label']:<26} {wall:5.1f}s {tag}", flush=True)
    finally:
        llm_explain.invoke_structured = original
    return time.perf_counter() - t_run


def finalize(rescore: bool, wall_total_s: float | None) -> None:
    records = _read_jsonl(RAW_PATH)
    if rescore:
        kb = load_kb()
        for r in records:
            if r["status"] == "ok":
                r["checks"] = score_record(r, kb)
        RAW_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = summarize(records)
    write_metrics_csv(summary)
    write_report(summary, records, wall_total_s=wall_total_s)
    print(f"[finalize] wrote {METRICS_PATH.name} and {REPORT_PATH.name} from {len(records)} raw records")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["alerts", "all"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--rebuild-alerts", action="store_true")
    args = ap.parse_args()
    if args.stage == "alerts":
        build_alerts()
        return 0
    if args.rescore:
        finalize(True, None)
        return 0
    started = time.perf_counter()
    if args.rebuild_alerts or not ALERTS_PATH.exists():
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        subprocess.run([ML_PYTHON, "-m", "src.evaluation.eval_llm_faithfulness", "--stage", "alerts"], cwd=ROOT, env=env, check=True)
    run_llm(args.limit)
    finalize(False, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
