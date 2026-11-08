"""Rendering helpers every notification channel shares.

LINE and Gmail show the same port, technique and tactic strings. They live here so the two
channels cannot drift apart, the failure fusion_layer.MODEL_USED_VALUES and
alert_dispatcher.derive_model_used already had to be guarded against by a contract test.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("alert_dispatcher.alert_format")

GRAFANA_DASHBOARD_PATH = "/d/soc-main"

SERVICE_BY_PORT = {
    20: "FTP-DATA", 21: "FTP", 22: "SSH", 23: "TELNET", 25: "SMTP", 53: "DNS", 67: "DHCP",
    80: "HTTP", 110: "POP3", 123: "NTP", 143: "IMAP", 161: "SNMP", 389: "LDAP", 443: "HTTPS",
    445: "SMB", 465: "SMTPS", 587: "SMTP", 636: "LDAPS", 993: "IMAPS", 995: "POP3S",
    1433: "MSSQL", 1521: "Oracle", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
    5900: "VNC", 6379: "Redis", 8080: "HTTP-ALT", 8443: "HTTPS-ALT", 27017: "MongoDB",
}

# Half the measured alerts carry no technique at all: llm_explain derives it by majority
# vote over the RAG hits and returns nothing when they disagree, so technique_id and
# mitre_tactic are both null for 50 of the 100 alerts in reports/llm_faithfulness_raw.jsonl.
# That is an honest absence, not a failure, and it reads as one only if the message says so.
# Printing "?" against those labels makes a working system look broken.
UNKNOWN_TECHNIQUE_TEXT = "ไม่ระบุ (RAG ไม่พบเทคนิคที่สอดคล้องกัน)"

_KB_NAMES: dict[str, str] | None = None


def technique_name(technique_id: str | None) -> str | None:
    """The ATT&CK name for a technique id, from the same KB the LLM layer reads.

    llm_explain derives technique_id from the RAG hits by looking it up in this file, so
    reading the name here cannot disagree with the id it accompanies. A missing KB returns
    None and the caller drops the name rather than inventing one.
    """
    global _KB_NAMES
    if not technique_id:
        return None
    if _KB_NAMES is None:
        try:
            import json
            from pathlib import Path

            kb = Path(__file__).resolve().parents[2] / "data/knowledge_base/mitre_techniques.json"
            raw = json.loads(kb.read_text(encoding="utf-8"))["techniques"]
            _KB_NAMES = {k: v.get("name", "") for k, v in raw.items()}
        except Exception as exc:
            logger.warning("technique name lookup unavailable: %s", exc)
            _KB_NAMES = {}
    return _KB_NAMES.get(technique_id) or None


def tactic_label(tactic: str | None) -> str | None:
    """ATT&CK tactics arrive lowercase and hyphenated. Analysts read them as MITRE writes them."""
    if not tactic:
        return None
    words = str(tactic).replace("_", "-").split("-")
    return " ".join(w if w == "and" else w.capitalize() for w in (w.lower() for w in words))


def port_label(port: Any) -> str:
    if port is None:
        return "?"
    try:
        number = int(port)
    except (TypeError, ValueError):
        return str(port)
    service = SERVICE_BY_PORT.get(number)
    return f"{number} ({service})" if service else str(number)


def grafana_dashboard_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{GRAFANA_DASHBOARD_PATH}"
