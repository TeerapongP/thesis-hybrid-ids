"""
notify_line.py
===============
LINE sender for HIGH-priority alerts -- per `diagrams/thesis_dataflow_v2.drawio`'s
"Line Notify / Line Bot" node: *"HIGH priority -> ส่งทันที · ข้อความ: priority · src_ip ·
mitre_tactic · explanation_th (สรุป 2 บรรทัด) + link Grafana"*.

Backend: LINE Messaging API push. LINE Notify was retired in 2025, so a channel access token
and explicit user/group/room recipient id (`LINE_TO_ID`) are both required.

Real-vs-dry-run rule: both `LINE_CHANNEL_ACCESS_TOKEN` and `LINE_TO_ID` must look configured;
otherwise the module returns a dry-run result without making an external call. Secret values are
never logged.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from .config_utils import looks_real

logger = logging.getLogger("alert_dispatcher.notify_line")

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"


def summarize_two_lines(explanation_th: str, max_chars: int = 150) -> str:
    """Truncate `explanation_th` to a ~2-line (~150 char) summary at a word/clause boundary --
    never mid-word. Thai text in this repo's examples has no period-delimited "sentences" (it
    uses em-dashes / commas as clause breaks instead -- see the few-shot examples in
    HANDOFF_LLM_LAYER.md), so this prefers cutting at the last em-dash or, failing that, the
    last whitespace before `max_chars`, rather than assuming '.'-delimited sentences exist."""
    text = (explanation_th or "").strip()
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    dash_cut = max(window.rfind("—"), window.rfind("-"))
    space_cut = window.rfind(" ")
    cut = dash_cut if dash_cut > max_chars // 3 else space_cut
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip(" ,;-—") + "..."


def build_line_message(record: dict[str, Any], config: dict[str, Any]) -> str:
    """Per the diagram's exact field list: priority, src_ip, mitre_tactic, a 2-line
    explanation_th summary, plus a Grafana deep link (soc-main dashboard uid, per
    grafana/dashboards/soc-overview.json)."""
    summary = summarize_two_lines(record.get("explanation_th", ""))
    grafana_link = f"{config['line_grafana_base_url'].rstrip('/')}/d/soc-main"
    return (
        f"\n[SOC ALERT - {record.get('priority', '?')}]\n"
        f"src_ip: {record.get('source_ip', '?')}\n"
        f"mitre_tactic: {record.get('mitre_tactic', '?')}\n"
        f"{summary}\n"
        f"Grafana: {grafana_link}"
    )


def send_line_alert(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Send (or dry-run-log) a HIGH-priority LINE Messaging API message for one assembled alert
    `record` (see src/notifications/alert_dispatcher.py's assemble_alert_record). Never raises -- any
    failure (network, non-200, missing credential) is returned in the status dict so the
    caller (alert_dispatcher.dispatch_alert) can log-and-continue without crashing."""
    token = config.get("line_channel_access_token")
    recipient = config.get("line_to_id")
    message = build_line_message(record, config)

    if not looks_real(token, min_len=20) or not looks_real(recipient, min_len=8):
        logger.info(
            "[notify_line] dry-run: LINE_CHANNEL_ACCESS_TOKEN/LINE_TO_ID are not both configured -- "
            "would-be message logged, no HTTP call made"
        )
        return {"status": "dry_run", "reason": "LINE credentials/recipient not configured", "would_send_message": message}

    try:
        resp = requests.post(
            LINE_PUSH_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"to": recipient, "messages": [{"type": "text", "text": message}]},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("[notify_line] Messaging API request failed: %s", type(exc).__name__)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code in (200, 202):
        logger.info("[notify_line] Messaging API accepted the message (HTTP %s)", resp.status_code)
        return {"status": "sent", "http_status": resp.status_code}

    logger.warning("[notify_line] Messaging API returned HTTP %s (not sent)", resp.status_code)
    return {"status": "error", "http_status": resp.status_code, "response_body": resp.text[:300]}
