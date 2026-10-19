
from __future__ import annotations

import logging
from typing import Any

import requests

from .config_utils import looks_real

logger = logging.getLogger("alert_dispatcher.notify_line")

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"


def summarize_two_lines(explanation_th: str, max_chars: int = 150) -> str:
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
