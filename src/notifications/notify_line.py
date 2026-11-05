
from __future__ import annotations

import logging
from typing import Any

import requests

from .config_utils import looks_real

logger = logging.getLogger("alert_dispatcher.notify_line")

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
# Broadcast reaches every account that has added the bot as a friend, so it needs no
# recipient id. That suits a lab SOC channel, where analysts subscribe by adding the bot
# and nobody maintains a roster of user ids -- and LINE_TO_ID cannot be obtained
# programmatically anyway (/v2/bot/followers/ids needs a verified or premium Official
# Account). Two costs come with it, both real rather than theoretical:
#   * 60 requests per hour per channel, against 2,000/s for push. A burst of HIGH alerts
#     above one per minute will be rate-limited, so broadcast is not a drop-in for a
#     production alerting path.
#   * No routing. Every friend of the channel receives every alert; HIGH-to-on-call is
#     not expressible.
LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"

DELIVERY_MODES = ("push", "broadcast")


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


def resolve_delivery(config: dict[str, Any]) -> tuple[str, str | None]:
    """Pick the endpoint, and say why, without reading the config twice at the call site.

    An unrecognised mode falls back to push rather than to broadcast. Broadcast is the
    wider blast radius of the two, so a typo in LINE_DELIVERY_MODE must not be what
    decides to message every friend of the channel.
    """
    mode = (config.get("line_delivery_mode") or "push").strip().lower()
    if mode not in DELIVERY_MODES:
        logger.warning("[notify_line] unknown LINE_DELIVERY_MODE %r -- falling back to push", mode)
        mode = "push"
    return mode, (LINE_BROADCAST_URL if mode == "broadcast" else LINE_PUSH_URL)


def send_line_alert(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    token = config.get("line_channel_access_token")
    recipient = config.get("line_to_id")
    message = build_line_message(record, config)
    mode, url = resolve_delivery(config)

    if not looks_real(token, min_len=20):
        logger.info(
            "[notify_line] dry-run: LINE_CHANNEL_ACCESS_TOKEN is not configured -- "
            "would-be message logged, no HTTP call made"
        )
        return {"status": "dry_run", "reason": "LINE channel access token not configured",
                "delivery_mode": mode, "would_send_message": message}

    # Only push needs a destination. Keeping this check inside the push branch is what
    # lets broadcast mode work with LINE_TO_ID left empty.
    if mode == "push" and not looks_real(recipient, min_len=8):
        logger.info(
            "[notify_line] dry-run: LINE_TO_ID is not configured and LINE_DELIVERY_MODE is "
            "push -- would-be message logged, no HTTP call made"
        )
        return {"status": "dry_run", "reason": "LINE recipient not configured",
                "delivery_mode": mode, "would_send_message": message}

    payload: dict[str, Any] = {"messages": [{"type": "text", "text": message}]}
    if mode == "push":
        payload["to"] = recipient

    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("[notify_line] Messaging API request failed: %s", type(exc).__name__)
        return {"status": "error", "delivery_mode": mode, "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code in (200, 202):
        logger.info("[notify_line] Messaging API accepted the %s (HTTP %s)", mode, resp.status_code)
        return {"status": "sent", "delivery_mode": mode, "http_status": resp.status_code}

    if resp.status_code == 429:
        # Distinguishable on purpose: broadcast allows only 60 requests per hour per
        # channel, so a burst of HIGH alerts hits this while the credentials are fine.
        logger.warning("[notify_line] Messaging API returned 429 -- rate limit or monthly "
                       "quota exhausted (broadcast allows 60 requests/hour)")
        return {"status": "error", "delivery_mode": mode, "http_status": 429,
                "reason": "rate limit or monthly message quota",
                "response_body": resp.text[:300]}

    logger.warning("[notify_line] Messaging API returned HTTP %s (not sent)", resp.status_code)
    return {"status": "error", "delivery_mode": mode, "http_status": resp.status_code,
            "response_body": resp.text[:300]}
