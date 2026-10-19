
from __future__ import annotations

import base64
from html import escape
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from .config_utils import looks_real

logger = logging.getLogger("alert_dispatcher.notify_gmail")

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def _gmail_creds_look_real(config: dict[str, Any]) -> bool:
    return (
        looks_real(config.get("gmail_client_id"))
        and looks_real(config.get("gmail_client_secret"))
        and looks_real(config.get("gmail_refresh_token"))
        and looks_real(config.get("gmail_sender_address"), min_len=5)
    )


def _build_gmail_service(config: dict[str, Any]):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=config["gmail_refresh_token"],
        client_id=config["gmail_client_id"],
        client_secret=config["gmail_client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=[GMAIL_SEND_SCOPE],
    )
    return build("gmail", "v1", credentials=creds)


def _send_html_email(config: dict[str, Any], subject: str, html_body: str, dry_run_label: str) -> dict[str, Any]:
    if not _gmail_creds_look_real(config):
        logger.info(
            "[notify_gmail] dry-run (%s): GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET/GMAIL_REFRESH_TOKEN/"
            "GMAIL_SENDER_ADDRESS are not all configured with real-looking values -- "
            "would-be email logged, no Gmail API call made",
            dry_run_label,
        )
        return {
            "status": "dry_run",
            "reason": "no real-looking GMAIL_* credentials configured",
            "would_send_subject": subject,
            "would_send_to": config.get("gmail_sender_address") or "(unset)",
        }

    try:
        import httplib2
        from google.auth.exceptions import GoogleAuthError
        from googleapiclient.errors import HttpError

        service = _build_gmail_service(config)
        message = MIMEMultipart("alternative")
        message["to"] = config["gmail_sender_address"]
        message["from"] = config["gmail_sender_address"]
        message["subject"] = subject
        message.attach(MIMEText(html_body, "html", "utf-8"))
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        resp = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except ImportError as exc:
        logger.warning("[notify_gmail] Gmail API send failed: %s", type(exc).__name__)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    except (HttpError, GoogleAuthError, httplib2.HttpLib2Error, OSError, TimeoutError, ValueError, KeyError) as exc:
        logger.warning("[notify_gmail] Gmail API send failed: %s", type(exc).__name__)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    logger.info("[notify_gmail] Gmail API accepted the message (id=%s)", resp.get("id"))
    return {"status": "sent", "message_id": resp.get("id")}


def build_immediate_email_html(record: dict[str, Any]) -> str:
    remediation_html = "".join(f"<li>{escape(str(step))}</li>" for step in (record.get("remediation") or []))
    return (
        "<html><body>"
        f"<h2>SOC Alert - {escape(str(record.get('priority', '?')))}</h2>"
        f"<p><b>alert_id:</b> {escape(str(record.get('alert_id', '?')))}<br>"
        f"<b>timestamp:</b> {escape(str(record.get('timestamp', '?')))}<br>"
        f"<b>source_ip:</b> {escape(str(record.get('source_ip', '?')))}<br>"
        f"<b>dst_port:</b> {escape(str(record.get('dst_port', '?')))}<br>"
        f"<b>affected_asset:</b> {escape(str(record.get('affected_asset', '?')))}<br>"
        f"<b>technique:</b> {escape(str(record.get('technique_id', '?')))} ({escape(str(record.get('mitre_tactic', '?')))})<br>"
        f"<b>risk_score:</b> {escape(str(record.get('risk_score', '?')))}</p>"
        f"<p>{escape(str(record.get('explanation_th', '')))}</p>"
        f"<ul>{remediation_html}</ul>"
        "</body></html>"
    )


def send_immediate_email(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    subject = f"[SOC ALERT - MEDIUM] {record.get('anomaly_type', '?')} @ {record.get('source_ip', '?')}"
    html = build_immediate_email_html(record)
    return _send_html_email(config, subject, html, dry_run_label="MEDIUM immediate")


def build_digest_email_html(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    top_types_html = "".join(f"<li>{escape(str(t))}: {escape(str(n))}</li>" for t, n in stats.get("top_attack_types", []))
    remediation_html = "".join(f"<li>{escape(str(step))}</li>" for step in stats.get("sample_remediation_steps", []))
    rows_html = "".join(
        f"<tr><td>{escape(str(r.get('timestamp', '?')))}</td><td>{escape(str(r.get('source_ip', '?')))}</td>"
        f"<td>{escape(str(r.get('anomaly_type', '?')))}</td><td>{escape(str(r.get('risk_score', '?')))}</td></tr>"
        for r in rows
    )
    avg_mttd = stats.get("avg_mttd_ms")
    avg_mttd_text = f"{avg_mttd:.1f} ms" if avg_mttd is not None else "N/A (missing ingest/detection timestamps)"
    return (
        "<html><body>"
        f"<h2>SOC Daily Digest - {escape(str(stats.get('date', '?')))}</h2>"
        f"<p><b>alert_count:</b> {escape(str(stats.get('alert_count', 0)))}<br>"
        f"<b>Average MTTD:</b> {avg_mttd_text}</p>"
        f"<p><b>Top attack types:</b></p><ul>{top_types_html}</ul>"
        f"<p><b>Sample remediation steps:</b></p><ul>{remediation_html}</ul>"
        f"<table border='1'><tr><th>timestamp</th><th>source_ip</th><th>anomaly_type</th><th>risk_score</th></tr>"
        f"{rows_html}</table>"
        "</body></html>"
    )


def send_digest_email(rows: list[dict[str, Any]], stats: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    subject = f"[SOC Daily Digest] {stats.get('date', '?')} -- {stats.get('alert_count', 0)} LOW alerts"
    html = build_digest_email_html(rows, stats)
    return _send_html_email(config, subject, html, dry_run_label="LOW digest")
