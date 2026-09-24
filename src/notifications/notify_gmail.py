"""
notify_gmail.py
================
Gmail sender for MEDIUM-priority immediate alerts and the LOW-priority daily digest --
per `diagrams/thesis_dataflow_v2.drawio`'s "Outlook (Microsoft Graph API)" node text (the
diagram itself is unchanged -- see HANDOFF_OUTPUT_LAYER.md's "notification channel decision
superseded" note): *"MEDIUM: ส่งทันที · LOW: daily digest · สรุปรายวัน: alert count · top attack
types · MTTD avg · remediation steps · HTML template"*. The provider was changed from
Outlook/Microsoft Graph API to Gmail API (google-api-python-client) after this handoff was
first drafted; the routing rule and HTML content spec are unchanged, only the delivery channel
and its auth model are different.

Backend: Gmail API v1 (`users.messages.send`), authenticated via a standard OAuth2
**user-consent** flow (a Google Cloud Console OAuth2 client id/secret + one authorized Google
account's own long-lived refresh token) -- NOT a Workspace service account with domain-wide
delegation. This is a real architecture difference from Graph's app-only client-credentials
flow, not just a renamed equivalent: the Gmail API only allows an app-only/service-account flow
to send mail on another user's behalf under Google WORKSPACE domain-wide delegation, which needs
a domain admin and a real Workspace tenant -- exactly the kind of real-tenant dependency this
project doesn't have (the same "no real tenant" fact that picked Graph's app-only flow now
pushes Gmail the OTHER way: a single consumer/personal Google account's own OAuth2 consent,
since there is no Workspace domain to delegate within). See HANDOFF_OUTPUT_LAYER.md's renumbered
Open Question #2 for this written out in full.

Status: stub/mock only, same as the Graph API stub it replaces -- GMAIL_CLIENT_ID/
GMAIL_CLIENT_SECRET/GMAIL_REFRESH_TOKEN/GMAIL_SENDER_ADDRESS are blank placeholders in
notification/.env (see that file's own comment for what's needed: an OAuth2 client id/secret
downloaded from Google Cloud Console, plus a refresh token minted once via a local
`InstalledAppFlow` consent run -- neither done here). Every call in this module returns
status="dry_run" until those are filled in. `google-api-python-client`/`google-auth-oauthlib`
are NOT installed in this repo's environment (confirmed: `ModuleNotFoundError` on both against
the system Anaconda Python this repo's other src/*.py modules use) -- deliberately imported
lazily, only inside the real-send path, so this module stays importable (and the dry-run path
fully exercisable) without them; installing them is only required once real credentials make the
real-send path reachable.

Real-vs-dry-run rule (same discipline src/notifications/notify_line.py uses for LINE_CHANNEL_ACCESS_TOKEN,
mirrored here per HANDOFF_OUTPUT_LAYER.md's instruction to keep the two channels consistent):
if GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET/GMAIL_REFRESH_TOKEN/GMAIL_SENDER_ADDRESS don't all look
real (empty, placeholder text, or too short), this module logs the would-be email and returns a
dry-run status WITHOUT attempting a Gmail API call. No secret value is ever logged or printed.
"""

from __future__ import annotations

import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from .config_utils import looks_real

logger = logging.getLogger("alert_dispatcher.notify_gmail")

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def _gmail_creds_look_real(config: dict[str, Any]) -> bool:
    """All four GMAIL_* values must look real before any live Gmail API call is attempted --
    a client id without a refresh token (or vice versa) can't authenticate, so a partial
    configuration is treated the same as no configuration (dry-run), never a half-attempt."""
    return (
        looks_real(config.get("gmail_client_id"))
        and looks_real(config.get("gmail_client_secret"))
        and looks_real(config.get("gmail_refresh_token"))
        and looks_real(config.get("gmail_sender_address"), min_len=5)
    )


def _build_gmail_service(config: dict[str, Any]):
    """Lazy import -- only reached once _gmail_creds_look_real(config) is True (see module
    docstring for why the google-api-python-client/google-auth libraries are not imported at
    module load time). Builds a Gmail API client from a long-lived refresh token, not a fresh
    interactive consent -- this module never opens a browser itself."""
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
    """Shared send path for both the MEDIUM immediate email and the LOW daily digest -- same
    dry-run gate, same Gmail API call shape, only the subject/body differ. Recipient defaults to
    GMAIL_SENDER_ADDRESS itself (send-to-self) since no SOC email distribution list is configured
    anywhere in this repo yet -- same placeholder-recipient precedent the Graph stub it replaces
    used. Never raises -- any failure (network, API error, missing credential) is returned in the
    status dict so the caller (alert_dispatcher.dispatch_alert / flush_low_priority_digest) can
    log-and-continue without crashing."""
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
        service = _build_gmail_service(config)
        message = MIMEMultipart("alternative")
        message["to"] = config["gmail_sender_address"]
        message["from"] = config["gmail_sender_address"]
        message["subject"] = subject
        message.attach(MIMEText(html_body, "html", "utf-8"))
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        resp = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except Exception as exc:  # noqa: BLE001 -- a Gmail API outage must never crash the dispatcher
        logger.warning("[notify_gmail] Gmail API send failed: %s", type(exc).__name__)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    logger.info("[notify_gmail] Gmail API accepted the message (id=%s)", resp.get("id"))
    return {"status": "sent", "message_id": resp.get("id")}


def build_immediate_email_html(record: dict[str, Any]) -> str:
    """MEDIUM content spec (unchanged from the Graph stub it replaces): alert_id/timestamp/
    source_ip/dst_port/affected_asset/technique_id+mitre_tactic/risk_score/explanation_th/
    remediation."""
    remediation_html = "".join(f"<li>{step}</li>" for step in (record.get("remediation") or []))
    return (
        "<html><body>"
        f"<h2>SOC Alert - {record.get('priority', '?')}</h2>"
        f"<p><b>alert_id:</b> {record.get('alert_id', '?')}<br>"
        f"<b>timestamp:</b> {record.get('timestamp', '?')}<br>"
        f"<b>source_ip:</b> {record.get('source_ip', '?')}<br>"
        f"<b>dst_port:</b> {record.get('dst_port', '?')}<br>"
        f"<b>affected_asset:</b> {record.get('affected_asset', '?')}<br>"
        f"<b>technique:</b> {record.get('technique_id', '?')} ({record.get('mitre_tactic', '?')})<br>"
        f"<b>risk_score:</b> {record.get('risk_score', '?')}</p>"
        f"<p>{record.get('explanation_th', '')}</p>"
        f"<ul>{remediation_html}</ul>"
        "</body></html>"
    )


def send_immediate_email(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Send (or dry-run-log) a MEDIUM-priority immediate email for one assembled alert `record`
    (see src/notifications/alert_dispatcher.py's assemble_alert_record). Called by
    src/alert_dispatcher.dispatch_alert() when priority == "MEDIUM". Never raises."""
    subject = f"[SOC ALERT - MEDIUM] {record.get('anomaly_type', '?')} @ {record.get('source_ip', '?')}"
    html = build_immediate_email_html(record)
    return _send_html_email(config, subject, html, dry_run_label="MEDIUM immediate")


def build_digest_email_html(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    """LOW daily-digest content spec (unchanged from the Graph stub it replaces): alert count,
    top attack types, avg MTTD PROXY (response_ms -- see models/output_layer_config.json's
    mttd_proxy_note; this module doesn't change that reasoning, only the delivery channel),
    sample remediation steps, full table of every alert in the batch."""
    top_types_html = "".join(f"<li>{t}: {n}</li>" for t, n in stats.get("top_attack_types", []))
    remediation_html = "".join(f"<li>{step}</li>" for step in stats.get("sample_remediation_steps", []))
    rows_html = "".join(
        f"<tr><td>{r.get('timestamp', '?')}</td><td>{r.get('source_ip', '?')}</td>"
        f"<td>{r.get('anomaly_type', '?')}</td><td>{r.get('risk_score', '?')}</td></tr>"
        for r in rows
    )
    avg_mttd = stats.get("avg_mttd_ms")
    avg_mttd_text = f"{avg_mttd:.1f} ms" if avg_mttd is not None else "N/A (missing ingest/detection timestamps)"
    return (
        "<html><body>"
        f"<h2>SOC Daily Digest - {stats.get('date', '?')}</h2>"
        f"<p><b>alert_count:</b> {stats.get('alert_count', 0)}<br>"
        f"<b>Average MTTD:</b> {avg_mttd_text}</p>"
        f"<p><b>Top attack types:</b></p><ul>{top_types_html}</ul>"
        f"<p><b>Sample remediation steps:</b></p><ul>{remediation_html}</ul>"
        f"<table border='1'><tr><th>timestamp</th><th>source_ip</th><th>anomaly_type</th><th>risk_score</th></tr>"
        f"{rows_html}</table>"
        "</body></html>"
    )


def send_digest_email(rows: list[dict[str, Any]], stats: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Send (or dry-run-log) the once-daily LOW-priority digest email. Called by
    src/alert_dispatcher.flush_low_priority_digest(). Never raises."""
    subject = f"[SOC Daily Digest] {stats.get('date', '?')} -- {stats.get('alert_count', 0)} LOW alerts"
    html = build_digest_email_html(rows, stats)
    return _send_html_email(config, subject, html, dry_run_label="LOW digest")
