from __future__ import annotations

import base64
import email
import os
import tempfile
import unittest
from email import policy
from pathlib import Path
from string import Template
from unittest.mock import MagicMock, patch

from src.notifications import alert_format, notify_gmail
from src.notifications.notify_gmail import (
    SEVERITY_STYLE,
    UNKNOWN_SEVERITY_STYLE,
    build_immediate_email_html,
    build_immediate_email_text,
    build_mime_message,
    email_template_values,
    send_immediate_email,
)

SENDER = "soc@example.test"
CONFIG = {"line_grafana_base_url": "http://grafana.test:3000/", "gmail_sender_address": SENDER}

RECORD = {
    "alert_id": "8f2c1e0a-0000-4000-8000-000000000001",
    "priority": "MEDIUM",
    "timestamp": "2017-07-05T09:48:00",
    "risk_score": 7,
    "source_ip": "172.16.0.1",
    "dst_ip": "192.168.10.50",
    "dst_port": 80,
    "anomaly_type": "DoS Slowhttptest",
    "technique_id": "T1499.002",
    "mitre_tactic": "impact",
    "explanation_th": "พบพฤติกรรมผิดปกติ",
    "remediation": ["Block the source", "Review web server logs"],
}

HOSTILE = "<script>alert(1)</script>\"'"


def parse(message):
    return email.message_from_bytes(message.as_bytes(), policy=policy.default)


def temp_file(testcase: unittest.TestCase, text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    testcase.addCleanup(os.unlink, handle.name)
    return Path(handle.name)


class MimeStructureTests(unittest.TestCase):
    def test_both_bodies_make_a_multipart_alternative_with_text_first(self):
        parsed = parse(build_mime_message(SENDER, "s", text_body="plain ไทย", html_body="<p>html ไทย</p>"))
        self.assertEqual(parsed.get_content_type(), "multipart/alternative")
        parts = list(parsed.iter_parts())
        self.assertEqual([p.get_content_type() for p in parts], ["text/plain", "text/html"])
        self.assertIn("plain ไทย", parts[0].get_content())
        self.assertIn("<p>html ไทย</p>", parts[1].get_content())

    def test_text_only_and_html_only_are_single_parts(self):
        text_only = parse(build_mime_message(SENDER, "s", text_body="plain"))
        html_only = parse(build_mime_message(SENDER, "s", html_body="<p>x</p>"))
        self.assertEqual(text_only.get_content_type(), "text/plain")
        self.assertEqual(html_only.get_content_type(), "text/html")

    def test_no_body_is_refused(self):
        with self.assertRaises(ValueError):
            build_mime_message(SENDER, "s")

    def test_thai_content_is_sent_as_seven_bit_safe_bytes(self):
        raw = build_mime_message(SENDER, "แจ้งเตือน", text_body="ไทย", html_body="<p>ไทย</p>").as_bytes()
        self.assertTrue(raw.isascii())

    def test_a_line_break_in_the_subject_is_refused(self):
        with self.assertRaises(ValueError):
            build_mime_message(SENDER, "x\r\nBcc: other@example.test", text_body="t")


class TemplateRenderingTests(unittest.TestCase):
    def test_renders_for_each_priority_with_its_own_colours(self):
        for priority, style in SEVERITY_STYLE.items():
            with self.subTest(priority=priority):
                html = build_immediate_email_html({**RECORD, "priority": priority}, CONFIG)
                self.assertNotIn("${", html)
                self.assertIn(f"{priority} SEVERITY", html)
                for colour in style:
                    self.assertIn(colour, html)

    def test_unknown_priority_gets_the_neutral_style(self):
        html = build_immediate_email_html({**RECORD, "priority": "URGENT"}, CONFIG)
        for colour in UNKNOWN_SEVERITY_STYLE:
            self.assertIn(colour, html)

    def test_every_style_is_distinct(self):
        styles = list(SEVERITY_STYLE.values()) + [UNKNOWN_SEVERITY_STYLE]
        self.assertEqual(len(set(styles)), len(styles))

    def test_values_come_from_the_record(self):
        html = build_immediate_email_html(RECORD, CONFIG)
        for expected in (
            RECORD["alert_id"], RECORD["timestamp"], "172.16.0.1", "192.168.10.50",
            "80 (HTTP)", "T1499.002", "(Impact)", "7/100", "พบพฤติกรรมผิดปกติ",
            "DoS Slowhttptest detected from 172.16.0.1", "http://grafana.test:3000/d/soc-main",
            notify_gmail.DEFAULT_SYSTEM_NAME,
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, html)

    def test_technique_name_matches_the_line_channel(self):
        name = alert_format.technique_name("T1499.002")
        self.assertTrue(name)
        self.assertIn(name, build_immediate_email_html(RECORD, CONFIG))

    def test_an_alert_with_no_technique_says_so(self):
        html = build_immediate_email_html({**RECORD, "technique_id": None, "mitre_tactic": None}, CONFIG)
        self.assertIn(alert_format.UNKNOWN_TECHNIQUE_TEXT, html)

    def test_configured_system_name_replaces_the_default(self):
        html = build_immediate_email_html(RECORD, {**CONFIG, "soc_system_name": "Lab SOC"})
        self.assertIn("Lab SOC", html)
        self.assertNotIn(notify_gmail.DEFAULT_SYSTEM_NAME, html)

    def test_markup_and_quotes_in_every_field_are_escaped(self):
        record = {k: (HOSTILE if isinstance(v, str) else v) for k, v in RECORD.items()}
        record["remediation"] = [HOSTILE]
        record["priority"] = "MEDIUM"
        html = build_immediate_email_html(record, {**CONFIG, "soc_system_name": HOSTILE})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&quot;", html)

    def test_a_quote_in_the_dashboard_url_cannot_break_out_of_the_href(self):
        config = {**CONFIG, "line_grafana_base_url": 'http://x/" onclick="evil()'}
        html = build_immediate_email_html(RECORD, config)
        self.assertNotIn('" onclick="evil()', html)

    def test_remediation_is_markup_with_each_item_escaped(self):
        record = {**RECORD, "remediation": ["<b>one</b>", "two & three"]}
        actions = email_template_values(record, CONFIG)["requiredActions"]
        self.assertEqual(actions.count("<div"), 2)
        self.assertIn("&lt;b&gt;one&lt;/b&gt;", actions)
        self.assertIn("two &amp; three", actions)
        self.assertNotIn("<b>", actions)

    def test_empty_remediation_says_so_instead_of_leaving_a_blank_box(self):
        actions = email_template_values({**RECORD, "remediation": []}, CONFIG)["requiredActions"]
        self.assertIn(notify_gmail.NO_REMEDIATION_TEXT, actions)

    def test_a_dollar_sign_in_a_value_is_not_expanded(self):
        html = build_immediate_email_html({**RECORD, "explanation_th": "cost $5 ${sourceIp}"}, CONFIG)
        self.assertIn("cost $5 ${sourceIp}", html)

    def test_template_and_values_agree_on_every_placeholder(self):
        text = notify_gmail.EMAIL_TEMPLATE_PATH.read_text(encoding="utf-8")
        wanted = {m.group("braced") for m in Template.pattern.finditer(text) if m.group("braced")}
        self.assertEqual(wanted, set(email_template_values(RECORD, CONFIG)))

    def test_a_placeholder_with_no_value_raises(self):
        path = temp_file(self, "hello ${noSuchField}")
        with patch.object(notify_gmail, "EMAIL_TEMPLATE_PATH", path):
            with self.assertRaises(KeyError):
                build_immediate_email_html(RECORD, CONFIG)


class PlainTextBodyTests(unittest.TestCase):
    def test_text_body_carries_the_incident_and_the_actions(self):
        text = build_immediate_email_text(RECORD, CONFIG)
        for expected in ("MEDIUM SEVERITY", "172.16.0.1", "80 (HTTP)", "T1499.002", "7/100",
                         "1. Block the source", "2. Review web server logs",
                         "http://grafana.test:3000/d/soc-main"):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        self.assertNotIn("<", text)


class FallbackTests(unittest.TestCase):
    def _send(self, template_path=None):
        with patch.object(notify_gmail, "_send_email", return_value={"status": "sent"}) as send:
            with patch.object(notify_gmail, "EMAIL_TEMPLATE_PATH", template_path or notify_gmail.EMAIL_TEMPLATE_PATH):
                result = send_immediate_email(RECORD, CONFIG)
        return result, send.call_args

    def test_normal_send_carries_both_bodies(self):
        result, call = self._send()
        self.assertEqual(result["body_format"], "multipart")
        self.assertIn("<html", call.kwargs["html_body"])
        self.assertIn("Incident Details", call.kwargs["text_body"])

    def test_a_template_failure_sends_text_only_instead_of_raising(self):
        path = temp_file(self, "${noSuchField}")
        with self.assertLogs("alert_dispatcher.notify_gmail", level="ERROR"):
            result, call = self._send(path)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body_format"], "text_only")
        self.assertIsNone(call.kwargs["html_body"])
        self.assertTrue(call.kwargs["text_body"])

    def test_a_missing_template_file_also_falls_back(self):
        with self.assertLogs("alert_dispatcher.notify_gmail", level="ERROR"):
            result, _ = self._send(Path("no/such/template.html"))
        self.assertEqual(result["body_format"], "text_only")

    def test_the_real_message_for_a_medium_alert_is_multipart_alternative(self):
        captured = {}

        def fake_send(config, subject, label, *, html_body=None, text_body=None):
            captured["message"] = build_mime_message(SENDER, subject, text_body=text_body, html_body=html_body)
            return {"status": "sent"}

        with patch.object(notify_gmail, "_send_email", fake_send):
            send_immediate_email(RECORD, CONFIG)
        parsed = parse(captured["message"])
        self.assertEqual([p.get_content_type() for p in parsed.iter_parts()], ["text/plain", "text/html"])


class SendPathTests(unittest.TestCase):
    def test_send_hands_the_gmail_api_a_multipart_message(self):
        try:
            import googleapiclient  # noqa: F401
        except ImportError:
            self.skipTest("google client libraries are not installed in this interpreter")
        service = MagicMock()
        service.users().messages().send().execute.return_value = {"id": "abc"}
        config = {"gmail_sender_address": SENDER}
        with patch.object(notify_gmail, "_gmail_creds_look_real", return_value=True), \
             patch.object(notify_gmail, "_build_gmail_service", return_value=service):
            result = notify_gmail._send_email(config, "s", "test", text_body="t", html_body="<p>h</p>")
        self.assertEqual(result, {"status": "sent", "message_id": "abc"})
        raw = service.users().messages().send.call_args.kwargs["body"]["raw"]
        sent = email.message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)
        self.assertEqual(sent.get_content_type(), "multipart/alternative")


if __name__ == "__main__":
    unittest.main()
