import unittest
from unittest.mock import MagicMock, patch

from src.notifications import notify_gmail
from src.notifications.notify_gmail import build_digest_email_html, build_immediate_email_html


class GmailHtmlEscapingTests(unittest.TestCase):
    def test_expected_transport_failure_returns_an_error_result(self):
        config = {"gmail_sender_address": "sender@example.test"}
        with patch.object(notify_gmail, "_gmail_creds_look_real", return_value=True):
            with patch.object(notify_gmail, "_build_gmail_service", side_effect=OSError("network unavailable")):
                result = notify_gmail._send_html_email(config, "subject", "body", "test")
        self.assertEqual(result["status"], "error")

    def test_subject_with_an_embedded_header_is_refused_without_sending(self):
        try:
            import googleapiclient  # noqa: F401
        except ImportError:
            self.skipTest("google client libraries are not installed in this interpreter")
        config = {"gmail_sender_address": "sender@example.test"}
        service = MagicMock()
        subject = "[SOC ALERT] x @ 1.2.3.4" + chr(13) + chr(10) + "Bcc: other@example.test"
        with patch.object(notify_gmail, "_gmail_creds_look_real", return_value=True):
            with patch.object(notify_gmail, "_build_gmail_service", return_value=service):
                result = notify_gmail._send_html_email(config, subject, "body", "test")
        self.assertEqual(result["status"], "error")
        self.assertIn("HeaderParseError", result["error"])
        service.users.assert_not_called()

    def test_immediate_alert_values_are_rendered_as_text(self):
        html = build_immediate_email_html({
            "priority": "<priority>", "alert_id": "<alert-id>", "timestamp": "<timestamp>",
            "source_ip": "<source-ip>", "dst_port": "<dst-port>", "affected_asset": "<asset>",
            "technique_id": "<technique>", "mitre_tactic": "<tactic>", "risk_score": "<risk>",
            "explanation_th": "<explanation>", "remediation": ["<remediation>"]
        })
        for value in ("priority", "alert-id", "timestamp", "source-ip", "dst-port", "asset",
                      "technique", "tactic", "risk", "explanation", "remediation"):
            with self.subTest(value=value):
                self.assertIn(f"&lt;{value}&gt;", html)
                self.assertNotIn(f"<{value}>", html)

    def test_digest_values_are_rendered_as_text(self):
        html = build_digest_email_html(
            [{"anomaly_type": "<anomaly>", "source_ip": "<source-ip>",
              "timestamp": "<timestamp>", "risk_score": "<risk>"}],
            {"date": "<date>", "alert_count": "<count>",
             "top_attack_types": [("<attack-type>", "<attack-count>")],
             "sample_remediation_steps": ["<remediation>"]},
        )
        for value in ("date", "count", "attack-type", "attack-count", "remediation", "timestamp",
                      "source-ip", "anomaly", "risk"):
            with self.subTest(value=value):
                self.assertIn(f"&lt;{value}&gt;", html)
                self.assertNotIn(f"<{value}>", html)


if __name__ == "__main__":
    unittest.main()
