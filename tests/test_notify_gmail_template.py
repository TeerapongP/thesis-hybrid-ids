from __future__ import annotations

import base64
import email
import unittest
from email import policy
from unittest.mock import MagicMock, patch

from src.notifications import notify_gmail
from src.notifications.notify_gmail import build_mime_message

SENDER = "soc@example.test"


def parse(message):
    return email.message_from_bytes(message.as_bytes(), policy=policy.default)


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


class SendPathTests(unittest.TestCase):
    def test_send_hands_the_gmail_api_a_multipart_message(self):
        try:
            import googleapiclient  # noqa: F401
        except ImportError:
            self.skipTest("google client libraries are not installed in this interpreter")
        service = MagicMock()
        service.users().messages().send().execute.return_value = {"id": "abc"}
        config = {"gmail_sender_address": SENDER}
        with patch.object(notify_gmail, "_gmail_creds_look_real", return_value=True),              patch.object(notify_gmail, "_build_gmail_service", return_value=service):
            result = notify_gmail._send_email(config, "s", "test", text_body="t", html_body="<p>h</p>")
        self.assertEqual(result, {"status": "sent", "message_id": "abc"})
        raw = service.users().messages().send.call_args.kwargs["body"]["raw"]
        sent = email.message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)
        self.assertEqual(sent.get_content_type(), "multipart/alternative")


if __name__ == "__main__":
    unittest.main()
