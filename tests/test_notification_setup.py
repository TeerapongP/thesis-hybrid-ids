# -*- coding: utf-8 -*-
"""Contracts for the two credential-handling helpers added for the output layer.

These are the only modules in src/notifications that WRITE to notification/.env, which
holds live credentials, so their failure mode is a corrupted secrets file rather than a
wrong number. Both bugs asserted below were found by running them, not by reading them.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.notifications import line_preflight, notify_line
from src.notifications.config_utils import looks_real
from src.notifications.gmail_oauth_setup import SCOPES, TARGET_KEY, read_env, write_key


class WriteKeyTests(unittest.TestCase):
    """write_key edits one line of a live credentials file; everything else must survive."""

    def _roundtrip(self, raw: bytes, key: str = "GMAIL_REFRESH_TOKEN", value: str = "1//FAKE") -> bytes:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "env"
            p.write_bytes(raw)
            write_key(p, key, value)
            return p.read_bytes()

    def test_preserves_crlf(self):
        """Found live: read_text and write_text each translate line endings by default, and
        the two translations cancelled -- so a CRLF file round-tripped correctly while the
        detection was broken, and the bug only surfaced on an LF file. Both ends now pass
        newline="". A silent CRLF->LF rewrite would show the whole secrets file as modified.
        """
        self.assertEqual(
            self._roundtrip(b"# comment\r\nGMAIL_REFRESH_TOKEN=\r\nOTHER=keep\r\n"),
            b"# comment\r\nGMAIL_REFRESH_TOKEN=1//FAKE\r\nOTHER=keep\r\n",
        )

    def test_preserves_lf(self):
        self.assertEqual(
            self._roundtrip(b"GMAIL_REFRESH_TOKEN=\nOTHER=keep\n"),
            b"GMAIL_REFRESH_TOKEN=1//FAKE\nOTHER=keep\n",
        )

    def test_appends_when_the_key_is_absent(self):
        self.assertEqual(
            self._roundtrip(b"KAFKA_BOOTSTRAP=localhost:9092\r\n"),
            b"KAFKA_BOOTSTRAP=localhost:9092\r\nGMAIL_REFRESH_TOKEN=1//FAKE\r\n",
        )

    def test_rewrites_only_the_first_match_and_keeps_comments(self):
        """The comments in notification/.env say where each credential comes from. Rebuilding
        the file from a parsed dict would drop them, so the substitution stays line-scoped.
        """
        out = self._roundtrip(
            b"# how to get this\r\nGMAIL_CLIENT_ID=abc\r\nGMAIL_REFRESH_TOKEN=old\r\n# trailing note\r\n"
        )
        self.assertIn(b"# how to get this", out)
        self.assertIn(b"# trailing note", out)
        self.assertIn(b"GMAIL_CLIENT_ID=abc", out)
        self.assertIn(b"GMAIL_REFRESH_TOKEN=1//FAKE", out)
        self.assertNotIn(b"=old", out)

    def test_does_not_match_a_key_that_merely_shares_a_prefix(self):
        out = self._roundtrip(b"GMAIL_REFRESH_TOKEN_BACKUP=keep\r\nGMAIL_REFRESH_TOKEN=\r\n")
        self.assertIn(b"GMAIL_REFRESH_TOKEN_BACKUP=keep", out)
        self.assertIn(b"GMAIL_REFRESH_TOKEN=1//FAKE", out)

    def test_requests_only_the_send_scope(self):
        """This mints a long-lived credential, so the scope it asks for is the blast radius.
        The project sends mail and never reads a mailbox.
        """
        self.assertEqual(SCOPES, ["https://www.googleapis.com/auth/gmail.send"])
        self.assertEqual(TARGET_KEY, "GMAIL_REFRESH_TOKEN")


class ReadEnvTests(unittest.TestCase):
    def test_skips_comments_and_blanks_and_keeps_values_containing_equals(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "env"
            p.write_bytes(b"# note\r\n\r\nA=1\r\nB=amqp://u:p@h:5672/soc\r\nnot_a_pair\r\n")
            env = read_env(p)
        self.assertEqual(env["A"], "1")
        # A RabbitMQ URL contains '=' in neither position but a token can; partition on the
        # FIRST '=' only, or a credential is silently truncated.
        self.assertEqual(env["B"], "amqp://u:p@h:5672/soc")
        self.assertNotIn("not_a_pair", env)
        self.assertNotIn("# note", env)


class PreflightTests(unittest.TestCase):
    def test_broadcast_mode_resolves_to_the_broadcast_endpoint(self):
        mode, url = notify_line.resolve_delivery({"line_delivery_mode": "broadcast"})
        self.assertEqual((mode, url), ("broadcast", notify_line.LINE_BROADCAST_URL))

    def test_preflight_shares_send_line_alerts_thresholds(self):
        """The check must agree with the dispatcher about what counts as configured.

        If the preflight used a looser threshold it would report PASS on a channel that
        send_line_alert still refuses, which is worse than having no check at all.
        """
        self.assertFalse(looks_real("short", min_len=20))
        self.assertFalse(looks_real("changeme-but-long-enough", min_len=20))
        self.assertTrue(looks_real("A" * 40, min_len=20))
        self.assertFalse(looks_real("", min_len=8))

    def test_send_test_is_not_the_default(self):
        """A push is visible to a real person and spends monthly quota, so it is opt-in.

        This asserts against the module's own parser. An earlier version of this test built
        an equivalent parser locally, which would have passed even if the real default
        flipped to sending.
        """
        parser = line_preflight.build_parser()
        self.assertFalse(parser.parse_args([]).send_test)
        self.assertTrue(parser.parse_args(["--send-test"]).send_test)


if __name__ == "__main__":
    unittest.main()
