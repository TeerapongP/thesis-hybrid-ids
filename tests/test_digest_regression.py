"""Offline digest delivery regressions: python -m unittest discover -s tests -p test_digest_regression.py."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from src.notifications import alert_dispatcher as dispatcher


class DigestDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        db_patch = patch.object(dispatcher, "DIGEST_DB_PATH", Path(self.temp.name) / "digest.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        self.config = {"offline_test": True}

    def add_alert(self, alert_id):
        dispatcher.accumulate_low_priority_digest({
            "alert_id": alert_id, "timestamp": "2026-09-08T00:00:00+00:00",
        })

    def pending_ids(self):
        with closing(sqlite3.connect(dispatcher.DIGEST_DB_PATH)) as conn:
            return [r[0] for r in conn.execute(
                "SELECT alert_id FROM low_priority_alerts WHERE flushed_at IS NULL ORDER BY id"
            )]

    def test_unsent_digest_remains_pending_and_can_be_retried(self):
        for status in ("error", "dry_run"):
            with self.subTest(status=status):
                self.add_alert(status)
                with patch.object(dispatcher.notify_gmail, "send_digest_email", return_value={"status": status}):
                    receipt = dispatcher.flush_low_priority_digest(self.config)
                self.assertEqual(receipt["status"], status)
                self.assertEqual(self.pending_ids(), [status])
                with patch.object(dispatcher.notify_gmail, "send_digest_email", return_value={"status": "sent"}) as send:
                    receipt = dispatcher.flush_low_priority_digest(self.config)
                self.assertEqual(receipt["status"], "flushed")
                self.assertEqual(send.call_args.args[0][0]["alert_id"], status)
                self.assertEqual(self.pending_ids(), [])

    def test_alert_arriving_during_send_waits_for_next_digest(self):
        self.add_alert("included")

        def send_with_new_arrival(rows, stats, config):
            self.assertEqual([row["alert_id"] for row in rows], ["included"])
            self.add_alert("arrived_during_send")
            return {"status": "sent"}

        with patch.object(dispatcher.notify_gmail, "send_digest_email", side_effect=send_with_new_arrival):
            receipt = dispatcher.flush_low_priority_digest(self.config)
        self.assertEqual(receipt["count"], 1)
        self.assertEqual(self.pending_ids(), ["arrived_during_send"])

    def test_send_exception_keeps_alert_pending(self):
        self.add_alert("retry")
        with patch.object(dispatcher.notify_gmail, "send_digest_email", side_effect=RuntimeError("offline failure")):
            with self.assertRaisesRegex(RuntimeError, "offline failure"):
                dispatcher.flush_low_priority_digest(self.config)
        self.assertEqual(self.pending_ids(), ["retry"])

    def test_empty_digest_does_not_send(self):
        with patch.object(dispatcher.notify_gmail, "send_digest_email") as send:
            self.assertEqual(dispatcher.flush_low_priority_digest(self.config), {"status": "no_alerts", "count": 0})
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
