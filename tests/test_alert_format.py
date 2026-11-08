from __future__ import annotations

import unittest

from src.notifications import alert_format, notify_line


class SharedFormattingTests(unittest.TestCase):
    def test_line_uses_the_shared_helpers_rather_than_copies(self):
        for name in ("technique_name", "tactic_label", "port_label", "grafana_dashboard_url"):
            with self.subTest(helper=name):
                self.assertIs(getattr(notify_line, name), getattr(alert_format, name))
        self.assertIs(notify_line.SERVICE_BY_PORT, alert_format.SERVICE_BY_PORT)

    def test_dashboard_url_tolerates_a_trailing_slash(self):
        self.assertEqual(alert_format.grafana_dashboard_url("http://g:3000/"), "http://g:3000/d/soc-main")
        self.assertEqual(alert_format.grafana_dashboard_url("http://g:3000"), "http://g:3000/d/soc-main")

    def test_line_unknown_technique_line_carries_the_shared_text(self):
        self.assertIn(alert_format.UNKNOWN_TECHNIQUE_TEXT, notify_line.UNKNOWN_TECHNIQUE_LINE)


if __name__ == "__main__":
    unittest.main()
