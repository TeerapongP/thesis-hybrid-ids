"""The analyst-facing LINE text, asserted on what an analyst can see.

These tests read the rendered string rather than the builder's internals, so reformatting
the message is free and dropping a field an analyst needs is not.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.notifications.notify_line import (  # noqa: E402
    build_line_message,
    port_label,
    risk_suffix,
    tactic_label,
    technique_name,
)

CONFIG = {"line_grafana_base_url": "http://localhost:3000"}

HIGH_RECORD = {
    "priority": "HIGH",
    "source_ip": "172.16.0.1",
    "dst_ip": "192.168.10.50",
    "dst_port": 80,
    "mitre_tactic": "Impact",
    "technique_id": "T1499.002",
    "explanation_th": "เครื่องต้นทาง 172.16.0.1 เชื่อมต่อผิดปกติไปยัง 192.168.10.50 พอร์ต 80",
}


class BuildLineMessage(unittest.TestCase):
    def test_carries_every_incident_field(self):
        message = build_line_message(HIGH_RECORD, CONFIG)
        for expected in (
            "SOC SECURITY ALERT",
            "HIGH SEVERITY",
            "Source IP: 172.16.0.1",
            "Destination IP: 192.168.10.50",
            "Destination Port: 80 (HTTP)",
            "MITRE ATT&CK Tactic: Impact",
            "T1499.002",
            "http://localhost:3000/d/soc-main",
        ):
            self.assertIn(expected, message)

    def test_names_the_technique_from_the_knowledge_base(self):
        """The id alone sends an analyst to look it up. The name is the point."""
        message = build_line_message(HIGH_RECORD, CONFIG)
        self.assertIn("Service Exhaustion Flood", message)

    def test_no_markdown_syntax_because_line_renders_plain_text(self):
        message = build_line_message(HIGH_RECORD, CONFIG)
        self.assertNotIn("`", message)
        self.assertNotIn("**", message)

    def test_medium_is_not_told_to_act_immediately(self):
        record = dict(HIGH_RECORD, priority="MEDIUM")
        message = build_line_message(record, CONFIG)
        self.assertIn("MEDIUM SEVERITY", message)
        self.assertNotIn("immediate investigation", message)

    def test_missing_explanation_still_renders_the_details(self):
        record = dict(HIGH_RECORD, explanation_th=None)
        message = build_line_message(record, CONFIG)
        self.assertIn("Source IP: 172.16.0.1", message)
        self.assertIn("Incident Details", message)

    def test_unknown_fields_render_as_a_question_mark_not_the_word_none(self):
        message = build_line_message({"priority": "HIGH"}, CONFIG)
        self.assertIn("Source IP: ?", message)
        self.assertNotIn("None", message)


class RiskScore(unittest.TestCase):
    def test_shown_beside_the_severity_so_two_highs_can_be_ordered(self):
        message = build_line_message(dict(HIGH_RECORD, risk_score=87), CONFIG)
        self.assertIn("Severity:", message)
        self.assertIn("(risk 87/100)", message)

    def test_absent_score_leaves_the_severity_line_alone(self):
        message = build_line_message(HIGH_RECORD, CONFIG)
        self.assertNotIn("risk", message)

    def test_unusable_value_is_dropped_rather_than_crashing_the_alert(self):
        """A malformed score must not be what stops a HIGH alert reaching an analyst."""
        self.assertEqual(risk_suffix("not a number"), "")


class AbsentTechnique(unittest.TestCase):
    """50 of the 100 measured alerts carry no technique: the RAG hits disagreed.

    The message has to read as "not determined" rather than as a rendering failure, so a
    bare question mark against a MITRE label is a bug even though nothing raises.
    """

    def test_says_it_was_not_determined_instead_of_printing_a_question_mark(self):
        record = dict(HIGH_RECORD, technique_id=None, mitre_tactic=None)
        message = build_line_message(record, CONFIG)
        self.assertIn("MITRE ATT&CK: ไม่ระบุ", message)
        self.assertNotIn("MITRE Technique: ?", message)
        self.assertNotIn("Tactic: ?", message)

    def test_the_rest_of_the_incident_block_survives(self):
        record = dict(HIGH_RECORD, technique_id=None, mitre_tactic=None)
        message = build_line_message(record, CONFIG)
        self.assertIn("Destination Port: 80 (HTTP)", message)
        self.assertIn("Severity:", message)


class TacticLabel(unittest.TestCase):
    def test_rendered_as_mitre_writes_it(self):
        self.assertEqual(tactic_label("impact"), "Impact")
        self.assertEqual(tactic_label("credential-access"), "Credential Access")

    def test_and_stays_lowercase(self):
        self.assertEqual(tactic_label("command-and-control"), "Command and Control")

    def test_absent_tactic_is_none(self):
        self.assertIsNone(tactic_label(None))


class PortLabel(unittest.TestCase):
    def test_known_service_is_named(self):
        self.assertEqual(port_label(443), "443 (HTTPS)")

    def test_unknown_port_is_bare(self):
        self.assertEqual(port_label(49152), "49152")

    def test_absent_port_does_not_raise(self):
        self.assertEqual(port_label(None), "?")


class TechniqueName(unittest.TestCase):
    def test_absent_id_returns_none(self):
        self.assertIsNone(technique_name(None))

    def test_id_outside_the_knowledge_base_returns_none(self):
        """11 techniques are indexed. An id outside them gets no invented name."""
        self.assertIsNone(technique_name("T9999"))


if __name__ == "__main__":
    unittest.main()
