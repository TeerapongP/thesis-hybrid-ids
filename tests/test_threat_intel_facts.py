"""Whether OpenCTI reaches the explanation layer at all.

Before `threat_intel_facts`, `enrichment` rode on every alert and nothing read it, so the
honest answer to "does OpenCTI affect any output" was no. These tests pin the two halves of
the fix: a match reaches `alert_context`, where `build_prompt` copies it into FACTS, and a
non-match leaves the context untouched so the deployed M2 prompts stay byte-identical.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.e2e_replay import threat_intel_facts  # noqa: E402

MATCHED = {
    "matched": True,
    "observables_checked": ["99.83.190.102", "192.168.10.50"],
    "indicators": [{"id": "indicator--1", "name": "99.83.190.102", "confidence": 75,
                    "valid_until": "2027-01-01T00:00:00Z"}],
    "related": [{"id": "malware--1", "type": "Malware", "name": "Mirai",
                 "relationship": "indicates"}],
    "source": "opencti",
}


class NoMatch(unittest.TestCase):
    """The unmatched case has to produce nothing, not an empty block.

    A constant "no threat intel" block would enter all 100 prompts and move every M2 number
    for a field that never varies on this corpus, which holds 2 ipv4-addr indicators out of
    55,219.
    """

    def test_unmatched_enrichment_contributes_nothing(self):
        self.assertIsNone(threat_intel_facts({"matched": False, "observables_checked": ["1.2.3.4"]}))

    def test_absent_enrichment_contributes_nothing(self):
        self.assertIsNone(threat_intel_facts(None))
        self.assertIsNone(threat_intel_facts({}))


class Match(unittest.TestCase):
    def test_carries_the_indicator_name_and_confidence(self):
        facts = threat_intel_facts(MATCHED)
        self.assertEqual(facts["matched_indicators"],
                         [{"name": "99.83.190.102", "confidence": 75}])

    def test_carries_the_graph_walk_context(self):
        """The graph walk is why enrich.py queries OpenCTI twice. Dropping it wastes the call."""
        facts = threat_intel_facts(MATCHED)
        self.assertEqual(facts["related_objects"],
                         [{"type": "Malware", "name": "Mirai", "relationship": "indicates"}])

    def test_says_whether_the_answer_came_from_cache(self):
        facts = threat_intel_facts(dict(MATCHED, source="cache"))
        self.assertEqual(facts["source"], "cache")

    def test_lists_are_capped_so_one_observable_cannot_crowd_out_the_alert(self):
        noisy = dict(
            MATCHED,
            observables_checked=[f"10.0.0.{n}" for n in range(20)],
            indicators=[{"name": f"i{n}", "confidence": 50} for n in range(20)],
            related=[{"type": "Malware", "name": f"m{n}", "relationship": "indicates"}
                     for n in range(20)],
        )
        facts = threat_intel_facts(noisy)
        self.assertEqual(len(facts["observables_checked"]), 5)
        self.assertEqual(len(facts["matched_indicators"]), 3)
        self.assertEqual(len(facts["related_objects"]), 3)


class ReachesTheLlmFacts(unittest.TestCase):
    """The point of putting it in alert_context: build_prompt copies that whole dict into FACTS."""

    def test_the_prompt_names_the_matched_indicator(self):
        try:
            from src.llm import llm_explain
        except ImportError as exc:  # langchain lives only in the uv .venv
            self.skipTest(f"llm_explain unavailable under this interpreter: {exc}")
        ctx = {
            "source_ip": "99.83.190.102", "dst_ip": "192.168.10.50", "dst_port": 80,
            "timestamp": "2017-07-07T09:00:00", "ingested_at": "2017-07-07T09:00:00",
            "detected_at": "2017-07-07T09:00:00", "rf_predicted_label": "DoS Hulk",
            "fused_score": 0.99, "fused_prediction": 1, "confidence": 0.98,
            "model_used": "random_forest", "threat_intel": threat_intel_facts(MATCHED),
        }
        prompt = llm_explain.build_prompt(ctx, [], 0.99, 1, 0.98)
        self.assertIn("Mirai", prompt)
        self.assertIn("threat_intel", prompt)


if __name__ == "__main__":
    unittest.main()
