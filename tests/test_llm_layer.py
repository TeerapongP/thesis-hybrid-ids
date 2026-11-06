import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm import llm_explain

FIXTURES = Path(__file__).parent / "fixtures"


class LlmLayerTests(unittest.TestCase):
    def test_explain_alert_overrides_model_supplied_facts(self):
        alert = {"dst_ip": "192.0.2.10", "affected_asset": "host.example"}
        hits = [
            {"metadata": {"technique_id": "T1046"}, "document": "one"},
            {"metadata": {"technique_id": "T1046"}, "document": "two"},
        ]
        raw = json.dumps({
            "priority": "HIGH", "explanation_th": "คำอธิบาย", "remediation": ["ตรวจสอบ"],
            "technique_id": "T9999", "confidence": 0.01,
        })
        with patch.object(llm_explain, "call_llm", return_value=raw):
            result = llm_explain.explain_alert(alert, hits, 0.9, 1, 0.97, {})
        self.assertEqual(result["technique_id"], "T1046")
        self.assertEqual(result["confidence"], 0.97)
        self.assertEqual(result["affected_asset"], "host.example")
        self.assertEqual(result["priority"], "HIGH")

    def test_priority_comes_from_rule_not_model(self):
        raw = json.dumps({"priority": "HIGH", "explanation_th": "คำอธิบาย", "remediation": ["ตรวจสอบ"]})
        with patch.object(llm_explain, "call_llm", return_value=raw):
            result = llm_explain.explain_alert({"dst_ip": "192.0.2.10"}, [], 0.55, 1, 0.55, {})
        self.assertEqual(result["priority"], "MEDIUM")

    def test_derive_priority_matches_fixture_expectations(self):
        for path in FIXTURES.glob("alert_context_*.json"):
            alert = json.loads(path.read_text(encoding="utf-8"))
            expected = json.loads(path.with_name(path.name.replace("alert_context", "llm_output")).read_text(encoding="utf-8"))
            with self.subTest(fixture=path.name):
                self.assertEqual(
                    llm_explain.derive_priority(alert["fused_score"], alert["fused_prediction"]),
                    expected["priority"],
                )

    def test_derive_priority_boundaries(self):
        high = llm_explain.PRIORITY_HIGH_MIN_SCORE
        self.assertEqual(llm_explain.derive_priority(high, 1), "HIGH")
        self.assertEqual(llm_explain.derive_priority(high - 0.01, 1), "MEDIUM")
        self.assertEqual(llm_explain.derive_priority(0.06, 1), "MEDIUM")
        self.assertEqual(llm_explain.derive_priority(0.99, 0), "LOW")

    def test_fallback_branch_uses_its_own_cutoff(self):
        """The fallback branch could not reach HIGH under the shared 0.80 cutoff.

        Its score is `(if_norm + ae_norm) / 2`, which never exceeded 0.7031 over 250,000
        test rows, while RF's branch emits a probability reaching 1.0. One cutoff over two
        incompatible scales meant no unsupervised-only detection could ever page an
        analyst -- the exact alerts the cascade exists to produce.
        """
        fb = llm_explain.PRIORITY_HIGH_MIN_SCORE_FALLBACK
        self.assertLess(fb, llm_explain.PRIORITY_HIGH_MIN_SCORE)
        self.assertEqual(llm_explain.derive_priority(fb, 1, "if_ae_fallback"), "HIGH")
        self.assertEqual(llm_explain.derive_priority(fb - 0.01, 1, "if_ae_fallback"), "MEDIUM")
        # the same score on RF's branch is MEDIUM, because RF's cutoff still applies there
        self.assertEqual(llm_explain.derive_priority(fb, 1, "random_forest"), "MEDIUM")
        # not an attack stays LOW on either branch, whatever the score
        self.assertEqual(llm_explain.derive_priority(0.99, 0, "if_ae_fallback"), "LOW")

    def test_unknown_branch_takes_the_stricter_cutoff(self):
        """A corrupted or absent branch label must not make the system page more often."""
        between = (llm_explain.PRIORITY_HIGH_MIN_SCORE_FALLBACK
                   + llm_explain.PRIORITY_HIGH_MIN_SCORE) / 2
        for label in (None, "", "unknown", "nonsense"):
            with self.subTest(model_used=label):
                self.assertEqual(llm_explain.derive_priority(between, 1, label), "MEDIUM")

    def test_null_technique_removes_the_technique_clause_from_both_shots(self):
        """No example may assert a technique for an alert that has none.

        With a null technique, TECHNIQUE_ID rendered as the stand-in "ไม่ระบุเทคนิค" and
        shot 1's clause became "ลักษณะตรงกับเทคนิค ไม่ระบุเทคนิค" -- broken Thai that one
        alert in the measured run copied verbatim into its explanation.
        """
        facts = {"alert_context": {"source_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
                                   "dst_port": 80, "rf_predicted_label": "PortScan"},
                 "priority": "MEDIUM", "technique_id": None, "mitre_tactic": None}
        for shot in llm_explain.render_few_shots(facts):
            with self.subTest(label=shot["input"].get("rf_predicted_label")):
                self.assertNotIn("ไม่ระบุเทคนิค", json.dumps(shot, ensure_ascii=False))
                self.assertNotIn("ตรงกับเทคนิค", shot["output"]["explanation_th"])
                self.assertIsNone(shot["input"]["technique_id"])

    def test_known_technique_still_shows_it(self):
        """The null-technique path must not strip the clause when there IS a technique."""
        facts = {"alert_context": {"source_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
                                   "dst_port": 80, "rf_predicted_label": "PortScan"},
                 "priority": "HIGH", "technique_id": "T1046", "mitre_tactic": "discovery"}
        shots = llm_explain.render_few_shots(facts)
        self.assertIn("T1046", shots[0]["output"]["explanation_th"])
        # and the module-level templates were not mutated by the null-technique branch
        self.assertIn("{{TECHNIQUE_ID}}",
                      llm_explain.FEW_SHOT_TEMPLATES[0]["output"]["explanation_th"])

    def test_fallback_cutoff_matches_its_recorded_derivation(self):
        """Pin the constant to the run that produced it, so a silent edit fails here.

        reports/priority_cutoff_derivation.txt records p99.5 of the fallback branch's
        train-BENIGN distribution as 0.4349. If that file is regenerated and the number
        moves, this test is the reminder that the deployed constant has to move with it.
        """
        report = (llm_explain.ROOT if hasattr(llm_explain, "ROOT") else Path(__file__).resolve().parents[1])
        path = Path(report) / "reports" / "priority_cutoff_derivation.txt"
        if not path.exists():
            self.skipTest("derivation report not present in this checkout")
        text = path.read_text(encoding="utf-8")
        self.assertIn("p99.5", text)
        self.assertIn(f"{llm_explain.PRIORITY_HIGH_MIN_SCORE_FALLBACK:.4f}", text)

    def test_disputed_rag_hits_abstain(self):
        hits = [{"metadata": {"technique_id": value}} for value in ("T1046", "T1498", "T1110")]
        self.assertEqual(llm_explain._technique_context(hits), (None, None))

    def test_langchain_structured_skeleton_without_live_model(self):
        from src.llm.langchain_client import LlmDraft, invoke_structured

        class FakeStructuredModel:
            def invoke(self, prompt):
                self.prompt = prompt
                return {
                    "explanation_th": "หลักฐานยังไม่เพียงพอ",
                    "remediation": ["ตรวจสอบเหตุการณ์"],
                }

        class FakeChatModel:
            def with_structured_output(self, schema, method):
                self.schema = schema
                self.method = method
                return FakeStructuredModel()

        fake_model = FakeChatModel()
        result = invoke_structured("FACTS={}", {"backend": "ollama"}, fake_model)
        self.assertIs(fake_model.schema, LlmDraft)
        self.assertEqual(fake_model.method, "json_schema")
        self.assertEqual(result["remediation"], ["ตรวจสอบเหตุการณ์"])
        self.assertNotIn("priority", LlmDraft.model_json_schema()["properties"])

    def test_few_shot_templates_carry_no_real_entities(self):
        """The TEMPLATES must stay entity-free; realistic values there were copied into outputs."""
        keys = ("source_ip", "dst_ip", "dst_port", "affected_asset", "technique_id")
        fixture_values = set()
        for path in FIXTURES.glob("alert_context_*.json"):
            alert = json.loads(path.read_text(encoding="utf-8"))
            alert["technique_id"] = (alert.get("rag_top_chunk") or {}).get("technique_id")
            fixture_values |= {alert.get(k) for k in keys}
        shot_values = {shot["input"].get(k) for shot in llm_explain.FEW_SHOT_TEMPLATES for k in keys}
        self.assertEqual((fixture_values & shot_values) - {None}, set())

    def test_rendered_few_shots_carry_this_alerts_values_and_no_placeholders(self):
        """Rendering is the root fix for the leak: the model copied {{SRC_IP}} literally on
        100% of alerts, so there must be no placeholder left in what reaches the prompt."""
        facts = {
            "alert_context": {"source_ip": "10.0.0.7", "dst_ip": "10.0.0.9", "dst_port": 445,
                              "rf_predicted_label": "PortScan"},
            "confidence": 0.93, "technique_id": "T1046", "mitre_tactic": "Discovery",
        }
        rendered = llm_explain.render_few_shots(facts)
        blob = json.dumps(rendered, ensure_ascii=False)
        self.assertNotIn("{{", blob)
        for token in ("10.0.0.7", "445", "T1046"):
            self.assertIn(token, blob)
        # the templates themselves must be untouched, so the next alert renders cleanly
        self.assertIn("{{SRC_IP}}", json.dumps(llm_explain.FEW_SHOT_TEMPLATES, ensure_ascii=False))

    def test_rendered_few_shots_tolerate_missing_context(self):
        """A missing field must render as Thai text, not an empty string that reads as truncated."""
        rendered = llm_explain.render_few_shots({"alert_context": {}})
        blob = json.dumps(rendered, ensure_ascii=False)
        self.assertNotIn("{{", blob)
        self.assertNotIn('""', blob)

    def test_remediation_is_deduplicated_and_capped(self):
        from src.llm.langchain_client import MAX_REMEDIATION, LlmDraft

        looped = ["ปิดกั้นต้นทาง"] * 25 + [f"ข้อ {i}" for i in range(10)]
        draft = LlmDraft.model_validate({"explanation_th": "x", "remediation": looped})
        self.assertEqual(draft.remediation, ["ปิดกั้นต้นทาง", "ข้อ 0", "ข้อ 1", "ข้อ 2", "ข้อ 3"][:MAX_REMEDIATION])
        self.assertEqual(LlmDraft.model_json_schema()["properties"]["remediation"]["maxItems"], MAX_REMEDIATION)

    def test_ollama_model_penalizes_repetition(self):
        from src.llm.langchain_client import build_chat_model

        model = build_chat_model({
            "backend": "ollama", "model": "m", "temperature": 0.1,
            "base_url": "http://localhost:11434", "max_tokens": 512, "timeout_s": 5,
        })
        self.assertGreater(model.repeat_penalty, 1.0)


if __name__ == "__main__":
    unittest.main()
