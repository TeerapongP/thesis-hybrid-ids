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

    def test_few_shots_share_no_entities_with_fixtures(self):
        keys = ("source_ip", "dst_ip", "dst_port", "affected_asset", "technique_id")
        fixture_values = set()
        for path in FIXTURES.glob("alert_context_*.json"):
            alert = json.loads(path.read_text(encoding="utf-8"))
            alert["technique_id"] = (alert.get("rag_top_chunk") or {}).get("technique_id")
            fixture_values |= {alert.get(k) for k in keys}
        shot_values = {shot["input"].get(k) for shot in llm_explain.FEW_SHOTS for k in keys}
        self.assertEqual((fixture_values & shot_values) - {None}, set())

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
