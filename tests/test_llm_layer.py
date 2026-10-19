import json
import unittest
from unittest.mock import patch

from src.llm import llm_explain


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

    def test_disputed_rag_hits_abstain(self):
        hits = [{"metadata": {"technique_id": value}} for value in ("T1046", "T1498", "T1110")]
        self.assertEqual(llm_explain._technique_context(hits), (None, None))

    def test_langchain_structured_skeleton_without_live_model(self):
        from src.llm.langchain_client import LlmDraft, invoke_structured

        class FakeStructuredModel:
            def invoke(self, prompt):
                self.prompt = prompt
                return {
                    "priority": "MEDIUM",
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
        self.assertEqual(result["priority"], "MEDIUM")


if __name__ == "__main__":
    unittest.main()
