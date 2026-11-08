import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm import llm_explain

FIXTURES = Path(__file__).parent / "fixtures"


class LlmLayerTests(unittest.TestCase):
    def test_explain_alert_overrides_model_supplied_facts(self):
        """A technique the model invented must never survive into the result.

        Asserted under both sources rather than under whichever is deployed, so flipping
        DEFAULT_TECHNIQUE_SOURCE cannot quietly stop testing the override. T9999 is not a
        real technique and reaching an analyst would send them to look up nothing.
        """
        alert = {"dst_ip": "192.0.2.10", "affected_asset": "host.example",
                 "rf_predicted_label": "PortScan"}
        hits = [
            {"metadata": {"technique_id": "T1046"}, "document": "one"},
            {"metadata": {"technique_id": "T1046"}, "document": "two"},
        ]
        raw = json.dumps({
            "priority": "HIGH", "explanation_th": "คำอธิบาย", "remediation": ["ตรวจสอบ"],
            "technique_id": "T9999", "confidence": 0.01,
        })
        for source, expected in (("rag", "T1046"), ("mapping", "T1046")):
            with self.subTest(source=source):
                with patch.object(llm_explain, "call_llm", return_value=raw):
                    result = llm_explain.explain_alert(alert, hits, 0.9, 1, 0.97, {},
                                                       technique_source=source)
                self.assertEqual(result["technique_id"], expected)
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


class PriorityConfigRecordTests(unittest.TestCase):
    """models/llm_layer_config.json is the record of this layer's decisions.

    It stated `fused_score >= 0.80 -> HIGH`, a single cutoff, and "fused_score and
    fused_prediction only" for three weeks after none of those were true. A config file
    that describes a rule the code does not implement is worse than no file: it is the
    thing a reader trusts instead of reading the code.
    """

    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "models" / "llm_layer_config.json"
        if not path.exists():
            self.skipTest("llm_layer_config.json not present in this checkout")
        self.rule = json.loads(path.read_text(encoding="utf-8"))["priority_rule"]

    def test_recorded_cutoffs_match_the_deployed_constants(self):
        self.assertEqual(self.rule["high_min_fused_score_random_forest"],
                         llm_explain.PRIORITY_HIGH_MIN_SCORE)
        self.assertEqual(self.rule["high_min_fused_score_if_ae_fallback"],
                         llm_explain.PRIORITY_HIGH_MIN_SCORE_FALLBACK)

    def test_record_names_model_used_as_an_input(self):
        self.assertIn("model_used", self.rule["inputs"])


class TechniqueSourceTests(unittest.TestCase):
    """The mapping source names the technique from the RF-predicted label via the MITRE table."""

    KB_TACTICS = json.loads(llm_explain.KB_PATH.read_text(encoding="utf-8"))["techniques"]

    def test_predicted_label_drives_the_technique_not_other_context(self):
        ctx = {"rf_predicted_label": "SSH-Patator", "Label": "PortScan", "true_label": "PortScan"}
        self.assertEqual(llm_explain.resolve_technique("mapping", ctx, [])[0], "T1110.001")

    def test_ddos_resolves_to_network_denial_of_service(self):
        self.assertEqual(llm_explain.technique_from_label("DDoS")[0], "T1498")

    def test_both_patators_resolve_to_password_guessing(self):
        for label in ("FTP-Patator", "SSH-Patator"):
            with self.subTest(label=label):
                self.assertEqual(llm_explain.technique_from_label(label)[0], "T1110.001")

    def test_benign_unknown_and_missing_labels_yield_no_technique(self):
        for label in ("BENIGN", "NotARealAttack", "", None):
            with self.subTest(label=label):
                self.assertEqual(llm_explain.technique_from_label(label), (None, None))

    def test_tactic_comes_from_the_kb_in_its_own_casing(self):
        for label, technique_id in (("Bot", "T1071.001"), ("PortScan", "T1046"), ("DDoS", "T1498")):
            with self.subTest(label=label):
                got_id, tactic = llm_explain.technique_from_label(label)
                self.assertEqual(got_id, technique_id)
                self.assertEqual(tactic, self.KB_TACTICS[technique_id]["tactics"][0])
                self.assertEqual(tactic, tactic.lower())
                self.assertNotIn(" ", tactic)

    def test_mapping_source_ignores_what_retrieval_found(self):
        hits = [{"metadata": {"technique_id": "T1499.002"}}] * 3
        ctx = {"rf_predicted_label": "DDoS"}
        self.assertEqual(llm_explain.resolve_technique("rag", ctx, hits)[0], "T1499.002")
        self.assertEqual(llm_explain.resolve_technique("mapping", ctx, hits)[0], "T1498")

    def test_mapping_source_names_a_technique_even_when_retrieval_abstains(self):
        disputed = [{"metadata": {"technique_id": value}} for value in ("T1046", "T1498", "T1110")]
        ctx = {"rf_predicted_label": "PortScan"}
        self.assertEqual(llm_explain.resolve_technique("rag", ctx, disputed), (None, None))
        self.assertEqual(llm_explain.resolve_technique("mapping", ctx, disputed)[0], "T1046")

    def test_the_deployed_default_is_the_mitre_table(self):
        """Chosen on the measured ablation, so the assertion names the winner, not a preference.

        Pinned because flipping it moves every M2 figure and five sections of chapter 4: the
        default must not drift without someone editing this line and noticing why.
        """
        self.assertEqual(llm_explain.DEFAULT_TECHNIQUE_SOURCE, "mapping")

    def test_omitting_the_source_uses_the_deployed_default(self):
        hits = [{"metadata": {"technique_id": "T1046"}}] * 2
        ctx = {"rf_predicted_label": "DDoS", "dst_ip": "192.0.2.10"}
        prompt = llm_explain.build_prompt(ctx, hits, 0.9, 1, 0.9)
        explicit = llm_explain.build_prompt(ctx, hits, 0.9, 1, 0.9,
                                            llm_explain.DEFAULT_TECHNIQUE_SOURCE)
        self.assertEqual(prompt, explicit)
        # DDoS maps to T1498, while the hits vote T1046, so this also proves which won.
        self.assertIn('"technique_id": "T1498"', prompt)

    def test_prompt_carries_the_mapped_technique_under_mapping(self):
        hits = [{"metadata": {"technique_id": "T1499.002"}}] * 3
        ctx = {"rf_predicted_label": "DDoS", "dst_ip": "192.0.2.10"}
        prompt = llm_explain.build_prompt(ctx, hits, 0.9, 1, 0.9, "mapping")
        self.assertIn('"technique_id": "T1498"', prompt)
        self.assertIn('"mitre_tactic": "impact"', prompt)

    def test_explain_alert_reports_the_mapped_technique(self):
        raw = json.dumps({"explanation_th": "คำอธิบาย", "remediation": ["ตรวจสอบ"]})
        ctx = {"rf_predicted_label": "PortScan", "dst_ip": "192.0.2.10"}
        with patch.object(llm_explain, "call_llm", return_value=raw):
            result = llm_explain.explain_alert(ctx, [], 0.9, 1, 0.9, {}, technique_source="mapping")
        self.assertEqual((result["technique_id"], result["mitre_tactic"]), ("T1046", "discovery"))

    def test_unknown_source_raises_before_any_llm_call(self):
        ctx = {"rf_predicted_label": "DDoS", "dst_ip": "192.0.2.10"}
        with patch.object(llm_explain, "call_llm", side_effect=AssertionError("LLM was called")):
            with self.assertRaises(ValueError):
                llm_explain.resolve_technique("mappng", ctx, [])
            with self.assertRaises(ValueError):
                llm_explain.build_prompt(ctx, [], 0.9, 1, 0.9, "mappng")
            with self.assertRaises(ValueError):
                llm_explain.explain_alert(ctx, [], 0.9, 1, 0.9, {}, technique_source="mappng")


class RetrievalContextTests(unittest.TestCase):
    ctx = {"rf_predicted_label": "FTP-Patator", "dst_ip": "192.0.2.10"}
    marker_hits = [{"metadata": {"technique_id": "T1110.001"}, "document": "WHOLE-CHUNK-MARKER"}] * 3

    def test_the_default_is_the_current_behaviour(self):
        self.assertEqual(llm_explain.DEFAULT_RETRIEVAL_CONTEXT, "hits")

    def test_omitting_the_context_equals_hits_and_pastes_the_chunks(self):
        omitted = llm_explain.build_prompt(self.ctx, self.marker_hits, 0.9, 1, 0.9)
        explicit = llm_explain.build_prompt(self.ctx, self.marker_hits, 0.9, 1, 0.9,
                                            retrieval_context="hits")
        self.assertEqual(omitted, explicit)
        self.assertIn("WHOLE-CHUNK-MARKER", omitted)
        self.assertNotIn("technique_reference", omitted)

    def test_technique_kb_replaces_the_chunks_with_the_kb_text_for_the_named_technique(self):
        prompt = llm_explain.build_prompt(self.ctx, self.marker_hits, 0.9, 1, 0.9,
                                          retrieval_context="technique_kb")
        self.assertNotIn("WHOLE-CHUNK-MARKER", prompt)
        self.assertNotIn("rag_hits", prompt)
        self.assertIn("Password Policies", prompt)
        self.assertIn("failed SSH login attempts", prompt)

    def test_technique_kb_changes_only_the_retrieval_entry_of_facts(self):
        def facts(context):
            prompt = llm_explain.build_prompt(self.ctx, self.marker_hits, 0.9, 1, 0.9,
                                              retrieval_context=context)
            return prompt, json.loads(prompt.split("FACTS=", 1)[1])

        hits_prompt, hits_facts = facts("hits")
        kb_prompt, kb_facts = facts("technique_kb")
        for key in ("alert_context", "fused_score", "priority", "technique_id", "mitre_tactic"):
            self.assertEqual(hits_facts[key], kb_facts[key])
        self.assertEqual(hits_prompt.split("FACTS=", 1)[0], kb_prompt.split("FACTS=", 1)[0])

    def test_technique_kb_with_no_technique_carries_no_retrieval_text(self):
        benign = {"rf_predicted_label": "BENIGN", "dst_ip": "192.0.2.10"}
        prompt = llm_explain.build_prompt(benign, self.marker_hits, 0.9, 1, 0.9,
                                          retrieval_context="technique_kb")
        self.assertIsNone(json.loads(prompt.split("FACTS=", 1)[1])["technique_reference"])
        self.assertNotIn("WHOLE-CHUNK-MARKER", prompt)

    def test_every_kb_reference_is_bounded_and_made_of_whole_items(self):
        for technique_id, entry in llm_explain._kb_techniques().items():
            with self.subTest(technique=technique_id):
                ref = llm_explain.technique_reference(technique_id)
                self.assertLessEqual(len(ref["detection"]), llm_explain.MAX_DETECTION_ANALYTICS)
                self.assertLessEqual(len(ref["mitigations"]), llm_explain.MAX_KB_MITIGATIONS)
                for line in ref["detection"]:
                    self.assertIn(line, entry["detection"])
                for item in ref["mitigations"]:
                    summary = item.get("summary")
                    if summary:
                        self.assertTrue(summary.endswith("."))
                        self.assertLessEqual(len(summary), llm_explain.MAX_MITIGATION_SUMMARY_CHARS)
                self.assertNotIn("�", json.dumps(ref, ensure_ascii=False))

    def test_network_telemetry_analytics_come_first(self):
        detection = ("Title line\n- Analytic 1: host thing [Log sources: WinEventLog:Sysmon]\n"
                     "- Analytic 2: flow thing [Log sources: NSM:Flow]")
        self.assertEqual(llm_explain._detection_analytics(detection)[0],
                         "flow thing [Log sources: NSM:Flow]")

    def test_a_mitigation_without_a_short_clean_first_sentence_is_shown_by_name_only(self):
        self.assertIsNone(llm_explain._first_sentence("No terminal punctuation here"))
        self.assertIsNone(llm_explain._first_sentence("x" * 300 + "."))
        self.assertEqual(llm_explain._first_sentence("Block it. Then do more."), "Block it.")

    def test_unknown_context_raises_before_any_llm_call(self):
        with patch.object(llm_explain, "call_llm", side_effect=AssertionError("LLM was called")):
            with self.assertRaises(ValueError):
                llm_explain.build_prompt(self.ctx, [], 0.9, 1, 0.9, retrieval_context="technique")
            with self.assertRaises(ValueError):
                llm_explain.explain_alert(self.ctx, [], 0.9, 1, 0.9, {}, retrieval_context="technique")

    def test_the_entity_guard_allows_what_the_prompt_showed_and_nothing_more(self):
        """powershell.exe is in T1059's KB detection text and in no retrieved chunk."""
        hits = [{"metadata": {"technique_id": "T1059"}, "document": "plain chunk"}] * 2
        raw = json.dumps({"explanation_th": "พบการรัน powershell.exe", "remediation": ["ตรวจสอบ"]})
        with patch.object(llm_explain, "call_llm", return_value=raw):
            shown = llm_explain.explain_alert(self.ctx, hits, 0.9, 1, 0.9, {},
                                              technique_source="rag", retrieval_context="technique_kb")
            self.assertEqual(shown["technique_id"], "T1059")
            with self.assertRaises(ValueError):
                llm_explain.explain_alert(self.ctx, hits, 0.9, 1, 0.9, {},
                                          technique_source="rag", retrieval_context="hits")
