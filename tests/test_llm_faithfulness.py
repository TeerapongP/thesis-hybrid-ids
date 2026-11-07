import copy
import json
import unittest
from pathlib import Path

from src.evaluation import eval_llm_faithfulness as m2

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "llm_faithfulness_records.json").read_text(encoding="utf-8"))
KB = m2.load_kb()


def verdicts(rec):
    return [c["verdict"] for c in m2.score_record(rec, KB)["claims"]]


class FaithfulnessScoringTests(unittest.TestCase):
    def test_known_good_output_passes_every_check(self):
        rec = FIXTURE["good"]
        checks = m2.score_record(rec, KB)
        self.assertTrue(checks["entity"]["pass"])
        self.assertEqual(checks["grounding"]["prose_ungrounded"], [])
        self.assertFalse(any(v in ("unsupported", "contradicted") for v in verdicts(rec)))
        self.assertFalse(checks["defects"]["example_echo"])

    def test_output_naming_entities_absent_from_facts_is_flagged(self):
        """Renamed from test_recorded_few_shot_copy_is_flagged. The hardcoded-example-entity
        check it also asserted is gone: examples now render with the alert's own values, so
        there is no generic entity left to leak. What this fixture still proves is the part
        that matters and is unchanged -- an output naming entities absent from FACTS fails."""
        rec = FIXTURE["fewshot_copy"]
        checks = m2.score_record(rec, KB)
        self.assertFalse(checks["entity"]["pass"])
        foreign = set(checks["entity"]["foreign"])
        self.assertIn("ip:172.16.33.9", foreign)
        self.assertIn("port:8443", foreign)
        self.assertIn("technique_id:T1190", foreign)
        self.assertTrue(checks["defects"]["waf_in_remediation"])
        self.assertIn("contradicted", verdicts(rec))
        self.assertIn("T1190", checks["grounding"]["prose_technique_ids"])
        self.assertEqual(checks["grounding"]["prose_ungrounded"], ["T1190"])

    def test_example_echo_is_measured_and_flagged(self):
        """The risk rendering traded into: right entities, example's sentence.

        Rendering the few-shot examples per alert means a copied value is the CORRECT value,
        so the entity guard cannot catch template reuse. This is what catches it instead.
        """
        rec = copy.deepcopy(FIXTURE["good"])
        rec["rendered_example_explanations"] = [rec["output"]["explanation_th"]]
        checks = m2.score_record(rec, KB)
        self.assertEqual(checks["defects"]["example_echo_ratio"], 1.0)
        self.assertTrue(checks["defects"]["example_echo"])

        rec2 = copy.deepcopy(FIXTURE["good"])
        rec2["rendered_example_explanations"] = ["ข้อความตัวอย่างที่ไม่เกี่ยวข้องกันเลยแม้แต่น้อย"]
        checks2 = m2.score_record(rec2, KB)
        self.assertLess(checks2["defects"]["example_echo_ratio"], m2.EXAMPLE_ECHO_THRESHOLD)
        self.assertFalse(checks2["defects"]["example_echo"])

    def test_example_echo_without_rendered_examples_reports_zero(self):
        """Absent examples must read as unmeasured-zero, not as a silent pass of a real check."""
        rec = copy.deepcopy(FIXTURE["good"])
        rec.pop("rendered_example_explanations", None)
        self.assertEqual(m2.example_echo_ratio(rec), 0.0)

    def test_mutated_good_output_is_caught_by_each_rule(self):
        base = FIXTURE["good"]

        def mutate(**fields):
            rec = copy.deepcopy(base)
            rec["output"].update(fields)
            return rec

        reformatted = mutate(explanation_th="โจมตีจาก 172.16.0.1 ไปยัง 192.168.10.50 พอร์ต ๘๐")
        self.assertTrue(m2.score_record(reformatted, KB)["entity"]["thai_digits"])

        wrong_ip = mutate(explanation_th="โจมตีจาก 172.16.0.2 ไปยัง 192.168.10.50 พอร์ต 80")
        self.assertIn("ip:172.16.0.2", m2.score_record(wrong_ip, KB)["entity"]["foreign"])
        self.assertEqual(verdicts(wrong_ip)[0], "contradicted")

        invented = mutate(remediation=["ติดตั้ง Snort เพื่อตรวจจับ"])
        self.assertEqual(verdicts(invented)[-1], "unsupported")

        wrong_tactic = mutate(mitre_tactic="discovery")
        self.assertFalse(m2.score_record(wrong_tactic, KB)["grounding"]["tactic_matches_kb"])

        omitted = mutate(explanation_th="เกิดการโจมตีแบบ DoS", remediation=["ตรวจสอบระบบ"])
        entity = m2.score_record(omitted, KB)["entity"]
        self.assertFalse(entity["pass"])
        self.assertTrue(entity["corruption_free"])

    def test_generic_advice_is_undecidable_not_supported(self):
        rec = copy.deepcopy(FIXTURE["good"])
        rec["output"]["remediation"] = ["ตรวจสอบระบบเพิ่มเติม"]
        self.assertEqual(verdicts(rec)[-1], "undecidable")

    def test_insufficient_evidence_regex(self):
        rec = copy.deepcopy(FIXTURE["good"])
        rec["output"]["explanation_th"] = "หลักฐานจาก RAG ไม่เพียงพอที่จะระบุเทคนิค"
        self.assertTrue(m2.check_defects(rec)["states_insufficient_evidence"])

    def test_hostname_entity_is_required_when_present(self):
        rec = copy.deepcopy(FIXTURE["good"])
        rec["alert_context"]["affected_asset"] = "AUTH-SVR01.campus.local"
        rec["output"]["explanation_th"] += " ที่ AUTH-SVR01"
        entity = m2.score_record(rec, KB)["entity"]
        self.assertIn("hostname:AUTH-SVR01.campus.local", entity["missing"])

    def test_wilson_interval_bounds(self):
        lo, hi = m2.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2)


def mapping_run_record(**overrides):
    """A DDoS alert whose retrieval says T1499.002 while the table says T1498."""
    rec = copy.deepcopy(FIXTURE["good"])
    rec["technique_source"] = "mapping"
    rec["true_label"] = "DDoS"
    rec["alert_context"]["rf_predicted_label"] = "DDoS"
    rec["ctx_technique_id"] = "T1498"
    rec["pred_label_mapping_technique_id"] = "T1498"
    rec["rag_hits"] = [{"rank": 1, "id": "x", "document": "d", "metadata": {"technique_id": "T1499.002"}}]
    rec["output"]["technique_id"] = "T1498"
    rec["output"]["explanation_th"] = ("พบการโจมตี DDoS จากเครื่อง 172.16.0.1 ไปยัง 192.168.10.50 "
                                       "ที่พอร์ต 80 ตรงกับเทคนิค T1498")
    rec.update(overrides)
    return rec


def finished(rec):
    return {**rec, "status": "ok", "checks": m2.score_record(rec, KB), "llm_metrics": {"wall_s": 1.0}}


class TechniqueProvenanceTests(unittest.TestCase):
    def test_supplied_technique_is_not_a_hallucination_when_retrieval_missed_it(self):
        grounding = m2.score_record(mapping_run_record(), KB)["grounding"]
        self.assertEqual(grounding["prose_technique_ids"], ["T1498"])
        self.assertEqual(grounding["prose_ungrounded"], [])

    def test_a_technique_that_is_neither_supplied_nor_retrieved_is_still_flagged(self):
        rec = mapping_run_record()
        rec["output"]["explanation_th"] += " และ T1046"
        self.assertEqual(m2.score_record(rec, KB)["grounding"]["prose_ungrounded"], ["T1046"])

    def test_mapped_technique_absent_from_hits_is_measured_as_a_retrieval_miss(self):
        grounding = m2.score_record(mapping_run_record(), KB)["grounding"]
        self.assertFalse(grounding["mapped_pred_technique_in_hits"])
        hit = mapping_run_record()
        hit["rag_hits"][0]["metadata"]["technique_id"] = "T1498"
        self.assertTrue(m2.score_record(hit, KB)["grounding"]["mapped_pred_technique_in_hits"])

    def test_naming_the_retrieved_technique_instead_of_the_supplied_one_is_contradicted(self):
        rec = mapping_run_record()
        rec["output"]["explanation_th"] = rec["output"]["explanation_th"].replace("T1498", "T1499.002")
        checks = m2.score_record(rec, KB)
        self.assertIn("technique_id:T1499.002", checks["entity"]["foreign"])
        reasons = [r for c in checks["claims"] for r in c["reasons"]]
        self.assertIn("technique_differs_from_mapping:T1499.002", reasons)

    def test_mapping_run_reports_the_two_provenance_dependent_checks_as_not_applicable(self):
        s = m2.summarize([finished(mapping_run_record())])
        self.assertEqual(s["technique_source"], "mapping")
        self.assertIsNone(s["result_technique_in_hits"])
        self.assertIsNone(s["consensus_matches_label_mapping"])
        self.assertEqual(s["retrieval_hits_contain_mapped_technique"]["den"], 1)
        self.assertEqual(s["retrieval_hits_contain_mapped_technique"]["num"], 0)

    def test_records_without_a_source_are_rag_and_keep_both_guards(self):
        rec = copy.deepcopy(FIXTURE["good"])
        rec["rag_hits"][0]["metadata"]["technique_id"] = "T1499.002"
        s = m2.summarize([finished(rec)])
        self.assertEqual(s["technique_source"], "rag")
        self.assertEqual(s["result_technique_in_hits"]["den"], 1)
        self.assertIsNotNone(s["consensus_matches_label_mapping"])

    def test_a_run_cannot_mix_sources(self):
        mixed = [finished(mapping_run_record()), finished(copy.deepcopy(FIXTURE["good"]))]
        with self.assertRaises(ValueError):
            m2.summarize(mixed)

    def test_metrics_csv_states_provenance_and_marks_inapplicable_rows(self):
        import csv
        import tempfile

        s = m2.summarize([finished(mapping_run_record())])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            m2.write_metrics_csv(s, path)
            rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        by_key = {r[0]: r for r in rows if r}
        self.assertEqual(by_key["# technique_source"][1], "mapping")
        for key in ("result_technique_in_hits", "consensus_matches_label_mapping"):
            self.assertEqual(by_key[key][3], "n/a")
            self.assertIn("n/a under technique_source=mapping", by_key[key][6])
        self.assertIn("retrieval_hits_contain_mapped_technique", by_key)

    def test_report_states_provenance(self):
        import tempfile

        records = [finished(mapping_run_record())]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.txt"
            m2.write_report(m2.summarize(records), records, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Technique source for this run: mapping", text)


if __name__ == "__main__":
    unittest.main()
