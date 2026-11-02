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


if __name__ == "__main__":
    unittest.main()
