import unittest

import numpy as np
import pandas as pd

from src.evaluation import siem_baseline as sb

KNOWN_TECHNIQUES = set(pd.read_csv(sb.MITRE_PATH)["technique_id"].dropna())
BEHAVIORAL = {f"{m}_{w}m" for m in ("conn_count", "uniq_dport_count", "fail_ratio", "repeat_ratio") for w in (1, 3, 5)}
KNOWN_COLUMNS = BEHAVIORAL | {"Flow Duration", "Flow Packets/s"}


def frame(benign_scale, n=1000, attack_value=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) * benign_scale for c in KNOWN_COLUMNS})
    df["Label"] = sb.BENIGN
    if attack_value is not None:
        atk = pd.DataFrame({c: np.full(10, attack_value) for c in KNOWN_COLUMNS})
        atk["Label"] = "DDoS"
        df = pd.concat([df, atk], ignore_index=True)
    return df


class RuleTableTests(unittest.TestCase):
    def test_ids_unique_and_techniques_in_mitre_mapping(self):
        ids = [r.rule_id for r in sb.RULES]
        self.assertEqual(len(ids), len(set(ids)))
        for r in sb.RULES:
            self.assertIn(r.technique_id, KNOWN_TECHNIQUES, r.rule_id)

    def test_implementable_rules_read_real_columns_with_valid_ops(self):
        self.assertGreaterEqual(len(sb.implementable()), 5)
        for r in sb.implementable():
            self.assertTrue(r.conditions)
            for c in r.conditions:
                self.assertIn(c.column, KNOWN_COLUMNS)
                self.assertIn(c.op, sb.OPS)
                self.assertTrue(0 < c.quantile < 1)

    def test_unimplementable_rules_say_why_and_have_no_conditions(self):
        for r in sb.RULES:
            if not r.implementable:
                self.assertFalse(r.conditions)
                self.assertTrue(r.reason)


class ThresholdProvenanceTests(unittest.TestCase):
    def test_thresholds_follow_the_train_split(self):
        low, high = sb.derive_thresholds(frame(1.0)), sb.derive_thresholds(frame(2.0))
        self.assertNotEqual(low, high)

    def test_thresholds_use_benign_rows_only(self):
        self.assertEqual(sb.derive_thresholds(frame(1.0, attack_value=50.0)),
                         sb.derive_thresholds(frame(1.0)))

    def test_scoring_a_different_test_set_leaves_thresholds_unchanged(self):
        th = sb.derive_thresholds(frame(1.0))
        before = dict(th)
        sb.evaluate(frame(5.0, attack_value=9.0), th)
        self.assertEqual(th, before)

    def test_flood_rows_far_above_benign_trip_connection_flood(self):
        th = sb.derive_thresholds(frame(1.0))
        hits, y = sb.evaluate(frame(1.0, attack_value=100.0), th)
        self.assertEqual(int(hits.loc[y, "R3"].sum()), 10)


if __name__ == "__main__":
    unittest.main()
