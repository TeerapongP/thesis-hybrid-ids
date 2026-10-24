import dataclasses
import unittest

import pandas as pd

from src.ingestion import synthetic_overlay as so

ATTACK_CLASSES = {"brute_force", "port_scan", "data_exfil", "c2_beacon", "credential_stuffing"}


class SyntheticOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cols = so.feature_columns()
        cls.n_benign = int((pd.read_parquet(so.FEATURES_TEST, columns=["Label"])["Label"] == "BENIGN").sum())
        cls.scaler = so.fit_real_scaler(cls.cols)
        cls.overlay = so.generate_overlay(cls.n_benign, cls.scaler, cls.cols)

    def test_injection_ratio_at_or_under_five_percent(self):
        n = len(self.overlay)
        self.assertLessEqual(n / (self.n_benign + n), 0.05)
        self.assertGreater(n, 0)

    def test_ceiling_is_enforced_not_just_observed(self):
        with self.assertRaises(ValueError):
            so.generate_overlay(self.n_benign, self.scaler, self.cols, target_ratio=0.08)

    def test_every_row_has_unique_id_and_tz_aware_timestamp(self):
        o = self.overlay
        self.assertTrue(o["inject_id"].is_unique)
        self.assertEqual(str(o["inject_timestamp"].dt.tz), "UTC")
        self.assertFalse(o["inject_timestamp"].isna().any())
        self.assertTrue(o["synthetic"].all())

    def test_all_five_classes_present(self):
        self.assertEqual(set(self.overlay["attack_class"]), ATTACK_CLASSES)

    def test_seed_reproducible_and_seed_sensitive(self):
        again = so.generate_overlay(self.n_benign, self.scaler, self.cols, seed=42)
        pd.testing.assert_frame_equal(self.overlay, again)
        other = so.generate_overlay(self.n_benign, self.scaler, self.cols, seed=7)
        self.assertFalse(self.overlay["Flow Duration"].equals(other["Flow Duration"]))

    def test_schema_matches_real_feature_columns(self):
        self.assertEqual(list(self.overlay.columns[len(so.META_COLS):]), self.cols)
        self.assertEqual(len(self.cols), 92)

    def test_new_profile_is_added_without_touching_generation(self):
        base = so.PROFILES[0]
        sixth = dataclasses.replace(base, name="brute_force_ftp", mitre_source_label="FTP-Patator",
                                    dports=so.fixed_port(21))
        o = so.generate_overlay(self.n_benign, self.scaler, self.cols, profiles=so.PROFILES + (sixth,))
        self.assertIn("brute_force_ftp", set(o["attack_class"]))
        self.assertEqual(set(o["Destination Port"][o["attack_class"] == "brute_force_ftp"]).__len__(), 1)

    def test_beacon_is_periodic_and_scan_is_a_port_sweep(self):
        o = self.overlay
        c = o[o["campaign_id"] == o.loc[o["attack_class"] == "c2_beacon", "campaign_id"].iloc[0]]
        gaps = c["inject_timestamp"].diff().dt.total_seconds().dropna()
        self.assertLess(gaps.std() / gaps.mean(), 0.35)
        s = o[o["attack_class"] == "port_scan"]
        camp = s[s["campaign_id"] == s["campaign_id"].iloc[0]]
        self.assertGreater(camp["Destination Port"].nunique(), 0.9 * len(camp))

    def test_mttd_from_inject_timestamp(self):
        o = self.overlay
        camp = o[o["campaign_id"] == o["campaign_id"].iloc[0]]
        first, second = camp.iloc[0], camp.iloc[min(2, len(camp) - 1)]
        det = pd.DataFrame({
            "inject_id": [first["inject_id"], second["inject_id"]],
            "detected_at": [first["inject_timestamp"] + pd.Timedelta(seconds=4),
                            second["inject_timestamp"] + pd.Timedelta(seconds=1)],
        })
        flow = so.mttd_seconds(o, det, per="flow")
        self.assertEqual(sorted(flow.round(6)), [1.0, 4.0])
        c = so.mttd_seconds(o, det, per="campaign")
        self.assertEqual(len(c), 1)
        expected = min(det["detected_at"]) - camp["inject_timestamp"].min()
        self.assertAlmostEqual(c.iloc[0], expected.total_seconds(), places=6)


if __name__ == "__main__":
    unittest.main()
