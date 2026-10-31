import unittest

import numpy as np
import pandas as pd

from src.evaluation import fusion_loao_calibrated as fc


def make_scores(rng, n, attack_frac):
    y = (rng.random(n) < attack_frac).astype(int)
    rf = np.clip(rng.normal(0.2 + 0.6 * y, 0.2), 0, 1)
    ae = np.clip(rng.normal(0.1 + 0.5 * y, 0.15), 0, 2)
    return rf, ae, y


class CalibrationLeakageTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_stratified(self):
        labels = ["BENIGN"] * 400 + ["DoS Hulk"] * 60 + ["Bot"] * 40
        df = pd.DataFrame({"Label": labels})
        fit, cal = fc.split_fit_calibration(df, 0.25)
        self.assertEqual(set(fit) & set(cal), set())
        self.assertEqual(len(fit) + len(cal), len(df))
        self.assertEqual(int((df.Label.iloc[cal] == "Bot").sum()), 10)

    def test_maps_depend_only_on_calibration_slice(self):
        rng = np.random.default_rng(0)
        rf, ae, y = make_scores(rng, 600, 0.3)
        n_cal = 400
        base = fc.calibrate_eval_scores(rf, ae, y[:n_cal], n_cal)
        rf2, ae2 = rf.copy(), ae.copy()
        rf2[n_cal:] = rng.random(len(rf2) - n_cal)
        ae2[n_cal:] = rng.random(len(ae2) - n_cal) * 3
        changed = fc.calibrate_eval_scores(rf2, ae2, y[:n_cal], n_cal)
        grid = np.linspace(0, 2, 101)
        np.testing.assert_array_equal(base[2].predict(grid), changed[2].predict(grid))
        np.testing.assert_array_equal(base[3].predict(grid), changed[3].predict(grid))
        np.testing.assert_array_equal(base[0][:n_cal], changed[0][:n_cal])

    def test_test_labels_cannot_reach_the_fit(self):
        rng = np.random.default_rng(1)
        rf, ae, y = make_scores(rng, 500, 0.3)
        n_cal = 300
        a = fc.calibrate_eval_scores(rf, ae, y[:n_cal], n_cal)
        self.assertEqual(len(y[:n_cal]), n_cal)
        self.assertEqual(a[2].X_thresholds_.min() >= rf[:n_cal].min(), True)
        self.assertEqual(a[2].X_thresholds_.max() <= rf[:n_cal].max(), True)

    def test_out_of_range_test_scores_are_clipped_not_extrapolated(self):
        rng = np.random.default_rng(2)
        rf, ae, y = make_scores(rng, 400, 0.3)
        iso = fc.fit_isotonic(ae[:300], y[:300])
        hi = iso.predict([ae[:300].max() + 100.0])[0]
        self.assertEqual(hi, iso.predict([ae[:300].max()])[0])
        self.assertTrue(0.0 <= hi <= 1.0)

    def test_calibrated_output_is_monotone_probability(self):
        rng = np.random.default_rng(3)
        rf, ae, y = make_scores(rng, 800, 0.4)
        iso = fc.fit_isotonic(rf, y)
        out = iso.predict(np.sort(rf))
        self.assertTrue(np.all(np.diff(out) >= 0))
        self.assertTrue(out.min() >= 0 and out.max() <= 1)


class CombinationTests(unittest.TestCase):
    def test_combine_values(self):
        p_rf, p_ae = np.array([0.0, 0.5, 0.9, 1.0]), np.array([0.0, 0.5, 0.1, 0.2])
        np.testing.assert_allclose(fc.combine(p_rf, p_ae, "max"), [0.0, 0.5, 0.9, 1.0])
        np.testing.assert_allclose(fc.combine(p_rf, p_ae, "noisy_or"), [0.0, 0.75, 0.91, 1.0])
        np.testing.assert_allclose(fc.combine(p_rf, p_ae, "mean"), [0.0, 0.5, 0.5, 0.6])

    def test_noisy_or_bounds(self):
        p_rf, p_ae = np.array([0.3, 0.8]), np.array([0.6, 0.1])
        nor = fc.combine(p_rf, p_ae, "noisy_or")
        self.assertTrue(np.all(nor >= np.maximum(p_rf, p_ae)))
        self.assertTrue(np.all(nor <= 1.0))

    def test_unknown_combiner_rejected(self):
        with self.assertRaises(ValueError):
            fc.combine([0.1], [0.2], "xor")

    def test_candidate_scores_deployed_is_raw_max(self):
        df = pd.DataFrame({"rf_prob": [0.01, 0.9], "ae_norm": [0.2, 0.05], "p_rf": [0.1, 0.9], "p_ae": [0.3, 0.2]})
        c = fc.candidate_scores(df)
        np.testing.assert_allclose(c["deployed"], [0.2, 0.9])
        np.testing.assert_allclose(c["cal_noisy_or"], [1 - 0.9 * 0.7, 1 - 0.1 * 0.8])


class SweepTests(unittest.TestCase):
    def test_brier(self):
        self.assertAlmostEqual(fc.brier([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(fc.brier([0.5, 0.5], [1, 0]), 0.25)
        self.assertAlmostEqual(fc.brier([3.0], [1]), 0.0)

    def test_sweep_matches_brute_force(self):
        rng = np.random.default_rng(4)
        per_cat = {}
        for c in ("a", "b", "c"):
            s, _, y = make_scores(rng, 300, 0.3)
            per_cat[c] = (s, y)
        grid = np.linspace(-0.1, 1.0, 50)
        ev = fc.evaluate_candidate(per_cat, grid)
        s = np.concatenate([v[0] for v in per_cat.values()])
        y = np.concatenate([v[1] for v in per_cat.values()])
        best = max(
            (2 * ((s > t) & (y == 1)).sum() / (((s > t).sum()) + y.sum()) if (s > t).sum() else 0.0) for t in grid
        )
        self.assertAlmostEqual(ev["POOLED"]["oracle_f1"], best, places=9)
        self.assertGreaterEqual(ev["POOLED"]["oracle_f1"], ev["POOLED"]["nested_f1"] - 1e-12)

    def test_nested_threshold_excludes_held_out_category(self):
        rng = np.random.default_rng(5)
        s1, _, y1 = make_scores(rng, 400, 0.3)
        s2, _, y2 = make_scores(rng, 400, 0.3)
        grid = np.linspace(-0.1, 1.0, 50)
        base = fc.evaluate_candidate({"a": (s1, y1), "b": (s2, y2)}, grid)
        flipped = fc.evaluate_candidate({"a": (s1, y1), "b": (1 - s2, y2)}, grid)
        self.assertEqual(base["a"]["nested_threshold"] != flipped["a"]["nested_threshold"], True)
        self.assertEqual(base["a"]["oracle_threshold"], flipped["a"]["oracle_threshold"])


if __name__ == "__main__":
    unittest.main()
