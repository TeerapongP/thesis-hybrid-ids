"""Contract tests for SHAP attribution over the deployed `confidence_weighted` cascade.

These were skipping while shap_explain was unmigrated. They are restored and rewritten for
the cascade, and two of them changed meaning rather than just names:

  * The old suite asserted that RF and AE attributions came from DIFFERENT feature spaces --
    true under the two-population design, and the thing that made the fallback inexplicable.
    The unified population has one space, so that test is replaced by its opposite.
  * A new test asserts the fallback attribution actually DEPENDS on the Isolation Forest.
    The old module passed np.zeros as the IF score, which was valid under the OR rule and
    would silently corrupt every fallback row under the cascade. Nothing caught that; this
    does.
"""

import unittest

import numpy as np
import pandas as pd

from src.modeling import fusion_layer
from src.modeling.shap_explain import (
    FusionExplainer,
    TEST_PATH,
    risk_score_from_fused,
)
from src.notifications import alert_dispatcher

SCAN_CHUNK = 20_000
SCAN_CAP = 120_000
WANT_FALLBACK = 3


class ShapExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test = pd.read_parquet(TEST_PATH)
        cls.explainer = FusionExplainer()
        cols = cls.explainer.columns

        # The fallback branch decides ~0.2% of in-dataset rows, so it has to be hunted for.
        # Taking the first N rows would almost certainly yield none and the test would only
        # ever exercise the RF path -- the easy half.
        fallback, scanned = [], 0
        rng = np.random.default_rng(42)
        order = rng.permutation(len(test))
        while scanned < SCAN_CAP and len(fallback) < WANT_FALLBACK:
            idx = order[scanned:scanned + SCAN_CHUNK]
            chunk = test.iloc[idx]
            prob = 1.0 - cls.explainer.rf.predict_proba(
                chunk[cols].to_numpy(dtype=np.float64)
            )[:, cls.explainer.benign_idx]
            undecided = np.abs(prob - 0.5) * 2 < fusion_layer.CONFIDENCE_THRESHOLD
            if undecided.any():
                fallback.append(chunk[undecided])
            scanned += SCAN_CHUNK
        assert fallback, f"no fallback-branch row found in {scanned:,} rows"
        fb = pd.concat(fallback).head(WANT_FALLBACK)

        rf_rows = pd.concat([
            test[test["Label"] == "PortScan"].head(3),
            test[test["Label"] == "FTP-Patator"].head(3),
            test[test["Label"] == "BENIGN"].head(3),
        ])
        cls.rows = pd.concat([rf_rows, fb])
        cls.records = cls.explainer.explain(cls.rows, top_k=5)

    def test_record_shape(self):
        for r in self.records:
            self.assertEqual(set(r), {"risk_score", "model_used", "fused_score", "top_features"})
            self.assertIsInstance(r["risk_score"], int)
            self.assertTrue(0 <= r["risk_score"] <= 100)
            self.assertEqual(len(r["top_features"]), 5)
            for f in r["top_features"]:
                self.assertEqual(set(f), {"feature", "value", "shap_value", "direction"})
                self.assertEqual(f["direction"],
                                 "increases_risk" if f["shap_value"] > 0 else "decreases_risk")
            mags = [abs(f["shap_value"]) for f in r["top_features"]]
            self.assertEqual(mags, sorted(mags, reverse=True))

    def test_both_branches_are_exercised(self):
        """A suite that only ever sees one branch would not be testing the cascade."""
        seen = {r["model_used"] for r in self.records}
        self.assertEqual(seen, set(fusion_layer.MODEL_USED_VALUES))

    def test_risk_score_matches_dispatcher(self):
        for r in self.records:
            if r["fused_score"] > 1.0:
                continue
            assembled = alert_dispatcher.assemble_alert_record(
                {}, r["fused_score"], {"priority": "HIGH"}, r["model_used"], "a1"
            )
            self.assertEqual(r["risk_score"], assembled["risk_score"])
        self.assertEqual(risk_score_from_fused(3.7), 100)
        self.assertEqual(risk_score_from_fused(0.5), 50)

    def test_all_members_share_one_feature_space(self):
        """Replaces the old two-space assertion, which the unified population made obsolete.

        Every attributed feature must come from the single 94-column space, and the AE's
        declared input width must match it -- if they ever diverge, the composite attribution
        silently explains a different function than the one the cascade thresholds.
        """
        space = set(self.explainer.columns)
        self.assertEqual(len(space), 94)
        self.assertEqual(self.explainer.background.shape[1], 94)
        for r in self.records:
            self.assertLessEqual({f["feature"] for f in r["top_features"]}, space)

    def test_model_used_and_fused_score_match_the_deployed_rule(self):
        x = self.rows[self.explainer.columns].to_numpy(dtype=np.float64)
        rf_prob = 1.0 - self.explainer.rf.predict_proba(x)[:, self.explainer.benign_idx]
        fused, _, _, used = fusion_layer.fuse_with_metadata(
            rf_prob, self.explainer._if_raw(x), self.explainer._ae_mse(x)
        )
        self.assertEqual([r["model_used"] for r in self.records], used.tolist())
        np.testing.assert_allclose([r["fused_score"] for r in self.records], fused)

    def test_fallback_attribution_depends_on_isolation_forest(self):
        """Guards the exact defect that disabled the old module.

        It passed np.zeros as the IF score. Under the OR rule IF could not affect the
        decision, so that was harmless; under the cascade the fallback is
        (if_norm + ae_norm) / 2, so zeroing IF changes the explained function. If the
        fallback score were computable without the IF, this assertion would fail.
        """
        fb = [i for i, r in enumerate(self.records) if r["model_used"] == "if_ae_fallback"]
        self.assertTrue(fb, "no fallback row available to test")
        x = self.rows.iloc[fb][self.explainer.columns].to_numpy(dtype=np.float64)
        with_if = self.explainer._fallback_fn(x)
        ae_only = fusion_layer.normalize_ae_score(self.explainer._ae_mse(x)) / 2.0
        self.assertFalse(
            np.allclose(with_if, ae_only),
            "the fallback score does not depend on the Isolation Forest, so either the rule "
            "or this test is wrong",
        )

    def test_shap_is_additive_for_each_branch(self):
        for member in fusion_layer.MODEL_USED_VALUES:
            sel = [i for i, r in enumerate(self.records) if r["model_used"] == member][:2]
            self.assertTrue(sel, f"no rows decided by {member}")
            gap = self.explainer.additivity_gap(member, self.rows.iloc[sel])
            self.assertLess(gap, 1e-6, f"{member} attributions are not additive (gap {gap:.2e})")


if __name__ == "__main__":
    unittest.main()
