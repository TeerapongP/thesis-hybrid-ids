import unittest

import numpy as np
import pandas as pd

from src.modeling import fusion_layer
from src.modeling.shap_explain import FusionExplainer, ROOT, risk_score_from_fused
from src.notifications import alert_dispatcher


class ShapExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rf_frame = pd.read_parquet(ROOT / "data/features/cicids2017_features_test.parquet")
        behavioral = pd.read_parquet(ROOT / "data/features/behavioral/cicids2017_behavioral_features_test.parquet")
        cls.rows = pd.concat([
            behavioral[behavioral["Label"] == "PortScan"].head(4),
            behavioral[behavioral["Label"] == "FTP-Patator"].head(4),
            behavioral[behavioral["Label"] == "BENIGN"].head(4),
        ])
        cls.explainer = FusionExplainer()
        cls.records = cls.explainer.explain(cls.rf_frame, cls.rows, top_k=5)

    def test_record_shape(self):
        for r in self.records:
            self.assertEqual(set(r), {"risk_score", "model_used", "fused_score", "top_features"})
            self.assertIsInstance(r["risk_score"], int)
            self.assertTrue(0 <= r["risk_score"] <= 100)
            self.assertEqual(len(r["top_features"]), 5)
            for f in r["top_features"]:
                self.assertEqual(set(f), {"feature", "value", "shap_value", "direction"})
                self.assertEqual(f["direction"], "increases_risk" if f["shap_value"] > 0 else "decreases_risk")
            mags = [abs(f["shap_value"]) for f in r["top_features"]]
            self.assertEqual(mags, sorted(mags, reverse=True))

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

    def test_attributions_come_from_deciding_member_feature_space(self):
        ae_only = set(self.explainer.ae_columns) - set(self.explainer.rf_columns)
        self.assertTrue(ae_only)  # behavioral columns exist only in the AE space
        seen = set()
        for r in self.records:
            seen.add(r["model_used"])
            names = {f["feature"] for f in r["top_features"]}
            space = self.explainer.rf_columns if r["model_used"] == "random_forest" else self.explainer.ae_columns
            self.assertLessEqual(names, set(space))
        self.assertEqual(seen, {"random_forest", "autoencoder_v2"})

    def test_model_used_matches_fusion_layer_and_fused_score(self):
        idx = self.rows["orig_row_index"].to_numpy()
        cols = self.explainer.rf_columns
        rf_prob = 1.0 - self.explainer.rf.predict_proba(self.rf_frame.iloc[idx][cols].to_numpy())[:, self.explainer.benign_idx]
        ae_err = self.explainer._ae_mse(self.rows[self.explainer.ae_columns].to_numpy(dtype=np.float32))
        fused, _, _, used = fusion_layer.fuse_with_metadata(rf_prob, np.zeros(len(rf_prob)), ae_err)
        self.assertEqual([r["model_used"] for r in self.records], used.tolist())
        np.testing.assert_allclose([r["fused_score"] for r in self.records], fused)

    def test_shap_is_additive_for_deciding_member(self):
        for member in ("random_forest", "autoencoder_v2"):
            sel = [i for i, r in enumerate(self.records) if r["model_used"] == member][:2]
            gap = self.explainer.additivity_gap(member, self.rf_frame, self.rows.iloc[sel])
            self.assertLess(gap, 1e-6)


if __name__ == "__main__":
    unittest.main()
