import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.evaluation import fusion_loao_trainnorm
from src.modeling import fusion_layer
from src.notifications import alert_dispatcher, notify_line


class FixedContractTests(unittest.TestCase):
    def test_deployed_thresholds_are_loao_selected_and_match_the_config(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(fusion_layer.CHOSEN_STRATEGY, "confidence_weighted")
        self.assertEqual(fusion_layer.FUSED_DECISION_THRESHOLD, 0.03)
        self.assertEqual(fusion_layer.CONFIDENCE_THRESHOLD, 0.95)
        self.assertEqual(config["chosen_strategy"], fusion_layer.CHOSEN_STRATEGY)
        self.assertEqual(config["fused_decision_threshold"], fusion_layer.FUSED_DECISION_THRESHOLD)
        self.assertEqual(config["confidence_threshold"], fusion_layer.CONFIDENCE_THRESHOLD)
        # The parameters must be LOAO-selected. Selecting them on val would tune on the data
        # the thesis then reports, which is the error that made the old cascade look good.
        self.assertIn("Nested LOAO", config["parameter_selection"])
        self.assertIn("not a value tuned on the data it is reported against",
                      config["parameter_selection"])

    def test_loao_scope_matches_what_the_component_run_actually_computed(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIn("train-fold normalized", config["loao_evaluation"]["method"])
        self.assertIn("fusion_loao_unified.py", config["loao_evaluation"]["method"])
        # The component run must not compute fused metrics: thresholds are selected
        # separately and nested, so a fused number from the same folds would be tuned-on-all.
        self.assertEqual(
            fusion_loao_trainnorm.LOAO_METRIC_SCOPE, "component_models_only_no_fusion"
        )
        for report in config["loao_evaluation"]["reports"]:
            self.assertTrue((Path(__file__).resolve().parents[1] / report).exists(), report)

    def test_normalization_constants_are_train_fitted_not_val_fitted(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        norm = config["normalization"]
        self.assertEqual(norm["ae_max"], fusion_layer.AE_MAX)
        self.assertEqual(norm["if_max"], fusion_layer.IF_MAX)
        self.assertEqual(norm["ae_min"], fusion_layer.AE_MIN)
        self.assertEqual(norm["if_min"], fusion_layer.IF_MIN)
        self.assertIn("unified", norm["fit_on"])
        self.assertIn("train", norm["fit_on"])
        # A val-fit ae_max was 0.811 against a true train-fit 0.1225 -- 6.6x too large, which
        # would shrink every deployed ae_norm by that factor and silence the detector.
        self.assertLess(fusion_layer.AE_MAX, 0.2)

    def test_fusion_emits_the_branch_that_decided_each_row(self):
        """Row 0: RF is decisive (|0.99-0.5|*2 = 0.98 >= 0.95) so RF owns it and keeps its own
        score. Row 1: RF sits at the midpoint, confidence 0, so the unsupervised mean decides."""
        rf = np.array([0.99, 0.5])
        if_raw = np.array([fusion_layer.IF_MIN, fusion_layer.IF_MAX])
        ae_raw = np.array([fusion_layer.AE_MIN, fusion_layer.AE_MAX])
        score, prediction, margin, model_used = fusion_layer.fuse_with_metadata(rf, if_raw, ae_raw)
        np.testing.assert_allclose(score, [0.99, 1.0])
        self.assertEqual(model_used.tolist(), ["random_forest", "if_ae_fallback"])
        self.assertEqual(sorted(set(model_used.tolist())),
                         sorted(fusion_layer.MODEL_USED_VALUES))
        np.testing.assert_allclose(margin, score - fusion_layer.FUSED_DECISION_THRESHOLD)
        self.assertEqual(prediction.tolist(), [1, 1])

    def test_isolation_forest_decides_undecided_rows_but_not_decisive_ones(self):
        """INVERTED from the previous suite on purpose.

        Under the retired `rf_ae_or` rule Isolation Forest provably could not change the
        decision -- it was scored for metadata only. The deployed cascade routes rows RF is
        unsure about to `(if_norm + ae_norm) / 2`, so IF DOES decide those rows. Both halves
        matter: if IF influenced a row RF was decisive about, the RF-primary design would be
        broken; if it influenced nothing, the cascade would be RF solo wearing a costume.
        """
        ae = np.array([fusion_layer.AE_MIN])
        # RF at the midpoint -> confidence 0 -> fallback -> IF must move the score.
        undecided_low = fusion_layer.fuse_with_metadata(
            np.array([0.5]), np.array([fusion_layer.IF_MIN]), ae)
        undecided_high = fusion_layer.fuse_with_metadata(
            np.array([0.5]), np.array([fusion_layer.IF_MAX]), ae)
        self.assertGreater(float(undecided_high[0][0]), float(undecided_low[0][0]))
        self.assertEqual(undecided_high[3].tolist(), ["if_ae_fallback"])

        # RF confidently BENIGN -> confidence 1.0 -> RF owns the row -> IF must be ignored.
        decisive_low = fusion_layer.fuse_with_metadata(
            np.array([0.0]), np.array([fusion_layer.IF_MIN]), ae)
        decisive_high = fusion_layer.fuse_with_metadata(
            np.array([0.0]), np.array([fusion_layer.IF_MAX * 10]), ae)
        np.testing.assert_allclose(decisive_low[0], decisive_high[0])
        self.assertEqual(decisive_high[3].tolist(), ["random_forest"])

    def test_dispatcher_copies_branch_instead_of_guessing_from_confidence(self):
        self.assertEqual(
            alert_dispatcher.derive_model_used({"model_used": "if_ae_fallback", "confidence": 1.0}, {}),
            "if_ae_fallback",
        )

    def test_dashboard_risk_score_clamps_display_without_changing_fused_score(self):
        # Both rows must be UNDECIDED (rf at the midpoint, confidence 0) to reach the
        # fallback. An rf of 0.0 would be maximally confident, so RF would own the row and
        # return 0.0 exactly -- which is not an overflow and would not exercise the clamp.
        below_score = float(
            fusion_layer.fuse_with_metadata(
                0.5,
                fusion_layer.IF_MIN - 0.1,
                fusion_layer.AE_MIN - 0.1,
            )[0]
        )
        above_score = float(
            fusion_layer.fuse_with_metadata(
                0.5,
                fusion_layer.IF_MAX + 0.1,
                fusion_layer.AE_MAX + 0.1,
            )[0]
        )
        # Under the cascade BOTH overflows are reachable, unlike the retired OR where
        # max(rf, ae) could never fall below RF's probability floor: an undecided row whose
        # unsupervised scores sit below the train minimum averages to a negative fused score.
        self.assertLess(below_score, 0.0)
        self.assertGreater(above_score, 1.0)

        records = [
            alert_dispatcher.assemble_alert_record(
                {}, score, {"priority": "HIGH"}, "if_ae_fallback", f"alert-{index}"
            )
            for index, score in enumerate((below_score, above_score))
        ]

        self.assertEqual([record["risk_score"] for record in records], [0, 100])
        self.assertEqual([record["fused_score"] for record in records], [below_score, above_score])
        self.assertEqual([record["priority"] for record in records], ["HIGH", "HIGH"])

    def test_line_uses_messaging_api_push_payload(self):
        response = Mock(status_code=200)
        config = {
            "line_channel_access_token": "AbCdEfGhIjKlMnOpQrStUvWxYz123456",
            "line_to_id": "U" + "1" * 32,
            "line_grafana_base_url": "http://localhost:3001",
        }
        with patch.object(notify_line.requests, "post", return_value=response) as post:
            result = notify_line.send_line_alert({"priority": "HIGH"}, config)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(post.call_args.args[0], notify_line.LINE_PUSH_URL)
        self.assertEqual(post.call_args.kwargs["json"]["to"], config["line_to_id"])

    def test_mttd_uses_ingest_and_detection_timestamps(self):
        self.assertEqual(
            alert_dispatcher._elapsed_ms("2026-09-24T10:00:00+07:00", "2026-09-24T10:00:02+07:00"),
            2000.0,
        )


if __name__ == "__main__":
    unittest.main()
