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
        self.assertEqual(fusion_layer.CHOSEN_STRATEGY, "rf_ae_or")
        self.assertEqual(fusion_layer.FUSED_DECISION_THRESHOLD, 0.05)
        self.assertEqual(config["chosen_strategy"], fusion_layer.CHOSEN_STRATEGY)
        self.assertEqual(config["fused_decision_threshold"], fusion_layer.FUSED_DECISION_THRESHOLD)
        self.assertIn("Nested LOAO", config["tuning_objective"])
        self.assertIn("NOT validation-selected", config["tuning_objective"])

    def test_loao_scope_matches_what_the_component_run_actually_computed(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["loao_evaluation"]["status"], "complete")
        self.assertEqual(config["loao_evaluation"]["normalization_scope"], "train_fold_fitted")
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
        self.assertIn("behavioral_features_train", norm["fit_population"])
        self.assertLess(fusion_layer.AE_MAX, 0.2)

    def test_fusion_emits_the_member_that_supplied_the_max(self):
        rf = np.array([0.99, 0.0])
        ae_raw = np.array([fusion_layer.AE_MIN, fusion_layer.AE_MAX])
        score, prediction, margin, model_used = fusion_layer.fuse_with_metadata(
            rf, np.array([fusion_layer.IF_MIN, fusion_layer.IF_MAX]), ae_raw,
        )
        np.testing.assert_allclose(score, [0.99, 1.0])
        self.assertEqual(model_used.tolist(), ["random_forest", "autoencoder_v2"])
        np.testing.assert_allclose(margin, score - fusion_layer.FUSED_DECISION_THRESHOLD)
        self.assertEqual(prediction.tolist(), [1, 1])

    def test_isolation_forest_does_not_change_the_deployed_decision(self):
        args = (np.array([0.02]), np.array([fusion_layer.IF_MAX * 10]), np.array([fusion_layer.AE_MIN]))
        low_if = fusion_layer.fuse_with_metadata(args[0], np.array([fusion_layer.IF_MIN]), args[2])
        high_if = fusion_layer.fuse_with_metadata(*args)
        np.testing.assert_allclose(low_if[0], high_if[0])
        self.assertEqual(low_if[1].tolist(), high_if[1].tolist())

    def test_dispatcher_copies_branch_instead_of_guessing_from_confidence(self):
        self.assertEqual(
            alert_dispatcher.derive_model_used({"model_used": "autoencoder_v2", "confidence": 1.0}, {}),
            "autoencoder_v2",
        )

    def test_dashboard_risk_score_clamps_display_without_changing_fused_score(self):
        below_score = float(
            fusion_layer.fuse_with_metadata(
                0.0,
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
        # max(rf, ae) cannot fall below RF's probability floor, so only the upper
        # overflow is reachable under rf_ae_or. The lower clamp still has to hold
        # for any caller that hands the dispatcher a negative score directly.
        self.assertGreaterEqual(below_score, 0.0)
        self.assertGreater(above_score, 1.0)
        below_score = -0.25

        records = [
            alert_dispatcher.assemble_alert_record(
                {}, score, {"priority": "HIGH"}, "autoencoder_v2", f"alert-{index}"
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
