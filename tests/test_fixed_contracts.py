import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.evaluation import fusion_loao_trainnorm
from src.modeling import fusion_layer
from src.notifications import alert_dispatcher, notify_line


class FixedContractTests(unittest.TestCase):
    def test_fusion_uses_provisional_validation_thresholds_until_laoo_is_valid(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(fusion_layer.CONFIDENCE_THRESHOLD, 0.1)
        self.assertEqual(fusion_layer.FUSED_DECISION_THRESHOLD, 0.3)
        self.assertEqual(config["confidence_threshold"], fusion_layer.CONFIDENCE_THRESHOLD)
        self.assertEqual(config["fused_decision_threshold"], fusion_layer.FUSED_DECISION_THRESHOLD)
        self.assertIn("Validation-set", config["tuning_objective"])
        self.assertIn("deferred", config["loao_evaluation_status"])

    def test_loao_scope_does_not_claim_fused_zero_day_metrics(self):
        config_path = Path(__file__).resolve().parents[1] / "models" / "fusion_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["loao_metric_scope"], fusion_loao_trainnorm.LOAO_METRIC_SCOPE)
        self.assertIn("component metrics only", config["loao_evaluation_status"])
        self.assertIn("Fused zero-day metrics remain deferred", config["loao_evaluation_status"])

    def test_fusion_emits_actual_branch(self):
        _, _, confidence, model_used = fusion_layer.fuse_with_metadata(
            np.array([0.99, 0.50]), np.array([fusion_layer.IF_MIN, fusion_layer.IF_MAX]),
            np.array([fusion_layer.AE_MIN, fusion_layer.AE_MAX]),
        )
        np.testing.assert_allclose(confidence, [0.98, 0.0])
        self.assertEqual(model_used.tolist(), ["random_forest", "if_ae_fallback"])

    def test_dispatcher_copies_branch_instead_of_guessing_from_confidence(self):
        self.assertEqual(
            alert_dispatcher.derive_model_used({"model_used": "if_ae_fallback", "confidence": 1.0}, {}),
            "if_ae_fallback",
        )

    def test_dashboard_risk_score_clamps_display_without_changing_fused_score(self):
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
