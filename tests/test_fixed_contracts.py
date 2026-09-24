import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.modeling import fusion_layer
from src.notifications import alert_dispatcher, notify_line


class FixedContractTests(unittest.TestCase):
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
