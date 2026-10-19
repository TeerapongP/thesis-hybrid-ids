import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import fusion_loao_trainnorm


class FusionLoaoCheckpointTests(unittest.TestCase):
    def test_signature_changes_when_input_or_configuration_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.parquet"
            config_path = root / "ae_config.json"
            data_path.write_bytes(b"input-one")
            config_path.write_text(json.dumps({"epochs": 2}), encoding="utf-8")
            settings = {"random_state": 42}

            initial = fusion_loao_trainnorm.compute_run_signature(
                [data_path], config_path, settings
            )
            data_path.write_bytes(b"input-two")
            changed_input = fusion_loao_trainnorm.compute_run_signature(
                [data_path], config_path, settings
            )
            data_path.write_bytes(b"input-one")
            config_path.write_text(json.dumps({"epochs": 3}), encoding="utf-8")
            changed_config = fusion_loao_trainnorm.compute_run_signature(
                [data_path], config_path, settings
            )
            changed_settings = fusion_loao_trainnorm.compute_run_signature(
                [data_path], config_path, {"random_state": 43}
            )

            self.assertNotEqual(initial, changed_input)
            self.assertNotEqual(initial, changed_config)
            self.assertNotEqual(initial, changed_settings)

    def test_checkpoint_requires_matching_signatures_in_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = "a" * 64
            result = {"category": "Bot", "rf_recall": 0.5}
            scores = pd.DataFrame({"category": ["Bot"], "rf_norm": [0.5]})

            fusion_loao_trainnorm.save_checkpoint(result, scores, signature, root)

            loaded = fusion_loao_trainnorm.load_checkpoint("Bot", signature, root)
            self.assertEqual(loaded["run_signature"], signature)
            self.assertIsNone(
                fusion_loao_trainnorm.load_checkpoint("Bot", "b" * 64, root)
            )

            scores_path = next((root / signature).glob("bot_*_scores.parquet"))
            pd.DataFrame({"run_signature": ["c" * 64]}).to_parquet(scores_path, index=False)
            self.assertIsNone(fusion_loao_trainnorm.load_checkpoint("Bot", signature, root))


if __name__ == "__main__":
    unittest.main()
