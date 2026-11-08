import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.features import feature_extractor, feature_extractor_behavioral


class FeatureExtractorContractTests(unittest.TestCase):
    def test_standard_smote_model_inputs_exclude_label_derived_cti_columns(self):
        model_columns = feature_extractor.model_input_feature_cols(
            ["network_a", "network_b"], ["sentinel_missing"]
        )

        self.assertEqual(model_columns, ["network_a", "network_b", "sentinel_missing"])
        self.assertFalse(
            {
                "technique_id_encoded",
                "tactic_encoded",
                "mapping_confidence_encoded",
                "is_known_technique",
            }
            & set(model_columns)
        )

    def test_behavioral_smote_model_inputs_exclude_label_derived_cti_columns(self):
        model_columns = feature_extractor_behavioral.model_input_feature_cols(
            ["network_a", "behavioral_1m"], ["sentinel_missing"]
        )

        self.assertEqual(model_columns, ["network_a", "behavioral_1m", "sentinel_missing"])
        self.assertFalse(
            {
                "technique_id_encoded",
                "tactic_encoded",
                "mapping_confidence_encoded",
                "is_known_technique",
            }
            & set(model_columns)
        )

    def test_output_replacement_preserves_previous_artifact_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_dir = root / "features"
            outputs = {split: features_dir / f"dataset_{split}.parquet" for split in ("train", "val", "test")}
            outputs["train"].parent.mkdir(parents=True)
            outputs["train"].write_bytes(b"previous-output")
            labeled_path = root / "labeled.parquet"
            split_path = root / "train-input.parquet"
            labeled_path.write_bytes(b"labeled-input")
            split_path.write_bytes(b"split-input")
            final_dfs = {
                split: pd.DataFrame({"feature": [index]})
                for index, split in enumerate(("train", "val", "test"))
            }
            cfg = {
                "labeled_path": str(labeled_path),
                "split_paths": {"train": str(split_path)},
                "output_paths": {split: str(path) for split, path in outputs.items()},
            }

            with patch.object(feature_extractor, "FEATURES_DATA_DIR", str(features_dir)):
                written, manifest_path, history_dir = feature_extractor.write_feature_outputs(
                    "dataset", final_dfs, cfg, {"model_input_feature_columns": ["feature"]}
                )

            self.assertEqual((history_dir / outputs["train"].name).read_bytes(), b"previous-output")
            self.assertEqual(pd.read_parquet(written["train"]).to_dict("list"), {"feature": [0]})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["settings"]["model_input_feature_columns"], ["feature"])
            self.assertIn("sha256", manifest["inputs"][str(labeled_path)])
            self.assertIn("sha256", manifest["outputs"]["train"])


if __name__ == "__main__":
    unittest.main()
