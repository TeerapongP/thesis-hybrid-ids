import unittest

import pandas as pd

from src.evaluation import build_cross_dataset_eval


class CrossDatasetEvalTests(unittest.TestCase):
    def test_detector_inputs_exclude_label_and_cti_derived_columns(self):
        columns = [
            "Flow Bytes/s",
            "Init_Win_bytes_forward_missing",
            "Label",
            "technique_id",
            "technique_name",
            "tactic",
            "mapping_confidence",
            "technique_id_encoded",
            "tactic_encoded",
            "mapping_confidence_encoded",
            "is_known_technique",
            "attack_cat",
            "row_index",
            "orig_row_index",
        ]

        self.assertEqual(
            build_cross_dataset_eval.model_input_feature_cols(columns),
            ["Flow Bytes/s", "Init_Win_bytes_forward_missing"],
        )

    def test_each_direction_scales_target_with_its_source_training_range(self):
        cic_source_train = pd.DataFrame({"flow": [0.0, 10.0]})
        unsw_target_test = pd.DataFrame({"flow": [20.0]})
        unsw_source_train = pd.DataFrame({"flow": [100.0, 200.0]})
        cic_target_test = pd.DataFrame({"flow": [0.0]})

        unsw_scaled, cic_scaler = build_cross_dataset_eval.scale_target_with_source_training(
            cic_source_train, unsw_target_test, ["flow"]
        )
        cic_scaled, unsw_scaler = build_cross_dataset_eval.scale_target_with_source_training(
            unsw_source_train, cic_target_test, ["flow"]
        )

        self.assertEqual(float(cic_scaler.data_min_[0]), 0.0)
        self.assertEqual(float(cic_scaler.data_max_[0]), 10.0)
        self.assertEqual(float(unsw_scaler.data_min_[0]), 100.0)
        self.assertEqual(float(unsw_scaler.data_max_[0]), 200.0)
        self.assertEqual(float(unsw_scaled.iloc[0]["flow"]), 2.0)
        self.assertEqual(float(cic_scaled.iloc[0]["flow"]), -1.0)


if __name__ == "__main__":
    unittest.main()
