import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookEvalIntegrityTests(unittest.TestCase):
    def test_laoo_cells_are_disabled_without_stale_outputs(self):
        notebook = json.loads((ROOT / "notebooks" / "08_fusion_layer.ipynb").read_text(encoding="utf-8"))
        cells = notebook["cells"]
        section_idx = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and "".join(cell["source"]).lstrip().startswith("## 7.")
        )
        section = "".join(cells[section_idx]["source"])
        self.assertIn("stale outputs were cleared", section)
        disabled_cells = [cell for cell in cells[section_idx + 1:] if cell["cell_type"] == "code"]
        self.assertTrue(disabled_cells)
        for cell in disabled_cells:
            self.assertIn("raise RuntimeError", "".join(cell["source"]))
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
        expected_reports = (
            "fusion_layer_loao_results_legacy_test_fold_normalization.csv",
            "fusion_layer_solo_swept_baseline_legacy_test_fold_normalization.csv",
            "fusion_layer_nested_threshold_selection_legacy_test_fold_normalization.csv",
        )
        for report_name in expected_reports:
            report = ROOT / "reports" / report_name
            self.assertTrue(report.is_file(), report_name)
            self.assertGreater(report.stat().st_size, 0, report_name)
        checkpoints = list((ROOT / "reports" / "fusion_layer_loao_checkpoints").glob("*_scores.parquet"))
        self.assertTrue(checkpoints)

    def test_active_thresholds_have_validation_sweep_provenance(self):
        notebook = json.loads((ROOT / "notebooks" / "08_fusion_layer.ipynb").read_text(encoding="utf-8"))
        cells = notebook["cells"]
        self.assertIn("cicids2017_rf_scores_val.parquet", "".join(cells[5]["source"]))
        self.assertIn("cicids2017_behavioral_features_val.parquet", "".join(cells[8]["source"]))
        sweep_cell = next(
            cell
            for cell in cells
            if cell["cell_type"] == "code" and "CONFIDENCE_THRESHOLD = float(best_conf_row" in "".join(cell["source"])
        )
        sweep_source = "".join(sweep_cell["source"])
        self.assertIn('y_true = fusion_pop["is_attack"].to_numpy()', sweep_source)
        self.assertIn("sweep_best_threshold(y_true, score)", sweep_source)
        output = "".join(
            "".join(item.get("text", []))
            for item in sweep_cell.get("outputs", [])
            if item.get("output_type") == "stream"
        )
        self.assertIn("0.10                0.30", output)
        self.assertIn("0.996529", output)
        self.assertIn("confidence_weighted  chosen threshold=0.300", output)
        # The notebook's own val-swept thresholds (0.10 / 0.30) must survive in the config as
        # history, and must NOT be what is deployed: val selection is what this whole trail
        # moved away from. The deployed pair is nested-LOAO-selected on the unified population.
        config = json.loads((ROOT / "models" / "fusion_config.json").read_text(encoding="utf-8"))
        prior = config["superseded_history"]["prior_superseded_entries"]
        self.assertEqual(prior["val_selected_thresholds"][:24], "confidence_threshold=0.1")
        self.assertEqual(config["fused_decision_threshold"], 0.03)
        self.assertEqual(config["confidence_threshold"], 0.95)
        self.assertIn("Nested LOAO", config["parameter_selection"])


if __name__ == "__main__":
    unittest.main()
