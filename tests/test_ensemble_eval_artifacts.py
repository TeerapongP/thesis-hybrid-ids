import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from src.evaluation.ensemble_eval_artifacts import (
    RUN_STARTED_AT_METADATA_KEY,
    RUN_STARTED_AT_REPORT_LABEL,
    build_ensemble_artifact_paths,
    ensure_artifact_paths_available,
    open_artifact_outputs_exclusively,
)


class EnsembleEvalArtifactTests(unittest.TestCase):
    def test_timestamp_metadata_names_the_run_start_time(self):
        self.assertEqual(RUN_STARTED_AT_METADATA_KEY, "run_started_at_utc")
        self.assertEqual(RUN_STARTED_AT_REPORT_LABEL, "Run started at UTC")

    def test_run_ids_are_unique_for_the_same_timestamp(self):
        timestamp = datetime(2026, 10, 16, tzinfo=timezone.utc)
        with patch(
            "src.evaluation.ensemble_eval_artifacts.uuid4",
            side_effect=(UUID(int=1), UUID(int=2)),
        ):
            first = build_ensemble_artifact_paths("reports", 0.1, 0.3, timestamp)
            second = build_ensemble_artifact_paths("reports", 0.1, 0.3, timestamp)

        self.assertNotEqual(first[1], second[1])
        self.assertTrue(set(first[2].values()).isdisjoint(second[2].values()))

    def test_existing_artifact_path_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.csv"
            new_path = Path(directory) / "new.csv"
            existing.write_text("original", encoding="utf-8")
            paths = {"report": new_path, "metrics": existing}

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                ensure_artifact_paths_available(paths)

            with self.assertRaises(FileExistsError):
                with open_artifact_outputs_exclusively(paths):
                    self.fail("existing outputs must block file creation")

            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
            self.assertFalse(new_path.exists())


if __name__ == "__main__":
    unittest.main()
