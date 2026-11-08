from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


RUN_STARTED_AT_METADATA_KEY = "run_started_at_utc"
RUN_STARTED_AT_REPORT_LABEL = "Run started at UTC"


def build_ensemble_artifact_paths(reports_dir, confidence_threshold, decision_threshold, run_started_at=None):
    started_at = run_started_at or datetime.now(timezone.utc)
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}"
    threshold_tag = (
        f"ct{confidence_threshold:.2f}_dt{decision_threshold:.2f}".replace(".", "p")
    )
    artifact_stem = f"ensemble_test_current_{threshold_tag}_{run_id}"
    root = Path(reports_dir)
    paths = {
        "metrics": root / f"{artifact_stem}_metrics.csv",
        "per_category": root / f"{artifact_stem}_per_category.csv",
        "report": root / f"{artifact_stem}_report.txt",
    }
    return started_at, run_id, paths


def ensure_artifact_paths_available(paths):
    existing = [str(path) for path in paths.values() if Path(path).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing evaluation artifacts: {existing}")


@contextmanager
def open_artifact_outputs_exclusively(paths):
    streams = {}
    created_paths = []
    try:
        for name, path in paths.items():
            output_path = Path(path)
            stream = output_path.open("x", encoding="utf-8", newline="")
            streams[name] = stream
            created_paths.append(output_path)
        yield streams
    except BaseException:
        for stream in streams.values():
            stream.close()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
