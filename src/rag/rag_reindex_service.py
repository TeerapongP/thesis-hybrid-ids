"""
rag_reindex_service.py
=======================
RAG Layer -- scheduled re-fetch service for the MITRE ATT&CK knowledge base.

One-shot script (see models/rag_reindex_config.json for why one-shot rather than a
long-running --loop process): each invocation fetches the latest STIX bundle via
src/mitre_kb_extract.py (the validated GitHub-STIX source -- NOT OpenCTI, see CLAUDE.md's
"RAG Layer -- knowledge base source decision," a decision this service does not re-open),
diffs it against the on-disk data/knowledge_base/mitre_techniques.json per technique_id
(description/detection/mitigations -- the fields that actually feed the 45-chunk index),
and only rebuilds the ChromaDB collection (via the existing src/rag_retrieval.build_index)
if content for at least one of the 11 techniques actually changed. Every invocation appends
one row to reports/rag_reindex_log.csv, whether or not anything changed.

Meant to be invoked on a schedule externally (Windows Task Scheduler / cron), at the
interval given by .env's RAG_REINDEX_HOURS (168 = 7 days) -- this script reads that value
only to log/document it, it does not sleep or loop itself. See the bottom of this file's
__main__ block for the exact command to register with an external scheduler.

This module imports nothing from opencti/ and makes no network calls other than the GitHub
STIX bundle fetch -- it never touches the OpenCTI platform or any of its connectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion import mitre_kb_extract as mke
from src.rag import rag_retrieval

ROOT = Path(__file__).resolve().parent.parent.parent
KB_PATH = ROOT / "data" / "knowledge_base" / "mitre_techniques.json"
KB_HISTORY_DIR = ROOT / "data" / "knowledge_base" / "history"
LOG_PATH = ROOT / "reports" / "rag_reindex_log.csv"

LOG_FIELDS = [
    "run_timestamp_utc",
    "previous_attack_version",
    "fetched_attack_version",
    "mitre_version_changed",
    "content_changed",
    "techniques_changed",
    "index_rebuilt",
    "chroma_collection_count",
    "dry_run",
    "status",
    "error_message",
]

# Fields that actually feed the 45-chunk index (docs/rag_chunking_plan.md) -- these are the
# only fields compared to decide whether a rebuild is warranted. Metadata-only fields
# (tactics, platforms, sub_techniques, deprecated/revoked/url/attack_version) are NOT
# compared here: none of them are embedded into any chunk, so a change to only those would
# not change retrieval behavior and should not trigger a re-embed.
CONTENT_FIELDS = ["description", "detection", "mitigations"]


def load_current_kb(path: Path | None = None) -> dict[str, Any] | None:
    """`path` defaults to the CURRENT value of the module-level KB_PATH, read inside the
    function body rather than as a default-argument value -- a default argument is bound
    once at function-definition time, so a caller reassigning rag_reindex_service.KB_PATH
    (e.g. to redirect this into a test scratch directory) would otherwise be silently
    ignored by this function while other code in this module that references KB_PATH
    directly would honor it, an inconsistency that was actually hit and fixed during this
    service's own verification run."""
    if path is None:
        path = KB_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_techniques(old_techniques: dict[str, Any], new_techniques: dict[str, Any]) -> list[str]:
    """Return the technique_ids whose content-bearing fields differ between old and new.
    A technique present in `new` but missing from `old` (or vice versa) also counts as
    changed. Comparison is on CONTENT_FIELDS only -- see the module-level comment above."""
    changed = []
    all_ids = set(old_techniques) | set(new_techniques)
    for tid in sorted(all_ids):
        old_entry = old_techniques.get(tid)
        new_entry = new_techniques.get(tid)
        if old_entry is None or new_entry is None:
            changed.append(tid)
            continue
        if any(old_entry.get(f) != new_entry.get(f) for f in CONTENT_FIELDS):
            changed.append(tid)
    return changed


def backup_current_kb(path: Path | None = None, history_dir: Path | None = None) -> Path:
    """See load_current_kb's docstring for why `path`/`history_dir` are read from the
    current KB_PATH/KB_HISTORY_DIR module globals inside the function body, not bound as
    default-argument values."""
    if path is None:
        path = KB_PATH
    if history_dir is None:
        history_dir = KB_HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = history_dir / f"mitre_techniques_{stamp}.json"
    shutil.copy2(path, backup_path)
    return backup_path


def _current_chroma_count() -> int | None:
    """Read the existing collection's chunk count without triggering a build/embed -- for
    logging visibility only. Returns None if no collection exists yet (first-ever run)."""
    try:
        config = rag_retrieval.load_env_config()
        client = rag_retrieval.get_chroma_client()
        collection = client.get_collection(name=config["chroma_collection"])
        return collection.count()
    except Exception:
        return None


def append_log_row(row: dict[str, Any], log_path: Path | None = None) -> None:
    if log_path is None:
        log_path = LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def check_and_reindex(dry_run: bool = False) -> dict[str, Any]:
    """Single check-and-maybe-rebuild cycle. Always appends one row to
    reports/rag_reindex_log.csv, whether or not anything changed, whether or not an error
    occurred. Returns the logged row as a dict."""
    run_timestamp = datetime.now(timezone.utc).isoformat()
    row: dict[str, Any] = {f: "" for f in LOG_FIELDS}
    row["run_timestamp_utc"] = run_timestamp
    row["dry_run"] = dry_run

    try:
        current_kb = load_current_kb()
        previous_version = current_kb["source"]["attack_version"] if current_kb else None
        row["previous_attack_version"] = previous_version or ""

        print(f"[rag_reindex_service] fetching latest STIX bundle from {mke.STIX_BUNDLE_URL} ...")
        bundle = mke.fetch_stix_bundle()
        new_kb = mke.build_knowledge_base(bundle)
        fetched_version = new_kb["source"]["attack_version"]
        row["fetched_attack_version"] = fetched_version or ""

        version_changed = bool(current_kb) and (fetched_version != previous_version)
        row["mitre_version_changed"] = version_changed

        if current_kb is None:
            # No local KB yet -- treat as fully changed (first-ever run).
            techniques_changed = sorted(new_kb["techniques"].keys())
        else:
            techniques_changed = diff_techniques(current_kb["techniques"], new_kb["techniques"])
        content_changed = bool(techniques_changed)
        row["content_changed"] = content_changed
        row["techniques_changed"] = ",".join(techniques_changed)

        print(
            f"[rag_reindex_service] previous_version={previous_version} fetched_version={fetched_version} "
            f"mitre_version_changed={version_changed} content_changed={content_changed} "
            f"techniques_changed={techniques_changed or '[]'}"
        )

        index_rebuilt = False
        if content_changed and not dry_run:
            backup_path = None
            if current_kb is not None:
                backup_path = backup_current_kb()
                print(f"[rag_reindex_service] backed up previous KB to {backup_path}")
            KB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(KB_PATH, "w", encoding="utf-8") as f:
                json.dump(new_kb, f, indent=2, ensure_ascii=False)
            print(f"[rag_reindex_service] wrote updated KB to {KB_PATH}")

            print("[rag_reindex_service] content changed -- rebuilding ChromaDB index (force=True) ...")
            try:
                collection = rag_retrieval.build_index(force=True)
            except Exception:
                # KB_PATH was already overwritten above -- if the rebuild itself fails (e.g.
                # verify_chunk_counts' assertion, an embedding-model error), roll it back so
                # the file and the (unchanged) ChromaDB index stay consistent, AND so the
                # next run's diff again sees old-vs-fetched content differ and retries the
                # rebuild, rather than believing (wrongly) that nothing changed forever.
                if backup_path is not None:
                    shutil.copy2(backup_path, KB_PATH)
                    print(f"[rag_reindex_service] rebuild failed -- rolled back {KB_PATH} from backup")
                else:
                    KB_PATH.unlink(missing_ok=True)
                    print(
                        f"[rag_reindex_service] rebuild failed -- removed freshly written {KB_PATH} "
                        "(no prior version existed to roll back to)"
                    )
                raise
            index_rebuilt = True
            row["chroma_collection_count"] = collection.count()
        else:
            reason = "dry_run=True" if (content_changed and dry_run) else "no content change"
            print(f"[rag_reindex_service] skipping index rebuild ({reason})")
            row["chroma_collection_count"] = _current_chroma_count()

        row["index_rebuilt"] = index_rebuilt
        row["status"] = "ok"
        row["error_message"] = ""

    except Exception as exc:  # noqa: BLE001 -- must still log a row before propagating
        row["status"] = "error"
        row["error_message"] = str(exc)
        append_log_row(row)
        print(f"[rag_reindex_service] ERROR: {exc}")
        raise
    else:
        append_log_row(row)

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and diff only -- never overwrite mitre_techniques.json or rebuild the ChromaDB index. Still logs the attempt.",
    )
    args = parser.parse_args()

    config = rag_retrieval.load_env_config()
    print(
        f"[rag_reindex_service] RAG_REINDEX_HOURS={config['rag_reindex_hours']} "
        f"(documentation only -- this run is a single one-shot check, not a loop)"
    )

    result = check_and_reindex(dry_run=args.dry_run)
    print(f"[rag_reindex_service] done. Logged to {LOG_PATH}: {result}")

    print(
        "\n[rag_reindex_service] To run this on a schedule, register it with an external "
        "scheduler at the RAG_REINDEX_HOURS interval, e.g.:\n"
        "  Windows Task Scheduler (every 168 hours / 7 days):\n"
        '    schtasks /create /tn "RAG KB Reindex" /tr "python E:\\thesis\\src\\rag\\rag_reindex_service.py" '
        '/sc weekly /d MON /st 03:00\n'
        "  cron (weekly, Monday 03:00):\n"
        "    0 3 * * 1 cd /path/to/thesis && python -m src.rag.rag_reindex_service\n"
        "This script itself does not loop or sleep -- each invocation is one check."
    )
