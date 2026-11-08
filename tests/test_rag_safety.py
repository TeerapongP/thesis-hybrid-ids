import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import chromadb

from src.rag import rag_reindex_service, rag_retrieval


class RagSafetyTests(unittest.TestCase):
    def test_name_only_change_triggers_technique_reindex(self):
        old = {
            "T1000": {
                "name": "Old technique name",
                "description": "same description",
                "detection": "same detection",
                "mitigations": [],
            }
        }
        new = json.loads(json.dumps(old))
        new["T1000"]["name"] = "New technique name"

        self.assertEqual(rag_reindex_service.diff_techniques(old, new), ["T1000"])

    def test_index_replaces_stale_ids_even_when_collection_count_matches(self):
        chunks = rag_retrieval.build_chunks(rag_retrieval.load_mitre_techniques())
        expected_ids = {chunk["id"] for chunk in chunks}
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            "rag-safety-test", metadata={"hnsw:space": "cosine"}
        )
        seed_chunks = chunks[:-1]
        collection.add(
            ids=[chunk["id"] for chunk in seed_chunks] + ["stale::id"],
            documents=[chunk["text"] for chunk in seed_chunks] + ["stale"],
            metadatas=[chunk["metadata"] for chunk in seed_chunks]
            + [{"chunk_type": "description"}],
            embeddings=[[0.0, 0.0, 1.0] for _ in range(len(seed_chunks) + 1)],
        )
        self.assertEqual(collection.count(), rag_retrieval.EXPECTED_CHUNK_COUNT)

        with (
            patch.object(rag_retrieval, "get_embedder", return_value=object()),
            patch.object(
                rag_retrieval,
                "embed_texts",
                return_value=[[0.0, 0.0, 1.0] for _ in chunks],
            ) as embed_texts,
        ):
            result = rag_retrieval.build_index(
                client=client,
                config={"chroma_collection": "rag-safety-test", "embed_model": "test"},
            )

        ids = result.get(include=[])["ids"]
        self.assertEqual(set(ids), expected_ids)
        self.assertNotIn("stale::id", ids)
        embed_texts.assert_called_once()

    def test_atomic_kb_write_preserves_original_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mitre_techniques.json"
            path.write_text('{"version":"old"}', encoding="utf-8")

            with patch.object(rag_reindex_service.os, "replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    rag_reindex_service.write_current_kb_atomic({"version": "new"}, path)

            self.assertEqual(path.read_text(encoding="utf-8"), '{"version":"old"}')
            self.assertEqual(list(Path(directory).glob(".mitre_techniques.json.*.tmp")), [])

    def test_reindex_failure_restores_previous_index_after_kb_rollback(self):
        current = rag_retrieval.load_mitre_techniques()
        updated = json.loads(json.dumps(current))
        technique_id = next(iter(updated["techniques"]))
        updated["techniques"][technique_id]["description"] += " updated"

        class Collection:
            def count(self):
                return rag_retrieval.EXPECTED_CHUNK_COUNT

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kb_path = root / "mitre_techniques.json"
            kb_path.write_text(json.dumps(current), encoding="utf-8")
            with (
                patch.object(rag_reindex_service, "KB_PATH", kb_path),
                patch.object(rag_reindex_service, "KB_HISTORY_DIR", root / "history"),
                patch.object(rag_reindex_service, "LOG_PATH", root / "log.csv"),
                patch.object(rag_reindex_service.mke, "fetch_stix_bundle", return_value=object()),
                patch.object(rag_reindex_service.mke, "build_knowledge_base", return_value=updated),
                patch.object(
                    rag_reindex_service.rag_retrieval,
                    "build_index",
                    side_effect=[RuntimeError("new index failed"), Collection()],
                ) as build_index,
            ):
                with self.assertRaisesRegex(RuntimeError, "new index failed"):
                    rag_reindex_service.check_and_reindex()

            self.assertEqual(json.loads(kb_path.read_text(encoding="utf-8")), current)
            self.assertEqual(build_index.call_args_list, [call(force=True), call(force=True)])


if __name__ == "__main__":
    unittest.main()
