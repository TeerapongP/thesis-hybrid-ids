
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from dotenv import dotenv_values
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT / ".env"
MITRE_JSON_PATH = ROOT / "data" / "knowledge_base" / "mitre_techniques.json"
CHROMA_DB_DIR = ROOT / "data" / "chroma_db"

EXPECTED_CHUNK_COUNT = 45
EXPECTED_DESCRIPTION_COUNT = 11
EXPECTED_DETECTION_COUNT = 11
EXPECTED_MITIGATION_COUNT = 23

DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5


def load_env_config(env_path: Path = ENV_PATH) -> dict[str, Any]:
    raw = dotenv_values(env_path)
    missing = [k for k in ("CHROMA_HOST", "CHROMA_PORT", "CHROMA_COLLECTION", "EMBED_MODEL", "RAG_TOP_K", "RAG_REINDEX_HOURS") if k not in raw or raw[k] is None]
    if missing:
        raise RuntimeError(f".env is missing required RAG keys: {missing}")
    return {
        "chroma_host": raw["CHROMA_HOST"],
        "chroma_port": int(raw["CHROMA_PORT"]),
        "chroma_collection": raw["CHROMA_COLLECTION"],
        "embed_model": raw["EMBED_MODEL"],
        "rag_top_k": int(raw["RAG_TOP_K"]),
        "rag_reindex_hours": int(raw["RAG_REINDEX_HOURS"]),
    }


def load_mitre_techniques(path: Path = MITRE_JSON_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks(mitre_data: dict[str, Any]) -> list[dict[str, Any]]:
    techniques: dict[str, Any] = mitre_data["techniques"]
    top_level_ids = list(techniques.keys())

    chunks: list[dict[str, Any]] = []

    for tid in top_level_ids:
        entry = techniques[tid]
        name = entry.get("name", "")

        chunks.append(
            {
                "id": f"{tid}::description",
                "text": entry["description"],
                "metadata": {
                    "technique_id": tid,
                    "technique_name": name,
                    "chunk_type": "description",
                },
            }
        )
        chunks.append(
            {
                "id": f"{tid}::detection",
                "text": entry["detection"],
                "metadata": {
                    "technique_id": tid,
                    "technique_name": name,
                    "chunk_type": "detection",
                },
            }
        )

    mitigation_first_seen: dict[str, dict[str, Any]] = {}
    mitigation_technique_ids: dict[str, list[str]] = {}

    for tid in top_level_ids:
        entry = techniques[tid]
        for mit in entry.get("mitigations", []):
            mid = mit["mitigation_id"]
            if mid not in mitigation_first_seen:
                mitigation_first_seen[mid] = {"name": mit["name"], "description": mit["description"]}
                mitigation_technique_ids[mid] = []
            if tid not in mitigation_technique_ids[mid]:
                mitigation_technique_ids[mid].append(tid)

    for mid, info in mitigation_first_seen.items():
        applicable = mitigation_technique_ids[mid]
        chunks.append(
            {
                "id": f"mitigation::{mid}",
                "text": info["description"],
                "metadata": {
                    "technique_id": applicable[0],
                    "technique_ids": ",".join(applicable),
                    "chunk_type": "mitigation",
                    "mitigation_id": mid,
                    "mitigation_name": info["name"],
                },
            }
        )

    return chunks


def verify_chunk_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"description": 0, "detection": 0, "mitigation": 0}
    for c in chunks:
        counts[c["metadata"]["chunk_type"]] += 1
    assert len(chunks) == EXPECTED_CHUNK_COUNT, f"expected {EXPECTED_CHUNK_COUNT} chunks total, got {len(chunks)}"
    assert counts["description"] == EXPECTED_DESCRIPTION_COUNT, counts
    assert counts["detection"] == EXPECTED_DETECTION_COUNT, counts
    assert counts["mitigation"] == EXPECTED_MITIGATION_COUNT, counts
    return counts


_embedder_cache: dict[str, Any] = {}


def get_embedder(model_name: str):
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer

        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_texts(embedder, texts: list[str]) -> list[list[float]]:
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def get_chroma_client(db_dir: Path = CHROMA_DB_DIR) -> chromadb.ClientAPI:
    db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_dir))


def build_index(
    client: chromadb.ClientAPI | None = None,
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> chromadb.Collection:
    config = config or load_env_config()
    client = client or get_chroma_client()

    collection = client.get_or_create_collection(
        name=config["chroma_collection"],
        metadata={"hnsw:space": "cosine"},
    )

    mitre_data = load_mitre_techniques()
    chunks = build_chunks(mitre_data)
    verify_chunk_counts(chunks)
    expected_ids = {chunk["id"] for chunk in chunks}
    if not force and set(collection.get(include=[])["ids"]) == expected_ids:
        return collection

    embedder = get_embedder(config["embed_model"])
    embeddings = embed_texts(embedder, [c["text"] for c in chunks])

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    stale_ids = set(collection.get(include=[])["ids"]) - expected_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))

    if collection.count() != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            f"mitre_attack_kb collection has {collection.count()} chunks after build, "
            f"expected {EXPECTED_CHUNK_COUNT}."
        )

    return collection


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _minmax_norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class RagRetriever:

    def __init__(self, config: dict[str, Any] | None = None, ensure_index: bool = True):
        self.config = config or load_env_config()
        self.client = get_chroma_client()
        if ensure_index:
            self.collection = build_index(client=self.client, config=self.config)
        else:
            self.collection = self.client.get_collection(name=self.config["chroma_collection"])

        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = stored["ids"]
        self.documents: list[str] = stored["documents"]
        self.metadatas: list[dict[str, Any]] = stored["metadatas"]
        self.id_to_index = {cid: i for i, cid in enumerate(self.ids)}

        self.embedder = get_embedder(self.config["embed_model"])
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.config["rag_top_k"]
        n = len(self.ids)

        query_embedding = embed_texts(self.embedder, [query])[0]
        dense_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["distances"],
        )
        dense_sim_by_id = {
            cid: 1.0 - dist
            for cid, dist in zip(dense_result["ids"][0], dense_result["distances"][0])
        }
        dense_scores_ordered = [dense_sim_by_id[cid] for cid in self.ids]
        dense_norm = _minmax_norm(dense_scores_ordered)

        bm25_scores_ordered = list(self._bm25.get_scores(tokenize(query)))
        bm25_norm = _minmax_norm(bm25_scores_ordered)

        combined = [dense_weight * d + bm25_weight * b for d, b in zip(dense_norm, bm25_norm)]

        ranked_indices = sorted(range(n), key=lambda i: combined[i], reverse=True)[:top_k]
        results = []
        for rank, i in enumerate(ranked_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "id": self.ids[i],
                    "document": self.documents[i],
                    "metadata": self.metadatas[i],
                    "combined_score": combined[i],
                    "dense_score": dense_norm[i],
                    "bm25_score": bm25_norm[i],
                }
            )
        return results

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    cfg = load_env_config()
    print(f"[rag_retrieval] collection={cfg['chroma_collection']} embed_model={cfg['embed_model']} top_k={cfg['rag_top_k']}")
    print(f"[rag_retrieval] Chroma mode: embedded PersistentClient at {CHROMA_DB_DIR} "
          f"(CHROMA_HOST/CHROMA_PORT read but unused -- no server in this phase)")

    retriever = RagRetriever(config=cfg)
    print(f"[rag_retrieval] collection count = {retriever.count()} (expected {EXPECTED_CHUNK_COUNT})")

    demo_query = "SSH brute force login attempts many failed passwords"
    print(f"[rag_retrieval] demo query: {demo_query!r}")
    for r in retriever.retrieve(demo_query, top_k=cfg["rag_top_k"]):
        meta = r["metadata"]
        print(
            f"  #{r['rank']} id={r['id']} technique_id={meta.get('technique_id')} "
            f"chunk_type={meta.get('chunk_type')} combined={r['combined_score']:.4f} "
            f"(dense={r['dense_score']:.4f}, bm25={r['bm25_score']:.4f})"
        )
