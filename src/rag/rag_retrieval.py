"""
rag_retrieval.py
=================
RAG Layer -- Phase 3 (ChromaDB setup + hybrid BM25 + dense-vector retrieval).

Indexes the 45-chunk MITRE ATT&CK knowledge base (Phase 1: data/knowledge_base/mitre_techniques.json,
Phase 2 chunking design: docs/rag_chunking_plan.md -- 11 description + 11 detection + 23
deduplicated mitigation chunks) into an embedded ChromaDB PersistentClient collection, and
exposes a hybrid BM25 (sparse/exact) + dense-vector (semantic, BAAI/bge-m3) retriever on top
of it.

Design precedent: mirrors src/modeling/fusion_layer.py's pattern of code-extracting a decided strategy
into a standalone, importable module with a JSON config sidecar (models/rag_retrieval_config.json)
recording the chosen approach and the reasoning behind it, rather than leaving the reasoning
in a notebook cell or code comment.

Chroma deployment mode: embedded `chromadb.PersistentClient()`, not a server (see
docs/handoffs/HANDOFF_RAG_PHASE3.md -- no docker-compose for Chroma exists in this repo, nothing listens on
CHROMA_HOST:CHROMA_PORT, Docker Desktop isn't running on this machine). CHROMA_HOST/CHROMA_PORT
are still read from .env for completeness/documentation but are unused in this phase.

Retrieval strategy: custom hybrid (BM25 via rank_bm25 + dense cosine similarity via ChromaDB),
NOT LangChain -- see models/rag_retrieval_config.json's "retrieval_strategy" section for the
full reasoning. In short: the knowledge base is a small, static, already-known 45-chunk corpus
(not a live/growing document store), this repo has zero prior LangChain usage and an established
precedent (fusion_layer.py, feature_extractor.py, ...) of small hand-written modules over heavy
framework dependencies, and a from-scratch weighted combination of ChromaDB's own dense query
API with rank_bm25's sparse scoring is a handful of lines that keeps every weighting/normalization
decision visible and directly tunable -- exactly what this repo's evaluation-per-category culture
expects to be able to inspect and justify, rather than delegate to a retriever-chain abstraction.

No leakage: config/mitre_mapping.csv (the evaluation query set) is never read or referenced by
any function in this module -- see eval_rag_precision.py for the separate script that reads it.
"""

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

# Default hybrid weighting (dense vs. BM25) -- also recorded in models/rag_retrieval_config.json.
# Equal weighting by default: the source diagram (thesis_dataflow_v2.drawio) lists BM25 (exact
# match) and dense vector (semantic) side by side with no stated priority between them.
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5


# ----------------------------------------------------------------------------
# Config (.env) -- never hardcode these, never print/log the raw .env contents
# ----------------------------------------------------------------------------
def load_env_config(env_path: Path = ENV_PATH) -> dict[str, Any]:
    """Read RAG-relevant config from the root .env. Does not read/return OPENCTI_TOKEN or any
    other secret -- only the six RAG scaffolding keys this module actually needs."""
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


# ----------------------------------------------------------------------------
# Chunk building -- Phase 2's approved 45-chunk design (docs/rag_chunking_plan.md), applied here
# ----------------------------------------------------------------------------
def load_mitre_techniques(path: Path = MITRE_JSON_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks(mitre_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the 45 chunks: 11 description + 11 detection + 23 deduplicated mitigation.

    Iterates ONLY over the top-level keys of techniques (11 entries) -- sub_techniques stub
    entries ({technique_id, name}, no text) are never touched, per docs/rag_chunking_plan.md /
    HANDOFF_RAG_PHASE3.md's sub-technique scoping rule. T1110.001 is both a sub_techniques stub
    under T1110 AND its own full top-level entry -- it is chunked here as its own top-level
    entry (independently of the T1110 stub list), which is the intended dual coverage, not a
    duplicate.

    Mitigation dedup rule (docs/rag_chunking_plan.md): keep the first-occurrence description
    text for each mitigation_id (verified conflict-free across sharing techniques); each
    deduplicated mitigation chunk's metadata records every technique_id it applies to, in
    first-encountered order.
    """
    techniques: dict[str, Any] = mitre_data["techniques"]
    top_level_ids = list(techniques.keys())  # exactly the 11 top-level entries

    chunks: list[dict[str, Any]] = []

    # --- description + detection chunks: 1:1 with the 11 top-level entries ---
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

    mitigation_first_seen: dict[str, dict[str, Any]] = {}  # mitigation_id -> {name, description}
    mitigation_technique_ids: dict[str, list[str]] = {}  # mitigation_id -> [technique_id, ...] in encounter order

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
                    # mandatory technique_id field: primary = first technique this mitigation
                    # was encountered under; the FULL list is carried in technique_ids below so
                    # retrieval against any of its parent techniques still surfaces it.
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


# ----------------------------------------------------------------------------
# Embedding (BAAI/bge-m3, read from .env -- never hardcoded)
# ----------------------------------------------------------------------------
_embedder_cache: dict[str, Any] = {}


def get_embedder(model_name: str):
    """Lazily load and cache a sentence-transformers embedder for `model_name` (from .env's
    EMBED_MODEL). First call downloads the model weights (~2GB+ for BAAI/bge-m3) to the local
    HuggingFace cache -- subsequent calls (including fresh-process re-runs) reuse the cached
    weights on disk, so this is not re-downloaded per process."""
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer

        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_texts(embedder, texts: list[str]) -> list[list[float]]:
    """BGE models are trained/intended for cosine similarity with normalized embeddings."""
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ----------------------------------------------------------------------------
# ChromaDB index build (idempotent -- upsert on deterministic chunk ids)
# ----------------------------------------------------------------------------
def get_chroma_client(db_dir: Path = CHROMA_DB_DIR) -> chromadb.ClientAPI:
    db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_dir))


def build_index(
    client: chromadb.ClientAPI | None = None,
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> chromadb.Collection:
    """(Re)build the mitre_attack_kb collection from data/knowledge_base/mitre_techniques.json.

    Idempotent: chunk ids are deterministic (f"{technique_id}::description" etc.), so this uses
    `collection.upsert` -- re-running never accumulates duplicates. Skips re-embedding entirely
    if the collection already holds exactly the expected set of chunk ids and `force=False`.
    The check is on ids, not on text, so a content-only change to an existing chunk is not
    noticed without `force=True` (rag_reindex_service passes it). Ids left in the collection
    from an earlier chunking run are deleted after the upsert.
    """
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

    # Stale ids were deleted above; this guard confirms the count really is EXPECTED_CHUNK_COUNT
    # rather than silently trusting upsert and delete.
    if collection.count() != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            f"mitre_attack_kb collection has {collection.count()} chunks after build, "
            f"expected {EXPECTED_CHUNK_COUNT}."
        )

    return collection


# ----------------------------------------------------------------------------
# Tokenization for BM25 (simple, deterministic -- lowercase + word-boundary split)
# ----------------------------------------------------------------------------
# Hyphen is deliberately NOT in this character class (unlike an earlier version of this
# regex): a hyphen almost never signals a single semantic unit that should be matched
# atomically -- "FTP-Patator" and "well-known" both read, for BM25 purposes, as two
# separate words ("ftp"/"patator", "well"/"known"), not one OOV compound token. Keeping
# the hyphen inside the token class silently zeroed out BM25's contribution for any
# hyphenated query (e.g. "FTP-Patator", "SSH-Patator") whose constituent words (FTP, SSH)
# do appear verbatim in the corpus (see T1110.001's description) -- the compound token
# just never matched. This is a general tokenizer fix, not tuned to any one category.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _minmax_norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ----------------------------------------------------------------------------
# Hybrid retriever
# ----------------------------------------------------------------------------
class RagRetriever:
    """Hybrid BM25 + dense-vector retriever over the mitre_attack_kb ChromaDB collection.

    Survives a fresh process: all state (chunk documents/metadatas for the BM25 corpus, the
    embedder, the Chroma collection handle) is rebuilt from disk on `__init__` -- nothing here
    depends on any notebook or previous-process in-memory state. The persisted ChromaDB
    collection at data/chroma_db/ is the only on-disk state this class reads.
    """

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
        """Return up to `top_k` chunks ranked by a weighted combination of dense (cosine, via
        ChromaDB) and BM25 (sparse, via rank_bm25) scores, each min-max normalized across the
        full 45-chunk universe before combining (equal weighting by default; see
        models/rag_retrieval_config.json)."""
        top_k = top_k or self.config["rag_top_k"]
        n = len(self.ids)

        # Dense side: query the persisted Chroma collection over the WHOLE corpus (n=45 is
        # trivially small) so every chunk gets a comparable similarity score to combine with BM25.
        query_embedding = embed_texts(self.embedder, [query])[0]
        dense_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["distances"],
        )
        dense_sim_by_id = {
            cid: 1.0 - dist  # cosine space: similarity = 1 - distance
            for cid, dist in zip(dense_result["ids"][0], dense_result["distances"][0])
        }
        dense_scores_ordered = [dense_sim_by_id[cid] for cid in self.ids]
        dense_norm = _minmax_norm(dense_scores_ordered)

        # Sparse side: BM25 over the same fixed-order corpus.
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
