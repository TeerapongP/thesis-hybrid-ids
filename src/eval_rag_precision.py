"""
eval_rag_precision.py
======================
Precision@3 evaluation of the hybrid RAG retriever (src/rag_retrieval.py) against
config/mitre_mapping.csv's CICIDS-2017 attack categories as ground truth.

This is the evaluation query set ONLY -- config/mitre_mapping.csv is never read by
rag_retrieval.py's chunking/embedding/indexing code (see that module's docstring), just here,
after the index already exists, to score retrieval quality. This mirrors the project's
established fit-on-train-only / no-leakage discipline applied to retrieval instead of ML
training.

Query set: all `dataset == "cicids2017"` rows in config/mitre_mapping.csv EXCEPT `BENIGN`
(which maps to no technique_id and is the 15th Label value, not an attack category) -- 14 rows.
The query text used for each row is the attack category name itself (`source_label`, e.g. "DoS
Hulk", "PortScan"), NOT `technique_name` -- `technique_name` is derived from the ground-truth
technique_id itself, so using it as the query text would leak the answer into the query and
make the evaluation trivial. `source_label` is what a real downstream consumer (e.g. Random
Forest's predicted multi-class label) would actually have at query time.

Relevance rule: a retrieved chunk counts as relevant to a category's expected technique_id if
either (a) its `technique_id` metadata field matches (description/detection chunks, or a
mitigation chunk's primary/first-seen technique_id), or (b) for mitigation chunks, the expected
technique_id appears anywhere in its comma-separated `technique_ids` metadata field (a shared
mitigation like M1037 must count as relevant against every technique it applies to, not just
the first one it was seen under).

Precision@3 per category = (# relevant chunks in the top-3) / 3 -- standard precision@k. This
repo's convention (see reports/random_forest_val_metrics.csv, isolation_forest_loao_results.csv)
is to report per-category, never a single blended number -- see reports/rag_retrieval_precision_at_3.csv.

Outputs:
- reports/rag_retrieval_precision_at_3.csv (14 per-category rows + 1 aggregate row)
- prints a summary to stdout
"""

from __future__ import annotations

import csv
from pathlib import Path

from rag_retrieval import RagRetriever, load_env_config

ROOT = Path(__file__).resolve().parent.parent
MAPPING_CSV = ROOT / "config" / "mitre_mapping.csv"
OUTPUT_CSV = ROOT / "reports" / "rag_retrieval_precision_at_3.csv"


def load_query_set(path: Path = MAPPING_CSV) -> list[dict[str, str]]:
    """CICIDS-2017 attack categories (Label values) with a known technique_id, BENIGN excluded."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "cicids2017":
                continue
            if row["source_label"] == "BENIGN":
                continue
            if not row["technique_id"]:
                continue
            rows.append(row)
    return rows


def is_relevant(metadata: dict, expected_technique_id: str) -> bool:
    if metadata.get("technique_id") == expected_technique_id:
        return True
    technique_ids_field = metadata.get("technique_ids")
    if technique_ids_field and expected_technique_id in technique_ids_field.split(","):
        return True
    return False


def evaluate(retriever: RagRetriever, query_set: list[dict[str, str]], top_k: int = 3) -> list[dict]:
    results = []
    for row in query_set:
        category = row["source_label"]
        expected_tid = row["technique_id"]
        hits = retriever.retrieve(category, top_k=top_k)

        relevant_flags = [is_relevant(h["metadata"], expected_tid) for h in hits]
        n_relevant = sum(relevant_flags)
        precision_at_k = n_relevant / top_k

        retrieved_ids = [h["id"] for h in hits]
        # For mitigation chunks, show the FULL applicable technique_id list (not just the
        # primary/first-seen one) so a reader can see why a mitigation chunk was scored
        # relevant to a category whose expected technique_id isn't the primary one.
        retrieved_tids = [
            h["metadata"]["technique_ids"] if h["metadata"].get("chunk_type") == "mitigation" else h["metadata"].get("technique_id")
            for h in hits
        ]

        results.append(
            {
                "category": category,
                "query_text": category,
                "expected_technique_id": expected_tid,
                "expected_technique_name": row.get("technique_name", ""),
                "precision_at_3": precision_at_k,
                "hit_at_3": int(n_relevant > 0),
                "n_relevant_in_top3": n_relevant,
                "retrieved_chunk_ids": "|".join(retrieved_ids),
                "retrieved_technique_ids": "|".join(str(t) for t in retrieved_tids),
            }
        )
    return results


def write_report_csv(results: list[dict], output_path: Path = OUTPUT_CSV) -> None:
    n = len(results)
    mean_precision = sum(r["precision_at_3"] for r in results) / n
    mean_hit_rate = sum(r["hit_at_3"] for r in results) / n

    fieldnames = [
        "category",
        "query_text",
        "expected_technique_id",
        "expected_technique_name",
        "precision_at_3",
        "hit_at_3",
        "n_relevant_in_top3",
        "retrieved_chunk_ids",
        "retrieved_technique_ids",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "# Precision@3 -- RAG retrieval (src/rag_retrieval.py) vs. config/mitre_mapping.csv ground truth\n"
        )
        f.write(
            "# BENIGN excluded explicitly: it is the 15th CICIDS-2017 Label value but maps to no technique_id "
            "in config/mitre_mapping.csv, so it is not part of this evaluation's query set (14 attack categories only)\n"
        )
        f.write("# per-category (14 rows, one per CICIDS-2017 attack Label with a known technique_id)\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

        f.write("#\n")
        f.write("# aggregate (macro mean across the 14 categories -- not a single blended retrieval run)\n")
        agg_writer = csv.DictWriter(f, fieldnames=["scope", "n_categories", "mean_precision_at_3", "mean_hit_at_3"])
        agg_writer.writeheader()
        agg_writer.writerow(
            {
                "scope": "aggregate_macro",
                "n_categories": n,
                "mean_precision_at_3": mean_precision,
                "mean_hit_at_3": mean_hit_rate,
            }
        )


if __name__ == "__main__":
    cfg = load_env_config()
    query_set = load_query_set()
    print(f"[eval_rag_precision] query set size = {len(query_set)} (expected 14, BENIGN excluded)")
    assert len(query_set) == 14, f"expected 14 CICIDS-2017 attack categories with a known technique_id, got {len(query_set)}"

    retriever = RagRetriever(config=cfg)
    print(f"[eval_rag_precision] collection count = {retriever.count()}")

    results = evaluate(retriever, query_set, top_k=3)
    write_report_csv(results)

    mean_p = sum(r["precision_at_3"] for r in results) / len(results)
    mean_hit = sum(r["hit_at_3"] for r in results) / len(results)
    print(f"[eval_rag_precision] mean precision@3 = {mean_p:.4f}  mean hit@3 = {mean_hit:.4f}")
    for r in results:
        print(
            f"  {r['category']:<28} expected={r['expected_technique_id']:<10} "
            f"precision@3={r['precision_at_3']:.2f}  hit@3={r['hit_at_3']}  "
            f"retrieved={r['retrieved_technique_ids']}"
        )
    print(f"[eval_rag_precision] wrote {OUTPUT_CSV}")
