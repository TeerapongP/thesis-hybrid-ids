# RAG Layer — Chunking Plan (Phase 2)

Records the approved chunking design for `data/knowledge_base/mitre_techniques.json` (Phase 1's MITRE Enterprise ATT&CK v19.1 knowledge base) ahead of Phase 3 (ChromaDB indexing + hybrid retrieval). This document did not previously exist on disk — the design had only been agreed in conversation; this file makes it verifiable and version-controlled going forward.

## Source data

`data/knowledge_base/mitre_techniques.json` contains 11 top-level technique entries, scoped exactly to the `technique_id` values referenced by `config/mitre_mapping.csv` (both CICIDS-2017's 14 attack `Label` categories and UNSW-NB15/CIC-UNSW-NB15's `attack_cat` categories):

```
T1046, T1059, T1071.001, T1110, T1110.001, T1190, T1210, T1498, T1499.002, T1505.003, T1595
```

Each of these 11 entries has three text-bearing fields: `description`, `detection`, and a `mitigations` list (each mitigation itself has an `mitigation_id`, `name`, `description`).

## Approved chunk design: 45 chunks

| Chunk type | Count | Basis |
|---|---|---|
| Description | 11 | One chunk per technique's `description` field — 1:1 with the 11 techniques. |
| Detection | 11 | One chunk per technique's `detection` field — 1:1 with the 11 techniques. |
| Mitigation (deduplicated) | 23 | See below. |
| **Total** | **45** | |

### Why mitigation chunks are deduplicated, not 1:1

Across the 11 techniques there are **43 raw mitigation entries** (i.e. techniques collectively list 43 `(technique_id, mitigation)` pairs), but only **23 unique `mitigation_id` values** — the same mitigation is frequently attached to multiple techniques (e.g. `M1030` Network Segmentation appears under 3 different techniques, `M1037` Filter Network Traffic under 4). Chunking on the raw 43 would mean redundant near-duplicate text in the vector store and would double-count the same mitigation's relevance signal across techniques it's attached to. Deduplicating by `mitigation_id` avoids both problems: **13 of the 23 unique mitigation IDs are shared across 2–4 techniques** (`M1030` ×3, `M1031` ×2, `M1042` ×4, `M1026` ×3, `M1037` ×4, `M1018` ×2, `M1027` ×2, `M1032` ×2, `M1036` ×2, `M1051` ×3, `M1016` ×2, `M1048` ×2, `M1050` ×2), and the remaining 10 appear under exactly one technique.

**Dedup rule**: keep the first-occurrence description for each `mitigation_id`. This is safe here — verified empirically (2026-07-14) that **zero mitigation IDs have conflicting description text** across the techniques that share them, so first-occurrence and any-occurrence are equivalent; there is no judgment call being made.

Each deduplicated mitigation chunk's metadata should record every `technique_id` it applies to (not just the one it was first seen under), so retrieval against any of its parent techniques still surfaces it.

### Why mitigation text isn't merged with description/detection into one chunk-per-technique

Mitigation descriptions vary enormously in length — from 74 to 3,492 characters — and a single technique can carry up to 4 mitigations. Folding all of a technique's mitigation text into one combined chunk would produce wildly uneven chunk sizes (undermining embedding quality and retrieval consistency) and would also break the earlier "no redundant duplicate text" rationale, since a shared mitigation would then be re-embedded in full inside every technique's combined chunk instead of once. Keeping description, detection, and each deduplicated mitigation as separate, independently retrievable chunks keeps chunk sizes comparable and avoids re-embedding the same mitigation text repeatedly.

## Sub-technique scope (see also HANDOFF_RAG_PHASE3.md's investigation note)

The 11 top-level entries are the only entries with chunkable text. Some of them (`T1059`, `T1498`, `T1595`, `T1110`) also list a `sub_techniques` array containing bare `{technique_id, name}` stubs (e.g. `T1059.001` PowerShell) with **no description/detection/mitigations of their own** — these are not separate top-level entries in the JSON and must **not** be treated as chunkable content. The exception is `T1110.001`, which *is* both listed as a sub-technique stub under `T1110` and present as its own full top-level entry (because `config/mitre_mapping.csv` maps FTP-/SSH-Patator to `T1110.001` specifically, distinct from `T1110` generic Brute Force used for Web Attack - Brute Force) — this is intentional dual coverage, not a duplicate to be collapsed.

**Rule for chunking code**: iterate only over the top-level keys of `techniques` (11 entries); ignore the `sub_techniques` field entirely except as read-only metadata — never attempt to derive a description/detection chunk from a `sub_techniques` stub entry.

## Evaluation query set

Precision@3 is evaluated against CICIDS-2017's **14** attack categories from `Label` that carry a known `technique_id` in `config/mitre_mapping.csv` (`BENIGN` is the 15th `Label` value but maps to no `technique_id` and is excluded, not silently dropped).
