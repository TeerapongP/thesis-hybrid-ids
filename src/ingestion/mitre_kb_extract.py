"""
mitre_kb_extract.py
====================
RAG Layer -- Phase 1 extraction logic, made reusable.

Fetches the MITRE Enterprise ATT&CK STIX bundle from the validated source
(mitre-attack/attack-stix-data on GitHub -- see CLAUDE.md's "RAG Layer -- knowledge base
source decision" section for why this is used instead of the locally running OpenCTI
instance, a decision investigated and closed permanently on 2026-07-17) and extracts the
11 target techniques into the exact dict shape already checked in at
data/knowledge_base/mitre_techniques.json.

This module did not exist before this file: mitre_techniques.json was originally produced
ad hoc in an earlier session with no script left behind (confirmed via `git log` -- the
file was checked in directly in the initial commit). This is that logic, written for the
first time as an importable, reusable module -- schema fields below were reverse-verified
field-by-field against the live bundle before writing this, not guessed.

Schema note (confirmed live, 2026-07-18): the current STIX bundle has NO flat
`x_mitre_detection` text field on attack-pattern objects any more -- detection guidance
now lives in separate `x-mitre-detection-strategy` / `x-mitre-analytic` objects, linked via
`detects` relationships (detection-strategy -> attack-pattern) and each strategy's
`x_mitre_analytic_refs` list. This is exactly the gap CLAUDE.md's OpenCTI investigation
found (OpenCTI has zero support for these two object types) -- reconstruct_detection_text()
below rebuilds the same "Behavioral Detection Strategy for ... \n- Analytic NNNN: ... [Log
sources: ...]" text format found in the existing mitre_techniques.json, from these objects,
matching it exactly (verified against the live T1046 entry: strategy name, analytic
name/description/log-source-names all line up character-for-character with what's already
in the checked-in file).

Only extraction (fetch + parse). No diffing, no ChromaDB, no scheduling, no file writes --
see src/rag/rag_reindex_service.py for the scheduled service built on top of this module.
"""

from __future__ import annotations

from typing import Any

import requests

STIX_BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)

# The 11 top-level technique_ids this knowledge base covers -- fixed by the approved
# Phase 1/2 scope (docs/rag_chunking_plan.md), not re-derived from anywhere else.
TARGET_TECHNIQUE_IDS = [
    "T1046",
    "T1059",
    "T1071.001",
    "T1110",
    "T1110.001",
    "T1190",
    "T1210",
    "T1498",
    "T1499.002",
    "T1505.003",
    "T1595",
]


def fetch_stix_bundle(url: str | None = None, timeout: int = 180) -> dict[str, Any]:
    """Fetch the full Enterprise ATT&CK STIX bundle. ~50MB, ~26k objects -- no caching here,
    caller decides how often to call this (see rag_reindex_service.py).

    `url` defaults to the CURRENT value of module-level STIX_BUNDLE_URL, read inside the
    function body rather than as a default-argument value (a default argument is bound once
    at function-definition time, so a caller/test that reassigns
    mitre_kb_extract.STIX_BUNDLE_URL -- e.g. to redirect this at a local fixture bundle --
    would otherwise be silently ignored). Same reasoning applies to extract_techniques'/
    build_knowledge_base's `technique_ids` default below; see rag_reindex_service.py's
    load_current_kb docstring for the same bug class caught and fixed there first."""
    if url is None:
        url = STIX_BUNDLE_URL
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _mitre_external_id(stix_obj: dict[str, Any]) -> str | None:
    for ref in stix_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _mitre_url(stix_obj: dict[str, Any]) -> str | None:
    for ref in stix_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url")
    return None


def reconstruct_detection_text(
    attack_pattern_id: str,
    objects_by_id: dict[str, dict[str, Any]],
    detects_by_target: dict[str, list[dict[str, Any]]],
) -> str:
    """Rebuild the "Behavioral Detection Strategy for X\\n- Analytic NNNN: ... [Log sources:
    ...]" text from x-mitre-detection-strategy / x-mitre-analytic objects, since no flat
    detection field exists on the attack-pattern object any more (see module docstring).
    Returns "" if the technique has no linked, non-deprecated detection strategy (should not
    happen for any of the 11 target techniques as of the 2026-07-18 schema check, but a
    future MITRE release could in principle drop one -- handled explicitly rather than
    letting a KeyError propagate).

    Both the strategy list and each strategy's analytic list are sorted by their own MITRE
    external_id (e.g. "DET0376", "AN1057") rather than left in raw STIX relationship/list
    encounter order -- matching the explicit-sort discipline used everywhere else in this
    module (sub_techniques, mitigations, log_source_names). Without this, a STIX bundle
    re-export with a different internal object/list ordering but IDENTICAL real content
    could produce a byte-different detection string, which rag_reindex_service.py's
    diff_techniques() would then misreport as a genuine content change and trigger an
    unnecessary backup+reindex."""
    strategy_rels = sorted(
        detects_by_target.get(attack_pattern_id, []),
        key=lambda rel: (_mitre_external_id(objects_by_id.get(rel["source_ref"], {})) or rel["source_ref"]),
    )
    lines: list[str] = []
    for rel in strategy_rels:
        strategy = objects_by_id.get(rel["source_ref"])
        if strategy is None or strategy.get("revoked") or strategy.get("x_mitre_deprecated"):
            continue
        lines.append(strategy["name"])
        analytic_refs = sorted(
            strategy.get("x_mitre_analytic_refs", []),
            key=lambda ref: (_mitre_external_id(objects_by_id.get(ref, {})) or ref),
        )
        for analytic_ref in analytic_refs:
            analytic = objects_by_id.get(analytic_ref)
            if analytic is None:
                continue
            log_source_names = sorted(
                {ls["name"] for ls in analytic.get("x_mitre_log_source_references", []) if ls.get("name")}
            )
            log_sources = ", ".join(log_source_names)
            lines.append(f"- {analytic['name']}: {analytic['description']} [Log sources: {log_sources}]")
    return "\n".join(lines)


def extract_techniques(
    bundle: dict[str, Any], technique_ids: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Extract `technique_ids` from `bundle` into the schema already used by
    data/knowledge_base/mitre_techniques.json. Raises if any requested technique_id is not
    found (fail loudly -- a missing technique means the scope itself needs a human decision,
    not a silent partial knowledge base).

    `technique_ids` defaults to the current TARGET_TECHNIQUE_IDS, read inside the function
    body -- see fetch_stix_bundle's docstring for why this isn't a plain default-argument
    value."""
    if technique_ids is None:
        technique_ids = TARGET_TECHNIQUE_IDS

    objects = bundle["objects"]
    objects_by_id = {o["id"]: o for o in objects}

    # Prefer a non-revoked/non-deprecated attack-pattern when the same external_id (e.g. a
    # technique renumbered/merged upstream) resolves to more than one STIX object -- ATT&CK
    # data has real precedent for this (a revoked stub can retain the old ID alongside the
    # current object). Without this, the plain "last object wins" behavior below would pick
    # whichever happens to appear later in bundle["objects"]'s list order, silently risking
    # the revoked/deprecated one -- not exercised by the current bundle (verified byte-for-
    # byte against the checked-in KB with zero collisions), but a real bundle-format
    # possibility, so guarded here rather than left to chance on a future re-fetch.
    attack_patterns_by_ext_id: dict[str, dict[str, Any]] = {}
    for o in objects:
        if o["type"] != "attack-pattern":
            continue
        ext_id = _mitre_external_id(o)
        if not ext_id:
            continue
        existing = attack_patterns_by_ext_id.get(ext_id)
        if existing is None:
            attack_patterns_by_ext_id[ext_id] = o
        else:
            existing_bad = bool(existing.get("revoked") or existing.get("x_mitre_deprecated"))
            new_bad = bool(o.get("revoked") or o.get("x_mitre_deprecated"))
            if existing_bad and not new_bad:
                attack_patterns_by_ext_id[ext_id] = o
            elif existing_bad == new_bad:
                raise RuntimeError(
                    f"technique_id {ext_id!r} resolves to more than one attack-pattern object "
                    f"with the same revoked/deprecated status ({existing['id']} vs {o['id']}) -- "
                    f"ambiguous, needs a human decision rather than a silent 'last one wins' pick"
                )

    missing = [tid for tid in technique_ids if tid not in attack_patterns_by_ext_id]
    if missing:
        raise RuntimeError(f"technique_id(s) not found in fetched STIX bundle: {missing}")

    # relationship indices, built once and reused across all 11 techniques
    mitigates_by_target: dict[str, list[dict[str, Any]]] = {}
    detects_by_target: dict[str, list[dict[str, Any]]] = {}
    subtechnique_of_by_target: dict[str, list[dict[str, Any]]] = {}
    for o in objects:
        if o["type"] != "relationship" or o.get("revoked"):
            continue
        rtype = o.get("relationship_type")
        target = o.get("target_ref")
        if rtype == "mitigates":
            mitigates_by_target.setdefault(target, []).append(o)
        elif rtype == "detects":
            detects_by_target.setdefault(target, []).append(o)
        elif rtype == "subtechnique-of":
            subtechnique_of_by_target.setdefault(target, []).append(o)

    techniques: dict[str, dict[str, Any]] = {}
    for tid in technique_ids:
        ap = attack_patterns_by_ext_id[tid]

        tactics = sorted({kcp["phase_name"] for kcp in ap.get("kill_chain_phases", [])})

        is_subtechnique = bool(ap.get("x_mitre_is_subtechnique"))
        parent_technique_id = None
        if is_subtechnique:
            parent_rels = [
                r for r in objects
                if r.get("type") == "relationship"
                and r.get("relationship_type") == "subtechnique-of"
                and r.get("source_ref") == ap["id"]
                and not r.get("revoked")
            ]
            if parent_rels:
                parent = objects_by_id.get(parent_rels[0]["target_ref"])
                parent_technique_id = _mitre_external_id(parent) if parent else None

        sub_techniques = []
        for rel in subtechnique_of_by_target.get(ap["id"], []):
            child = objects_by_id.get(rel["source_ref"])
            if child is None or child.get("revoked") or child.get("x_mitre_deprecated"):
                continue
            child_id = _mitre_external_id(child)
            if child_id:
                sub_techniques.append({"technique_id": child_id, "name": child["name"]})
        sub_techniques.sort(key=lambda s: s["technique_id"])

        mitigations = []
        for rel in mitigates_by_target.get(ap["id"], []):
            coa = objects_by_id.get(rel["source_ref"])
            if coa is None or coa.get("revoked") or coa.get("x_mitre_deprecated"):
                continue
            mid = _mitre_external_id(coa)
            if mid:
                mitigations.append({"mitigation_id": mid, "name": coa["name"], "description": coa["description"]})
        mitigations.sort(key=lambda m: m["mitigation_id"])

        techniques[tid] = {
            "technique_id": tid,
            "name": ap["name"],
            "tactics": tactics,
            "is_subtechnique": is_subtechnique,
            "parent_technique_id": parent_technique_id,
            "sub_techniques": sub_techniques,
            "description": ap["description"],
            "detection": reconstruct_detection_text(ap["id"], objects_by_id, detects_by_target),
            "mitigations": mitigations,
            "platforms": list(ap.get("x_mitre_platforms", [])),
            "deprecated": bool(ap.get("x_mitre_deprecated", False)),
            "revoked": bool(ap.get("revoked", False)),
            "attack_version": ap.get("x_mitre_version"),
            "url": _mitre_url(ap),
        }

    return techniques


def build_knowledge_base(bundle: dict[str, Any], technique_ids: list[str] | None = None) -> dict[str, Any]:
    """Build the full top-level dict matching data/knowledge_base/mitre_techniques.json's
    schema: {"source": {...}, "techniques": {...}}.

    `technique_ids` defaults to the current TARGET_TECHNIQUE_IDS, read inside the function
    body -- see fetch_stix_bundle's docstring."""
    if technique_ids is None:
        technique_ids = TARGET_TECHNIQUE_IDS
    collection = next((o for o in bundle["objects"] if o["type"] == "x-mitre-collection"), None)
    if collection is None:
        raise RuntimeError("no x-mitre-collection object found in fetched STIX bundle")

    return {
        "source": {
            "dataset": collection.get("name", "Enterprise ATT&CK"),
            "attack_version": collection.get("x_mitre_version"),
            "stix_spec_version": collection.get("x_mitre_attack_spec_version"),
            "last_modified": collection.get("modified"),
            "fetched_from": "https://github.com/mitre-attack/attack-stix-data (enterprise-attack/enterprise-attack.json)",
        },
        "techniques": extract_techniques(bundle, technique_ids),
    }


if __name__ == "__main__":
    print(f"[mitre_kb_extract] fetching {STIX_BUNDLE_URL} ...")
    bundle = fetch_stix_bundle()
    kb = build_knowledge_base(bundle)
    print(f"[mitre_kb_extract] source.attack_version = {kb['source']['attack_version']}")
    print(f"[mitre_kb_extract] extracted {len(kb['techniques'])} techniques: {list(kb['techniques'].keys())}")
