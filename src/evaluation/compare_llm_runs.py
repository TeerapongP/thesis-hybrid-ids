"""Side-by-side comparison of two eval_llm_faithfulness runs, including the diversity figures.

Run under uv (it renders the few-shot examples through llm_explain):
    uv run python -m src.evaluation.compare_llm_runs reports/llm_faithfulness_raw.mapping.jsonl \
        reports/llm_faithfulness_raw.technique_kb.jsonl

Diversity is counted over exact strings: a distinct explanation is a distinct `explanation_th`,
a distinct remediation list is a distinct ordered tuple of items. Only status=ok records count,
so a run with a timed-out alert has one fewer row than the sample.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation import eval_llm_faithfulness as m2
from src.llm import llm_explain


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rendered_remediations(rec: dict[str, Any]) -> set[tuple[str, ...]]:
    facts = {"alert_context": rec["alert_context"], "priority": rec["priority"],
             "technique_id": rec.get("ctx_technique_id"), "mitre_tactic": rec.get("ctx_tactic")}
    return {tuple(shot["output"]["remediation"]) for shot in llm_explain.render_few_shots(facts)}


def diversity(ok: list[dict[str, Any]]) -> dict[str, Any]:
    explanations = Counter(r["output"]["explanation_th"] for r in ok)
    remediations = Counter(tuple(r["output"]["remediation"]) for r in ok)
    copied = sum(1 for r in ok if tuple(r["output"]["remediation"]) in rendered_remediations(r))
    return {
        "distinct_explanations": len(explanations),
        "top_explanation_count": max(explanations.values(), default=0),
        "distinct_remediation_lists": len(remediations),
        "top_remediation_count": max(remediations.values(), default=0),
        "top3_remediation_counts": [c for _, c in remediations.most_common(3)],
        "remediation_equals_rendered_example": copied,
    }


def entity_failures(ok: list[dict[str, Any]]) -> dict[str, Any]:
    omitted: Counter = Counter()
    foreign = 0
    failing = omission_only = 0
    for r in ok:
        e = r["checks"]["entity"]
        if e["pass"]:
            continue
        failing += 1
        omission_only += int(bool(e["missing"]) and e["corruption_free"])
        foreign += int(not e["corruption_free"])
        omitted.update(m.split(":")[0] for m in e["missing"])
    return {"entity_fail": failing, "fail_omission_only": omission_only,
            "fail_with_foreign_or_altered": foreign, "omitted_by_type": dict(omitted)}


def summarize_run(path: Path) -> dict[str, Any]:
    records = load(path)
    ok = [r for r in records if r.get("status") == "ok"]
    s = m2.summarize(records)
    out: dict[str, Any] = {"n_total": len(records), "n_ok": len(ok)}
    out.update(diversity(ok))
    out["example_echo_alerts"] = f'{s["example_echo_alerts"]["num"]}/{s["example_echo_alerts"]["den"]}'
    out["example_echo_ratio_mean"] = s["example_echo_ratio_mean"]
    out["entity_pass"] = f'{s["entity_pass"]["num"]}/{s["entity_pass"]["den"]}'
    out.update(entity_failures(ok))
    out["entity_corruption_free"] = f'{s["entity_corruption_free"]["num"]}/{s["entity_corruption_free"]["den"]}'
    out["alerts_with_bad_claim"] = f'{s["alerts_with_bad_claim"]["num"]}/{s["alerts_with_bad_claim"]["den"]}'
    out["hallucination_lower_bound"] = round(s["claims"]["hallucination_lower_bound"]["rate"], 4)
    out["undecidable_share"] = round(s["claims"]["undecidable_share"]["rate"], 4)
    out["alerts_with_technique"] = sum(1 for r in ok if r["output"].get("technique_id"))
    out["prose_technique_mention_alerts"] = s["prose_technique_mention_alerts"]["num"]
    out["prompt_tokens_mean"] = round(s["throughput"]["prompt_tokens_mean"], 1)
    return out


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = [Path(a) for a in argv]
    if len(paths) < 2:
        print(__doc__)
        return 2
    results = [summarize_run(p) for p in paths]
    keys = list(results[0])
    width = max(len(k) for k in keys)
    print(f'{"metric":<{width}}  ' + "  ".join(p.name for p in paths))
    for key in keys:
        print(f"{key:<{width}}  " + "  ".join(str(r[key]) for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
