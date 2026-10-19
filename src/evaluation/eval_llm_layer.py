
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.llm.llm_explain import _technique_context, build_prompt, parse_llm_response

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reports" / "llm_layer_eval_results.csv"
REPORT_PATH = ROOT / "reports" / "llm_layer_report.txt"


def _hit(technique_id: str) -> dict:
    return {"metadata": {"technique_id": technique_id}, "document": "evidence"}


def run_cases() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for case_id, context in (
        ("TC-ENT-01", {"source_ip": "10.20.14.87", "dst_port": 445, "affected_asset": "FS-PRINT02.campus.local"}),
        ("TC-ENT-02", {"source_ip": "203.0.113.44", "dst_port": 22, "affected_asset": "AUTH-SVR01.campus.local"}),
    ):
        prompt = build_prompt(context, [_hit("T1046"), _hit("T1046")], 0.91, 1, 0.91)
        passed = all(str(value) in prompt for value in context.values())
        rows.append({"case_id": case_id, "passed": str(passed).lower(), "detail": "technical entities preserved in prompt"})

    invalid_payloads = (
        ("TC-FAITH-01", {"priority": "URGENT", "explanation_th": "x", "remediation": []}),
        ("TC-FAITH-02", {"priority": "HIGH", "explanation_th": "x", "remediation": "not-a-list"}),
    )
    for case_id, payload in invalid_payloads:
        try:
            parse_llm_response(json.dumps(payload))
            passed = False
        except ValueError:
            passed = True
        rows.append({"case_id": case_id, "passed": str(passed).lower(), "detail": "malformed output rejected"})

    for case_id, wrong_ids in (
        ("TC-ADV-01", ("T1046", "T1499.002", "T1498")),
        ("TC-ADV-02", ("T1071.001", "T1499.002", "T1110")),
    ):
        technique_id, tactic = _technique_context([_hit(value) for value in wrong_ids])
        passed = technique_id is None and tactic is None
        rows.append({"case_id": case_id, "passed": str(passed).lower(), "detail": "disputed RAG hits cause abstention"})
    return rows


def main() -> int:
    rows = run_cases()
    aggregate = {"case_id": "aggregate", "passed": str(all(r["passed"] == "true" for r in rows)).lower(), "detail": f"{sum(r['passed'] == 'true' for r in rows)}/{len(rows)} passed"}
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "passed", "detail"))
        writer.writeheader()
        writer.writerows(rows + [aggregate])
    REPORT_PATH.write_text(
        "LLM Layer Offline Contract Evaluation\n"
        "=====================================\n"
        f"Result: {aggregate['detail']}\n"
        "Scope: prompt/entity preservation, JSON validation, and adversarial RAG abstention.\n"
        "Live Typhoon quality/latency is pending because no Ollama runtime is installed.\n",
        encoding="utf-8",
    )
    return 0 if aggregate["passed"] == "true" else 1


if __name__ == "__main__":
    raise SystemExit(main())
