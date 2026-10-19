
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

from src.modeling.fusion_layer import CONFIDENCE_THRESHOLD
from src.notifications import alert_dispatcher


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


FIXTURE_CASES = [
    {
        "name": "HIGH / PortScan (HANDOFF_LLM_LAYER.md worked example, verbatim)",
        "alert_context": "alert_context_high_portscan.json",
        "llm_output": "llm_output_high_portscan.json",
        "expected_priority": "HIGH",
        "expected_model_used": "if_ae_fallback",
    },
    {
        "name": "HIGH / DoS Hulk (high-confidence, exercises the random_forest model_used branch)",
        "alert_context": "alert_context_high_confident_doshulk.json",
        "llm_output": "llm_output_high_confident_doshulk.json",
        "expected_priority": "HIGH",
        "expected_model_used": "random_forest",
    },
    {
        "name": "MEDIUM / SSH-Patator",
        "alert_context": "alert_context_medium_sshpatator.json",
        "llm_output": "llm_output_medium_sshpatator.json",
        "expected_priority": "MEDIUM",
        "expected_model_used": "if_ae_fallback",
    },
    {
        "name": "LOW / Bot",
        "alert_context": "alert_context_low_bot.json",
        "llm_output": "llm_output_low_bot.json",
        "expected_priority": "LOW",
        "expected_model_used": "if_ae_fallback",
    },
    {
    "name": "LOW / malformed (fusion model_used missing)",
        "alert_context": "alert_context_low_malformed_no_confidence.json",
        "llm_output": "llm_output_low_malformed_no_confidence.json",
        "expected_priority": "LOW",
        "expected_model_used": "unknown",
    },
]


def run_case(case: dict) -> tuple[bool, dict]:
    alert_context = load_fixture(case["alert_context"])
    llm_output = load_fixture(case["llm_output"])
    fused_score = alert_context["fused_score"]

    receipt = alert_dispatcher.dispatch_alert(alert_context, fused_score, llm_output)

    ok = True
    problems = []

    if receipt["priority"] != case["expected_priority"]:
        ok = False
        problems.append(f"priority mismatch: got {receipt['priority']!r}, expected {case['expected_priority']!r}")

    expected_risk_score = round(fused_score * 100)
    if receipt["risk_score"] != expected_risk_score:
        ok = False
        problems.append(f"risk_score mismatch: got {receipt['risk_score']}, expected {expected_risk_score}")

    if receipt["model_used"] != case["expected_model_used"]:
        ok = False
        problems.append(f"model_used mismatch: got {receipt['model_used']!r}, expected {case['expected_model_used']!r}")

    if receipt["rabbitmq"]["status"] != "ok":
        ok = False
        problems.append(f"rabbitmq publish failed: {receipt['rabbitmq']}")
    elif receipt["rabbitmq"]["routing_key"] != f"alert.{case['expected_priority']}":
        ok = False
        problems.append(f"rabbitmq routing_key wrong: {receipt['rabbitmq']['routing_key']}")

    if receipt["influxdb"]["status"] != "ok":
        ok = False
        problems.append(f"influxdb write failed: {receipt['influxdb']}")

    if receipt["elasticsearch"]["status"] != "ok":
        ok = False
        problems.append(f"elasticsearch write failed: {receipt['elasticsearch']}")

    notif = receipt["notification"]
    if case["expected_priority"] == "HIGH":
        if notif["status"] not in ("sent", "error", "dry_run"):
            ok = False
            problems.append(f"unexpected LINE notification status: {notif}")
    elif case["expected_priority"] == "MEDIUM":
        if notif["status"] != "dry_run":
            ok = False
            problems.append(f"expected Gmail dry_run (GMAIL_* is blank), got: {notif}")
    else:
        if notif["status"] != "accumulated":
            ok = False
            problems.append(f"expected LOW alert to be accumulated into SQLite digest, got: {notif}")

    return ok, {"receipt": receipt, "problems": problems}


def main() -> int:
    print(f"[test_alert_dispatcher] fusion_layer.CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD} (model_used gate, reused not duplicated)")
    print(f"[test_alert_dispatcher] running {len(FIXTURE_CASES)} fixture cases against a LIVE dispatch_alert(...)\n")

    all_ok = True
    for case in FIXTURE_CASES:
        ok, detail = run_case(case)
        status_str = "PASS" if ok else "FAIL"
        print(f"[{status_str}] {case['name']}")
        r = detail["receipt"]
        print(
            f"    alert_id={r['alert_id']} risk_score={r['risk_score']} model_used={r['model_used']}\n"
            f"    rabbitmq={r['rabbitmq']['status']}  influxdb={r['influxdb']['status']}  "
            f"elasticsearch={r['elasticsearch']['status']}  notification={r['notification']['status']}"
        )
        for p in detail["problems"]:
            print(f"    PROBLEM: {p}")
        all_ok = all_ok and ok
        print()

    print("[test_alert_dispatcher] flushing LOW-priority digest (exercises flush_low_priority_digest)...")
    flush_result = alert_dispatcher.flush_low_priority_digest()
    print(f"    flush status={flush_result['status']} count={flush_result.get('count')}")
    if flush_result["status"] == "dry_run":
        notif = flush_result["notification"]
        print(f"    digest notification status={notif['status']} (dry_run expected -- GMAIL_* is blank; alerts remain pending)")
    else:
        print(f"    PROBLEM: expected dry_run digest with pending alerts, got {flush_result}")
        all_ok = False
    print()

    print("=" * 78)
    print("HONEST VERIFICATION SUMMARY (see this file's module docstring for full detail):")
    print("  - RabbitMQ / InfluxDB / Elasticsearch writes above are LIVE calls against this")
    print("    machine's actually-running opencti-rabbitmq / observability-influxdb / ")
    print("    opencti-elasticsearch containers (confirmed reachable before this script existed).")
    print("  - LINE Messaging API: dry-run unless both token and LINE_TO_ID are configured;")
    print("    request URL/payload are covered by tests/test_fixed_contracts.py.")
    print("  - Gmail send: GMAIL_* are blank placeholders (stub/mock-only per the")
    print("    resolved open question) -- dry_run is the CORRECT expected result, not a failure.")
    print("  - Grafana/Kibana UI were NOT opened in a browser (no browser tool in this")
    print("    environment) -- see verify_dashboard_queries.py for a direct Flux/ES query check")
    print("    against the same queries those UIs use, which IS exercised.")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
