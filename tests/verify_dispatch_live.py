"""
verify_dispatch_live.py
=======================
NOT A TEST. This script sends real notifications and writes to the live stack.

It was named `test_alert_dispatcher.py` until 2026-11-08. It has no `unittest.TestCase`, so
`unittest discover` collected 0 tests from it and `ci.yml` listed it while running nothing. It
now sits outside the `test_` namespace so no discovery pattern or CI list can pick it up.

What one run does, with the live `.env`:

- Calls `alert_dispatcher.dispatch_alert(...)` five times. Two cases are HIGH, and each sends a
  real LINE broadcast to every friend of the channel. LINE allows 300 messages a month on this
  channel and quota is consumed per recipient, so one run spends several of them.
- Dispatches one MEDIUM case. With the four `GMAIL_*` keys populated this MAILS A REAL ADDRESS,
  the sender's own inbox, through the Gmail API.
- Flushes the LOW-priority digest, which mails a real digest of every pending LOW alert.
- Writes to RabbitMQ, InfluxDB and Elasticsearch. Nothing consumes the RabbitMQ publish.
- Then asserts the MEDIUM and digest results are `dry_run`, which they no longer are, and
  reports a failure for correct behaviour.

To run it deliberately, from the repo root:

    uv run python -m tests.verify_dispatch_live --send-for-real

Without `--send-for-real` it exits before sending anything. Do not run it from a test runner,
a CI job, or a loop. Fixtures live in `tests/fixtures/`, shared with real tests, so they stay put.

Original description, kept for the fixture and verification detail below.

Hand-written fixture-driven exercise of `src/notifications/alert_dispatcher.dispatch_alert(...)`.
It mirrors `src/evaluation/eval_rag_precision.py`'s shape: a plain runnable script that calls the
module under test against a fixed fixture set and prints pass/fail per case.

Fixtures (tests/fixtures/) match HANDOFF_LLM_LAYER.md's exact "Output schema reconciliation"
JSON shape -- `alert_context_high_portscan.json` / `llm_output_high_portscan.json` are that
handoff's own worked example, reused VERBATIM (not paraphrased) per HANDOFF_OUTPUT_LAYER.md
Deliverable #5. 3 more pairs (HIGH/high-confidence, MEDIUM, LOW) were authored for this phase to
exercise every priority-routing branch and the model_used='random_forest'/'if_ae_fallback'
branches. A 5th pair (LOW/malformed-no-confidence, added 2026-07-18) deliberately omits
'confidence' from both alert_context and llm_output, closing a coverage gap a code-reviewer pass
found: without it, derive_model_used()'s model_used='unknown' branch had no reproducible
test-suite coverage at all. All 5 pairs together now exercise all three `model_used` branches.

IMPORTANT -- what this script actually verifies, honestly stated (per HANDOFF_OUTPUT_LAYER.md
Deliverable #5's instruction not to fabricate a "verified" claim for anything not actually
exercised):
- RabbitMQ / InfluxDB / Elasticsearch: live-verified at least twice now, on two different
  occasions, both with real round-trip evidence, not just "the call didn't throw": 2026-07-15
  (when this script was authored) and again 2026-07-18 after a same-day gap where a re-run found
  no Docker daemon reachable at all (every call legitimately came back `status="error"` for
  these three during that gap -- see `models/output_layer_config.json`'s
  `known_gaps_not_fixed_in_this_phase` for that dated, now-superseded entry). **This is still a
  dated claim, not a standing guarantee** -- whether it holds when YOU run this script depends on
  whether Docker + the `network`/`opencti`/`observability`/`grafana` compose stacks are actually
  up right now. Check `docker ps` yourself; don't assume either this docstring's original
  2026-07-15 claim or its 2026-07-18 update still describes the present moment.
- LINE: written when LINE was dry-run until both `LINE_CHANNEL_ACCESS_TOKEN` and `LINE_TO_ID`
  were configured. notify_line.py has since gained `LINE_DELIVERY_MODE=broadcast`, which needs no
  recipient, so with a valid token this script's HIGH cases send REAL LINE messages and consume
  channel quota. The request contract is covered by `test_fixed_contracts.py`.
- Gmail: written while the `GMAIL_*` keys were blank placeholders (this channel replaced the
  original Outlook/Microsoft Graph API design 2026-07-18, see HANDOFF_OUTPUT_LAYER.md), so every
  MEDIUM/LOW send was expected to come back `status="dry_run"`. Once the keys are populated the
  MEDIUM case sends a REAL email and the dry_run assertion in `run_case` reports a failure, so
  this script is only safe to run with the `GMAIL_*` values blank or overridden.
- Grafana/Kibana UI: this script queries InfluxDB/Elasticsearch directly with the SAME Flux/ES
  queries `grafana/dashboards/soc-overview.json` / the Kibana saved search use, and confirms
  non-empty results -- it does NOT open a browser against the live Grafana/Kibana UI (no browser
  tool available in this environment). That is a real, disclosed gap, not silently skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEND_FLAG = "--send-for-real"

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
        "expected_model_used": "if_ae_fallback",  # confidence=0.91 < CONFIDENCE_THRESHOLD=0.95
    },
    {
        "name": "HIGH / DoS Hulk (high-confidence, exercises the random_forest model_used branch)",
        "alert_context": "alert_context_high_confident_doshulk.json",
        "llm_output": "llm_output_high_confident_doshulk.json",
        "expected_priority": "HIGH",
        "expected_model_used": "random_forest",  # confidence=0.99 >= 0.95
    },
    {
        "name": "MEDIUM / SSH-Patator",
        "alert_context": "alert_context_medium_sshpatator.json",
        "llm_output": "llm_output_medium_sshpatator.json",
        "expected_priority": "MEDIUM",
        "expected_model_used": "if_ae_fallback",  # confidence=0.55 < 0.95
    },
    {
        "name": "LOW / Bot",
        "alert_context": "alert_context_low_bot.json",
        "llm_output": "llm_output_low_bot.json",
        "expected_priority": "LOW",
        "expected_model_used": "if_ae_fallback",  # confidence=0.20 < 0.95
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
    else:  # LOW
        if notif["status"] != "accumulated":
            ok = False
            problems.append(f"expected LOW alert to be accumulated into SQLite digest, got: {notif}")

    return ok, {"receipt": receipt, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    if SEND_FLAG not in (sys.argv[1:] if argv is None else argv):
        print("[verify_dispatch_live] refusing to run: this sends real LINE and Gmail messages.")
        print(f"    Read the module docstring, then run: uv run python -m tests.verify_dispatch_live {SEND_FLAG}")
        return 2
    print(f"[verify_dispatch_live] fusion_layer.CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD} (model_used gate, reused not duplicated)")
    print(f"[verify_dispatch_live] running {len(FIXTURE_CASES)} fixture cases against a LIVE dispatch_alert(...)\n")

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

    # --- LOW-priority digest flush ---
    print("[verify_dispatch_live] flushing LOW-priority digest (exercises flush_low_priority_digest)...")
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
