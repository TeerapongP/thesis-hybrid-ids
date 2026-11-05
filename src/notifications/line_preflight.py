# -*- coding: utf-8 -*-
"""Check the LINE Messaging API channel before relying on it, without sending anything.

send_line_alert() reports one undifferentiated dry-run for "LINE_CHANNEL_ACCESS_TOKEN or
LINE_TO_ID is not configured", which is the right behaviour for a dispatcher -- a
notification path must not stall the pipeline arguing about its own credentials -- but it
makes a misconfiguration hard to tell apart from a revoked token or an unreachable
network. This separates those causes using read-only endpoints, so a failing channel can
be diagnosed without pushing a message to anyone.

    uv run python -m src.notifications.line_preflight
    uv run python -m src.notifications.line_preflight --send-test   # actually pushes one message

--send-test is opt-in because a push is visible to a real person and counts against the
channel's monthly quota, which /v2/bot/message/quota reports.

One trap this checks for explicitly: /v2/bot/info returns the BOT's own userId, and the
LINE Developers console shows it on the same screens as the channel's other identifiers.
Pushing to it fails, so LINE_TO_ID matching it is reported as a configuration error
rather than passed through to a confusing API response.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

from .config_utils import looks_real

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "notification" / ".env"
API = "https://api.line.me"


def read_env(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"not found: {path}\nCopy notification/.env.example to notification/.env first.")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", newline="").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def api_get(path: str, token: str) -> tuple[int | None, dict]:
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {}
    except Exception as exc:  # DNS failure, TLS failure, timeout
        return None, {"transport_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send-test", action="store_true",
                    help="push one real test message to LINE_TO_ID (visible to a person, uses quota)")
    args = ap.parse_args()

    env = read_env(ENV_PATH)
    token = env.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    recipient = env.get("LINE_TO_ID", "")
    problems: list[str] = []

    # Mirror send_line_alert's own thresholds, so a pass here means that function will
    # attempt the call rather than drop to dry-run. Diverging thresholds would make this
    # check pass while the dispatcher still refused.
    if not looks_real(token, min_len=20):
        problems.append("LINE_CHANNEL_ACCESS_TOKEN is empty or placeholder")
        print("FAIL  token      not configured")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"ok    token      configured ({len(token)} chars; value not printed)")

    status, info = api_get("/v2/bot/info", token)
    if status is None:
        print(f"FAIL  reachable  {info.get('transport_error')}")
        return 1
    if status != 200:
        print(f"FAIL  token      /v2/bot/info -> HTTP {status} {info.get('message', '')}")
        print("      401 means the token is revoked or regenerated; reissue it in the console.")
        return 1
    bot_id = info.get("userId", "")
    print(f"ok    channel    \"{info.get('displayName')}\" {info.get('basicId')} "
          f"chatMode={info.get('chatMode')}")

    status, quota = api_get("/v2/bot/message/quota", token)
    if status == 200:
        kind, value = quota.get("type"), quota.get("value")
        print(f"ok    quota      type={kind}" + (f" value={value}" if value is not None else ""))
        status2, used = api_get("/v2/bot/message/quota/consumption", token)
        if status2 == 200 and value is not None:
            print(f"ok    used       {used.get('totalUsage')} of {value} this month")

    if not looks_real(recipient, min_len=8):
        problems.append(
            "LINE_TO_ID is empty. It is the push destination and cannot be derived from the "
            "token: /v2/bot/followers/ids needs a verified or premium Official Account and "
            "returns HTTP 403 otherwise, and a group or room id only arrives on a webhook "
            "event, which this project does not run. Copy \"Your user ID\" from the LINE "
            "Developers console's Basic settings tab (33 characters, starts with U), and add "
            "the bot as a friend from that account first -- a push to a non-friend is rejected."
        )
        print("FAIL  recipient  LINE_TO_ID not configured")
    elif recipient == bot_id:
        problems.append("LINE_TO_ID is the bot's OWN userId from /v2/bot/info, not a destination.")
        print("FAIL  recipient  LINE_TO_ID equals the bot's own userId")
    else:
        print(f"ok    recipient  configured ({len(recipient)} chars)")

    if problems:
        print()
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    print("\nPASS  channel is usable; send_line_alert will attempt a real push.")

    if args.send_test:
        from .notify_line import send_line_alert
        record = {"priority": "HIGH", "source_ip": "198.51.100.23",
                  "mitre_tactic": "Credential Access",
                  "explanation_th": "ข้อความทดสอบจาก line_preflight ไม่ใช่การแจ้งเตือนจริง"}
        config = {"line_channel_access_token": token, "line_to_id": recipient,
                  "line_grafana_base_url": env.get("LINE_GRAFANA_BASE_URL", "http://localhost:3000")}
        result = send_line_alert(record, config)
        print(f"\nsend-test: {json.dumps({k: v for k, v in result.items() if k != 'would_send_message'}, ensure_ascii=False)}")
        return 0 if result.get("status") == "sent" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
