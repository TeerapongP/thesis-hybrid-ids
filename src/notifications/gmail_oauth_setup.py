# -*- coding: utf-8 -*-
"""Mint the Gmail refresh token that notify_gmail.py needs, and store it.

Why this exists as a script rather than a documented sequence of clicks: the refresh
token is the one Gmail credential that cannot be copied out of the Google Cloud console.
It is only ever returned once, by an OAuth2 consent exchange, so obtaining it requires
running code on a machine that has both a browser and outbound internet.

The token is written straight into notification/.env rather than printed. A refresh
token grants send-as-you access until revoked, and anything printed to a terminal ends
up in scrollback, shell history and any transcript of the session. The script reports
only the length it wrote.

Run from the repo root on a machine with a browser:

    uv run python -m src.notifications.gmail_oauth_setup --write

Prerequisites, all from the Google Cloud console (see notification/.env.example):
the Gmail API enabled, your own address added as a test user on the OAuth consent
screen, and a Desktop-app OAuth client whose id and secret are already in
notification/.env as GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "notification" / ".env"

# Only the send scope. This project sends mail and never reads a mailbox, so a broader
# scope would be privilege the code has no use for.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

TARGET_KEY = "GMAIL_REFRESH_TOKEN"


def read_env(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"not found: {path}\nCopy notification/.env.example to notification/.env first.")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def write_key(path: pathlib.Path, key: str, value: str) -> None:
    """Replace one key in place, preserving every other line and the file's line endings.

    Rewriting the whole file from a parsed dict would silently drop the comments that
    explain where each credential comes from, so the substitution is line-scoped.
    """
    # newline="" on both read and write: the default translates CRLF to LF on the way in
    # and LF back to CRLF on the way out, so line endings can only be detected and
    # reproduced faithfully with translation off at both ends.
    raw = path.read_text(encoding="utf-8", newline="")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    # newline="" disables Windows' default LF -> CRLF translation on write, which would
    # otherwise rewrite an LF file as CRLF and show the whole file as modified.
    path.write_text(newline.join(lines) + newline, encoding="utf-8", newline="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help=f"write {TARGET_KEY} into notification/.env (otherwise this is a dry run)")
    ap.add_argument("--port", type=int, default=0,
                    help="local redirect port; 0 lets the OS pick a free one")
    args = ap.parse_args()

    env = read_env(ENV_PATH)
    client_id = env.get("GMAIL_CLIENT_ID", "")
    client_secret = env.get("GMAIL_CLIENT_SECRET", "")
    missing = [k for k, v in (("GMAIL_CLIENT_ID", client_id),
                              ("GMAIL_CLIENT_SECRET", client_secret)) if len(v) < 10]
    if missing:
        sys.exit("not set in notification/.env: " + ", ".join(missing)
                 + "\nCreate a Desktop-app OAuth client first; see notification/.env.example.")

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ModuleNotFoundError:
        sys.exit("google-auth-oauthlib is not installed in this interpreter.\n"
                 "Run `uv sync` from the repo root, then retry with `uv run`.")

    # The console's Desktop-app client is normally downloaded as a client_secrets.json.
    # Rebuilding that structure from the two values already in .env keeps one source of
    # truth and avoids a second credential file sitting in the working tree.
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    # prompt="consent" is load-bearing: Google omits refresh_token on a repeat
    # authorization unless consent is forced, which makes a re-run look successful
    # while returning nothing to store.
    creds = flow.run_local_server(port=args.port, prompt="consent",
                                  access_type="offline", include_granted_scopes="false")

    token = creds.refresh_token
    if not token:
        sys.exit("Google returned no refresh token. Revoke this app's access at "
                 "https://myaccount.google.com/permissions and run again.")

    print(f"refresh token obtained ({len(token)} chars; value not printed)")
    if not args.write:
        print("dry run -- nothing written. Re-run with --write to store it.")
        return 0

    write_key(ENV_PATH, TARGET_KEY, token)
    print(f"wrote {TARGET_KEY} into {ENV_PATH}")
    if len(env.get("GMAIL_SENDER_ADDRESS", "")) < 3:
        print("still empty: GMAIL_SENDER_ADDRESS -- set it to the account you just "
              "consented as, or Gmail will reject the From: address.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
