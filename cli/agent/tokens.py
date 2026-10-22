"""Per-machine token storage. macOS: login Keychain (via `security -i` on stdin, so
the token never appears on a command line). Elsewhere: a 0600 file in a 0700 dir.
The token is never logged, printed, put in a child's env, or sent in a report."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .settings import agent_dir

SERVICE = "com.llm-speed.agent"
TOKEN_RE = re.compile(r"lsa_[A-Za-z0-9_\-]{8,}")


def redact(text: str) -> str:
    return TOKEN_RE.sub("lsa_[redacted]", text)


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if TOKEN_RE.search(msg):
            record.msg, record.args = redact(msg), ()
        return True


def _backend() -> str:
    forced = os.environ.get("LLM_SPEED_AGENT_TOKEN_STORE")
    if forced in ("file", "keychain"):
        return forced
    return "keychain" if sys.platform == "darwin" else "file"


def token_file() -> Path:
    return agent_dir() / "token"


def _q(s: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", s):
        raise ValueError("unexpected characters in keychain field")
    return s


def save_token(machine_id: str, token: str) -> str:
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("server returned a token in an unexpected format")
    if _backend() == "keychain":
        cmd = (
            f"add-generic-password -U -s {SERVICE} -a {_q(machine_id)} -w {_q(token)}\n"
        )
        r = subprocess.run(
            ["security", "-i"],
            input=cmd,
            text=True,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError("could not store the token in the login Keychain")
        return "macOS Keychain"
    p = token_file()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(p.parent, 0o700)
    tmp = p.with_name("token.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token)
    os.replace(tmp, p)
    os.chmod(p, 0o600)
    return f"{p} (0600)"


def load_token(machine_id: str | None) -> str | None:
    if not machine_id:
        return None
    if _backend() == "keychain":
        r = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                SERVICE,
                "-a",
                _q(machine_id),
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        tok = r.stdout.strip() if r.returncode == 0 else ""
        return tok or None
    p = token_file()
    if not p.exists():
        return None
    if p.stat().st_mode & 0o077:
        os.chmod(p, 0o600)
    return p.read_text(encoding="utf-8").strip() or None


def delete_token(machine_id: str | None) -> None:
    if _backend() == "keychain":
        if machine_id:
            subprocess.run(
                [
                    "security",
                    "delete-generic-password",
                    "-s",
                    SERVICE,
                    "-a",
                    _q(machine_id),
                ],
                check=False,
                capture_output=True,
                timeout=30,
            )
        return
    try:
        token_file().unlink()
    except FileNotFoundError:
        pass
