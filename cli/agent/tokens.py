"""Per-machine token storage.

macOS: the login Keychain (via `security -i` on stdin, so the token never appears on a
command line). If the Keychain can't be used (typically a locked login Keychain in a
session that can't show the unlock prompt: ssh, a remote-exec helper, a launchd job
before anyone logged in; `security` then fails with errSecInteractionNotAllowed,
-25308, "User interaction is not allowed"), enrollment falls back to the file store in
the same process, so a token the server delivers exactly once is never dropped.
Elsewhere (and as that fallback): a 0600 file in the 0700 agent config dir, readable by
the launchd/systemd job without an unlocked Keychain.

Which store holds the token is recorded in agent/config.json as `token_store`
("keychain" or "file"); the token itself never is. The token is never logged, printed,
put in a child's env, or sent in a report."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .settings import agent_dir

SERVICE = "com.llm-speed.agent"
TOKEN_RE = re.compile(r"lsa_[A-Za-z0-9_\-]{8,}")
STORES = ("keychain", "file")
KEYCHAIN_DESC = f"macOS login Keychain (service {SERVICE})"

log = logging.getLogger("cli.agent.tokens")
# Enrolment tokens (R-8) and per-job upload nonces (R-9) are redacted the same way.
_OTHER_SECRETS_RE = re.compile(r"\b(lse|lsn)_[A-Za-z0-9_\-]{8,}")


def redact(text: str) -> str:
    return _OTHER_SECRETS_RE.sub(
        lambda m: f"{m.group(1)}_[redacted]", TOKEN_RE.sub("lsa_[redacted]", text)
    )


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if TOKEN_RE.search(msg):
            record.msg, record.args = redact(msg), ()
        return True


class TokenStoreError(RuntimeError):
    """A store could not keep the token. The message never contains the token."""


@dataclass(frozen=True)
class Stored:
    kind: str  # "keychain" | "file"
    where: str  # human-readable location, no secret
    fallback_reason: str | None = None  # why the Keychain was skipped, if it was


def _keychain_available() -> bool:
    return (
        sys.platform == "darwin"
        or os.environ.get("LLM_SPEED_AGENT_TOKEN_STORE") == "keychain"
    )


def _backend(prefer: str | None = None) -> str:
    if prefer in STORES:
        return prefer
    forced = os.environ.get("LLM_SPEED_AGENT_TOKEN_STORE")
    if forced in STORES:
        return forced
    return "keychain" if sys.platform == "darwin" else "file"


def token_file() -> Path:
    return agent_dir() / "token"


def _q(s: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", s):
        raise ValueError("unexpected characters in keychain field")
    return s


_OSSTATUS_RE = re.compile(r"-?\d{3,6}")


def _keychain_reason(rc: int, stderr: str) -> str:
    """A short, token-free reason from `security`'s stderr."""
    err = redact(stderr or "").strip().splitlines()
    text = err[-1][:200] if err else ""
    low = text.lower()
    if "interaction is not allowed" in low or "-25308" in low:
        return (
            "the login Keychain is locked and this session can't show the unlock prompt "
            "(errSecInteractionNotAllowed, -25308)"
        )
    if "no default keychain" in low or "-25307" in low:
        return "this session has no default (login) Keychain (errSecNoDefaultKeychain, -25307)"
    if "-128" in low or "canceled" in low or "cancelled" in low:
        return "the Keychain prompt was cancelled (userCanceledErr, -128)"
    if "-25293" in low or "authorization" in low or "authentication" in low:
        return "the Keychain refused the request (auth failed)"
    return f"`security` exited {rc}" + (f": {text}" if text else "")


def _keychain_save(machine_id: str, token: str) -> None:
    cmd = f"add-generic-password -U -s {SERVICE} -a {_q(machine_id)} -w {_q(token)}\n"
    try:
        r = subprocess.run(
            ["security", "-i"],
            input=cmd,
            text=True,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise TokenStoreError(
            "`security` timed out (a Keychain prompt nobody can answer?)"
        ) from None
    except OSError as e:
        raise TokenStoreError(f"could not run `security`: {e.strerror or e}") from None
    # `security -i` can report a failed command on stderr; check both.
    if (
        r.returncode != 0
        or "error" in (r.stderr or "").lower()
        or ("not allowed" in (r.stderr or "").lower())
    ):
        raise TokenStoreError(_keychain_reason(r.returncode, r.stderr))
    if _keychain_load(machine_id) != token:
        raise TokenStoreError(
            "the token could not be read back from the login Keychain"
        )


def _keychain_load(machine_id: str) -> str | None:
    try:
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
    except (OSError, subprocess.TimeoutExpired):
        return None
    tok = r.stdout.strip() if r.returncode == 0 else ""
    return tok or None


def _file_save(token: str) -> str:
    p = token_file()
    try:
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(p.parent, 0o700)
        tmp = p.with_name("token.tmp")
        # A leftover token.tmp could carry looser permissions (O_CREAT keeps an existing
        # file's mode); remove it and create a fresh 0600 file exclusively.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        os.chmod(p, 0o600)
    except OSError as e:
        raise TokenStoreError(f"could not write {p}: {e.strerror or e}") from None
    return f"{p} (0600)"


def _file_load() -> str | None:
    p = token_file()
    if not p.exists():
        return None
    if p.stat().st_mode & 0o077:
        os.chmod(p, 0o600)
    return p.read_text(encoding="utf-8").strip() or None


def save_token(machine_id: str, token: str, prefer: str | None = None) -> Stored:
    """Store the token; never lose it to a Keychain failure.

    `prefer` = "keychain" / "file" / None (auto: Keychain on macOS). When the Keychain
    is tried and fails, the token (still only in this process's memory) goes to the
    file store instead and `fallback_reason` says why. Raises TokenStoreError only when
    no store could keep it."""
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("server returned a token in an unexpected format")
    _q(machine_id)
    reason = None
    if _backend(prefer) == "keychain":
        try:
            _keychain_save(machine_id, token)
            try:  # a stale file from an earlier enrollment must not shadow it
                token_file().unlink()
            except OSError:
                pass
            return Stored("keychain", KEYCHAIN_DESC)
        except TokenStoreError as e:
            reason = str(e)
            log.warning("login Keychain not usable (%s); using the file store", reason)
    try:
        where = _file_save(token)
    except TokenStoreError as e:
        raise TokenStoreError(
            (f"login Keychain: {reason}; file: {e}") if reason else str(e)
        ) from None
    return Stored("file", where, reason)


def _recorded_store() -> str | None:
    try:
        from .settings import Settings

        v = Settings.load().token_store
    except (OSError, ValueError, TypeError):  # unreadable config: platform default
        return None
    return v if v in STORES else None


def load_token(machine_id: str | None, store: str | None = None) -> str | None:
    """Read from the store enrollment recorded (agent/config.json `token_store`), then
    the other one, so a token kept by the fallback is found by the runner too."""
    if not machine_id:
        return None
    first = store if store in STORES else (_recorded_store() or _backend())
    order = [first] + [k for k in STORES if k != first]
    for kind in order:
        if kind == "keychain":
            if not _keychain_available():
                continue
            tok = _keychain_load(machine_id)
        else:
            tok = _file_load()
        if tok:
            return tok
    return None


def describe_store(store: str | None) -> str:
    """Where the token lives, for `agent status` / `doctor` (never the token)."""
    if store == "keychain":
        return KEYCHAIN_DESC
    if store == "file":
        return f"file {token_file()} (0600)"
    return (
        "not recorded (enrolled before 0.0.10: "
        + (KEYCHAIN_DESC if _backend() == "keychain" else f"file {token_file()}")
        + ")"
    )


def delete_token(machine_id: str | None) -> None:
    if machine_id and _keychain_available():
        try:
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
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        token_file().unlink()
    except FileNotFoundError:
        pass
