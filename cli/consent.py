"""First-run consent for upload.

Stored as `~/.config/llm-speed/consent.json` (mode 0600). Contains the user's
explicit yes plus enough provenance (timestamp, CLI version, the trimmed
fingerprint hash) to prove "you said yes from this machine on this date" if
ever disputed.

Strict-anon mode never calls into here — choosing the most-private option IS
explicit consent and there's no persistent identity to attach to a consent file.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

from .config import CONFIG_DIR

log = logging.getLogger("cli.consent")


def consent_path() -> Path:
    return CONFIG_DIR / "consent.json"


def has_consented() -> bool:
    p = consent_path()
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("consent file unreadable (%s); treating as no consent", exc)
        return False
    return bool(data.get("consented"))


def _write_consent(record: dict) -> None:
    p = consent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def prompt_for_consent(
    *,
    api_base: str,
    cli_version: str,
    fingerprint_hash: str | None = None,
    stream=None,
) -> bool:
    """Show the consent text and read a Y/N answer. Persist on yes.

    Hard rule: this is a *prompt*. If stdin isn't a TTY (piped, EOF'd, in CI,
    backgrounded), we DO NOT auto-consent — we treat absence of an explicit
    interactive yes as no. The user can opt in non-interactively by either
    running `llm-speed bench --strict-anon` (the most-private mode, which
    implies consent), pre-creating `~/.config/llm-speed/consent.json`, or
    explicitly piping `y\\n` into stdin.

    Empty input *from a TTY* still defaults to yes — that matches the
    `[Y/n]` convention. The TTY check is what blocks the silent CI bypass.
    """
    out = stream or sys.stdout

    text = (
        "\n"
        "This will collect:\n"
        "  - Hardware (GPU model, CPU, RAM bucket - no serial numbers, no driver builds)\n"
        "  - OS major version and backend versions\n"
        "  - Benchmark numbers (no prompts, no model output)\n"
        f"And submit them to {api_base}.\n"
        "\n"
        "Run llm-speed bench --no-upload to keep results local only.\n"
        "Run llm-speed bench --strict-anon to submit without persistent identity.\n"
        "Run llm-speed bench --print-payload to see exactly what would be sent.\n"
        "\n"
    )

    # Refuse to prompt when there's no interactive user. Sending the question
    # to a non-tty + treating empty/EOF as yes silently uploads on CI / when
    # piped from /dev/null / when backgrounded — that's a policy bypass.
    if not sys.stdin.isatty():
        out.write(text)
        out.write(
            "Stdin is not a TTY; refusing to auto-consent. Re-run interactively "
            "or use --strict-anon / --no-upload / pre-create consent.json.\n"
        )
        out.flush()
        return False

    out.write(text)
    out.write("Continue? [Y/n] ")
    out.flush()

    try:
        answer = sys.stdin.readline()
    except KeyboardInterrupt:
        out.write("\n")
        return False

    # readline() returns '' on EOF (closed stdin); the TTY check above prevents
    # piped EOF from reaching this path. A real user who hits Ctrl-D after the
    # prompt sends '' too — treat as no.
    if not answer:
        out.write("\n(no input — not consenting)\n")
        return False

    answer = answer.strip().lower()
    if answer in ("", "y", "yes"):
        _write_consent(
            {
                "consented": True,
                "consented_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cli_version": cli_version,
                "api_base": api_base,
                "fingerprint_hash": fingerprint_hash or "",
            }
        )
        return True
    return False


def revoke_consent() -> bool:
    """Delete the consent file. Returns True if a file was removed."""
    p = consent_path()
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except OSError as exc:
        log.warning("failed to remove consent file %s: %s", p, exc)
        return False
