"""`llm-speed submit` — upload runs that were saved locally.

With paths, each file goes through `bench --resume`'s upload (`_upload_saved_run`):
the saved JWS is verified and posted as-is. With no path, unsent runs in the
cache are listed and offered through the existing consent prompt.

Which runs were already uploaded is a local sidecar (`<file>.json.uploaded`).
Nothing is added to the saved JSON or to the upload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.text import Text

from .. import __version__, consent, upload
from ..config import DEFAULT_API_BASE
from ..credit import CreditError, check_flags, maybe_ask
from ..output import get_console, print_error, print_warning
from ..signing import verify_jws_token
from .bench import _upload_saved_run

# Batch decline is not a missing consent. Only the consent gate prints "(no consent)".
_ANSWERED_NO = "Not uploaded: you answered no."
_NO_TERMINAL = (
    "Not uploaded: no terminal to confirm; pass --yes to upload the listed runs."
)
_NO_CONSENT = "Skipping upload (no consent)."
_EMPTY_NEXT = "Run `llm-speed bench` to create one."


def cmd_submit(args) -> int:
    try:
        check_flags(args)
    except CreditError as exc:
        print_error(str(exc))
        return 2

    paths = [Path(p) for p in (getattr(args, "paths", None) or [])]
    if paths:
        return _submit_paths(paths, args)
    return _submit_unsent(args)


def _submit_paths(paths: list[Path], args) -> int:
    """Named files. Same upload as `bench --resume`.

    Without consent.json, and without `--strict-anon`, the existing consent
    prompt runs first. A non-TTY answer is no. `--yes` does not grant consent.
    """
    if not getattr(args, "strict_anon", False) and not consent.has_consented():
        if not _consent_ok(args, {}):
            get_console().print(Text(_NO_CONSENT), soft_wrap=True)
            return 1
    rc = 0
    for path in paths:
        one = _upload_saved_run(path, args, verb="submit")
        if one != 0:
            rc = one
            if one == 2:
                break
    return rc


def _submit_unsent(args) -> int:
    """List cache runs that have no local upload marker, then the consent gate."""
    console = get_console()
    runs_dir = upload.RUNS_DIR
    if not runs_dir.exists():
        _print_empty(console, f"No saved runs in {runs_dir}.")
        return 0
    if not runs_dir.is_dir():
        print_error(f"not a directory: {runs_dir}")
        return 1
    try:
        names = sorted(p for p in runs_dir.iterdir() if p.suffix == ".json")
    except OSError as exc:
        print_error(f"cannot read {runs_dir}: {exc}")
        return 1

    pending: list[tuple[Path, dict, str, str, str]] = []
    already = 0
    for path in names:
        if path.is_symlink() or not path.is_file():
            print_warning(f"skipping {path.name}: not a regular file")
            continue
        if upload.run_already_uploaded(path):
            already += 1
            continue
        payload, token, reason = _read_run(path)
        if payload is None:
            print_warning(f"skipping {path.name}: {reason}")
            continue
        if payload.get("agent_job") is not None:
            print_warning(f"skipping {path.name}: saved run carries an agent job field")
            continue
        if getattr(args, "strict_anon", False) and upload.saved_run_is_identified(
            payload, token
        ):
            print_warning(
                f"skipping {path.name}: saved without --strict-anon; "
                "not posted under --strict-anon"
            )
            continue
        pending.append((path, payload, *_describe(payload)))

    if not pending:
        if already:
            _print_empty(console, f"No unsent runs in {runs_dir}.")
        else:
            _print_empty(console, f"No saved runs in {runs_dir}.")
        return 0

    if already:
        console.print(Text(f"Already submitted, not listed: {already}"))
    console.print(Text(f"Unsent runs in {runs_dir}:"), soft_wrap=True)
    for path, _payload, ts, model, hardware in pending:
        console.print(
            Text.assemble("  ", ts, "  ", model, "  ", hardware), soft_wrap=True
        )
        console.print(Text(f"    {path}"), soft_wrap=True)

    if not _consent_ok(args, pending[0][1]):
        console.print(Text(_NO_CONSENT), soft_wrap=True)
        return 1
    # Consent already held. A batch decline is not a missing consent.
    answer = _confirm_batch(args, len(pending))
    if answer == "no":
        console.print(Text(_ANSWERED_NO), soft_wrap=True)
        return 1
    if answer == "no-tty":
        console.print(Text(_NO_TERMINAL), soft_wrap=True)
        return 1

    # Same one-time credit question bench asks after consent. Never reads stdin
    # unless a human is at a TTY and it has not been asked before. Storage info
    # is not asked: the saved JWS is posted unchanged, so a new block cannot
    # be attached.
    maybe_ask(args)

    n = len(pending)
    console.print(f"Submitting {n} unsent run{'' if n == 1 else 's'}.")
    rc = 0
    for path, *_rest in pending:
        one = _upload_saved_run(path, args, verb="submit")
        if one != 0:
            rc = one
            if one == 2:
                break
    return rc


def _print_empty(console, line: str) -> None:
    """Nothing to upload. The next sentence tells the user how to make a run."""
    console.print(Text(line), soft_wrap=True)
    console.print(Text(_EMPTY_NEXT), soft_wrap=True)


def _confirm_batch(args, n: int) -> str:
    """'yes', 'no', or 'no-tty' for the listed runs. Default no.

    `--yes` skips only this question. It does not grant consent. A non-TTY
    without `--yes` is 'no-tty', including under `--strict-anon`: the question
    is not printed. A shown question that is not answered yes is 'no'.
    """
    if getattr(args, "yes", False):
        return "yes"
    if not sys.stdin.isatty():
        return "no-tty"
    word = "run" if n == 1 else "runs"
    sys.stdout.write(f"Upload the {n} {word} listed above? [y/N] ")
    sys.stdout.flush()
    try:
        answer = sys.stdin.readline()
    except KeyboardInterrupt:
        return "no"
    return "yes" if answer.strip().lower() in ("y", "yes") else "no"


def _consent_ok(args, payload: dict) -> bool:
    """Bench's consent gate. `--strict-anon` is consent; otherwise the prompt.

    The prompt's wording and rules are `cli.consent.prompt_for_consent`: empty
    input on a TTY is yes, and a non-TTY is no unless consent was already saved.
    """
    if getattr(args, "strict_anon", False) or consent.has_consented():
        return True
    fp = payload.get("fingerprint")
    fp_hash = ""
    if isinstance(fp, dict) and isinstance(fp.get("fingerprint_hash"), str):
        fp_hash = fp["fingerprint_hash"]
    api_base = getattr(args, "api_base", None) or DEFAULT_API_BASE
    return consent.prompt_for_consent(
        api_base=api_base,
        cli_version=__version__,
        fingerprint_hash=fp_hash,
    )


def _read_run(path: Path) -> tuple[dict | None, str | None, str]:
    """(payload, token, "") or (None, None, reason to skip). Does not upload."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, None, "corrupt JSON (not utf-8)"
    except OSError as exc:
        return None, None, f"unreadable ({exc})"
    try:
        saved = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, None, f"corrupt JSON ({exc.msg})"
    token = saved.get("jws") if isinstance(saved, dict) else None
    if not isinstance(token, str) or not token:
        return None, None, "not an llm-speed run (no jws)"
    ok, reason, payload = verify_jws_token(token)
    if not ok or not isinstance(payload, dict):
        return None, None, f"not a verifiable run ({reason})"
    return payload, token, ""


def _describe(payload: dict) -> tuple[str, str, str]:
    """Timestamp, model, hardware for the unsent-run list. Display only."""
    started = payload.get("started_at")
    if isinstance(started, str) and started.strip():
        ts = started.strip()
    else:
        ts = "(unknown time)"
    model = "(unknown model)"
    results = payload.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            raw = result.get("model")
            if not isinstance(raw, dict):
                continue
            name = raw.get("name") or raw.get("identifier")
            if isinstance(name, str) and name.strip():
                model = name.strip()
                break
    hardware = "(unknown hardware)"
    fp = payload.get("fingerprint")
    if isinstance(fp, dict):
        summary = fp.get("accelerator_summary")
        if isinstance(summary, str) and summary.strip():
            hardware = summary.strip()
    return ts, model, hardware
