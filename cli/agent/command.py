"""`llm-speed agent ...` subcommands."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import sys
import time

from .. import __version__
from ..config import DEFAULT_API_BASE
from . import service
from .runner import Runner, approvals_dir, capabilities, read_status
from .settings import (
    SETTABLE,
    Settings,
    atomic_write_json,
    ensure_dirs,
    logs_dir,
    pause_path,
)
from .tokens import RedactFilter, delete_token, load_token, save_token
from .transport import QueueClient, QueueError

CONSENT_TEXT = """\
This machine will run benchmark presets from the llm-speed queue automatically
(only presets shipped inside this installed CLI, never commands) and upload the
results, without asking each time. Uploads are exactly what `llm-speed bench
--strict-anon` uploads (see https://llm-speed.com/privacy): bucketed hardware
(GPU name + memory bucket, CPU, RAM bucket, OS major), backend + version, model
file name and quant, benchmark numbers and fit results (incl. out-of-memory
failures, peak memory, llama.cpp build, driver major), CLI version, timestamps,
and a one-off signature. Never: hostnames, usernames, file paths, serials,
prompts, model output, logs or the machine token.
The queue additionally learns, per job: status, counts, the run ids uploaded,
and this machine's coarse hardware bucket, tied to this machine's enrollment.
Stop any time: `llm-speed agent pause` (local) or `llm-speed agent uninstall --forget`.
"""


def setup_logging() -> None:
    ensure_dirs()
    root = logging.getLogger("cli")
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return
    h = logging.handlers.RotatingFileHandler(
        logs_dir() / "agent.log", maxBytes=5_000_000, backupCount=5
    )
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    h.addFilter(RedactFilter())
    root.addHandler(h)
    root.setLevel(logging.INFO)


def _client(settings: Settings, args, *, auth: bool = True) -> QueueClient:
    """Enroll uses --api-base / $LLM_SPEED_API / the default. Afterwards the token goes
    only to the server that issued it (settings.api_base; unset = the default)."""
    asked = getattr(args, "api_base", None)
    if not auth:
        return QueueClient(asked)
    enrolled = (settings.api_base or DEFAULT_API_BASE).rstrip("/")
    if asked and asked.rstrip("/") != enrolled:
        raise QueueError(
            0,
            "api_base_mismatch",
            f"this machine is enrolled with {enrolled}; its token is only sent there "
            "(run `llm-speed agent enroll --force --api-base URL` to switch)",
        )
    return QueueClient(enrolled, load_token(settings.machine_id))


def _ask_consent() -> bool:
    print(CONSENT_TEXT)
    if not sys.stdin.isatty():
        print(
            "Stdin is not a TTY; refusing to auto-consent. Run `llm-speed agent enroll` in a terminal."
        )
        return False
    try:
        ans = input(
            "Allow this machine to run queued presets and upload results automatically? [y/N] "
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in ("y", "yes")


def cmd_enroll(args) -> int:
    ensure_dirs()
    s = Settings.load()
    if s.machine_id and load_token(s.machine_id) and not getattr(args, "force", False):
        print(f"Already enrolled as {s.machine_id}. Use --force to enroll again.")
        return 0
    if not _ask_consent():
        print("Not enrolled (no consent). Nothing was stored.")
        return 1
    try:
        client = _client(s, args, auth=False)
        p = client.pair_start(capabilities(s), getattr(args, "label", None))
    except QueueError as e:
        print(f"Could not start pairing: {e}")
        return 1
    code, pid, secret = p.get("code"), p.get("pairing_id"), p.get("poll_secret")
    if not (code and pid and secret):
        print("Unexpected pairing response from the server.")
        return 1
    print(f"\n  Pairing code:  {code}\n")
    print(
        "Send this code to your queue admin to approve this machine. Waiting (Ctrl-C to stop)..."
    )
    deadline = time.monotonic() + float(getattr(args, "timeout", 600))
    wait = float(p.get("poll_after_s") or 5)
    try:
        while time.monotonic() < deadline:
            time.sleep(min(max(wait, 1.0), 30.0))
            try:
                st, body = client.pair_poll(pid, secret)
            except QueueError as e:
                if e.status == 429 and e.retry_after:
                    wait = e.retry_after
                continue
            if st == "approved":
                mid, token = (
                    str(body.get("machine_id") or ""),
                    str(body.get("token") or ""),
                )
                if not mid or not token:
                    print("Approved, but the server sent no token.")
                    return 1
                where = save_token(mid, token)
                s.machine_id, s.label = mid, getattr(args, "label", None) or s.label
                s.api_base = client.api_base
                s.consent = {
                    "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "cli_version": __version__,
                    "text_sha256": hashlib.sha256(CONSENT_TEXT.encode()).hexdigest(),
                }
                s.save()
                print(f"Approved. Enrolled as {mid}; token stored in {where}.")
                return 0
            if st in ("expired", "denied", "used", "unknown"):
                why = {
                    "used": "already used (its token was delivered once)",
                    "unknown": "not found on the server",
                }.get(st, st)
                print(f"Pairing {why}. Run `llm-speed agent enroll` again.")
                return 1
            wait = float(body.get("poll_after_s") or wait)
    except KeyboardInterrupt:
        print("\nStopped waiting. Nothing was stored.")
        return 1
    print("Pairing timed out. Run `llm-speed agent enroll` again.")
    return 1


def cmd_install(args) -> int:
    s = Settings.load()
    if not getattr(args, "dry_run", False) and not (
        s.machine_id and load_token(s.machine_id)
    ):
        rc = cmd_enroll(args)
        if rc != 0:
            return rc
    rc, steps = service.install(dry_run=getattr(args, "dry_run", False))
    for line in steps:
        print(line)
    if getattr(args, "dry_run", False):
        print("(dry run: nothing was written or started)")
    elif rc == 0:
        print("Service installed and started.")
    else:
        print(f"Service install failed (exit {rc}).")
    return rc


def cmd_uninstall(args) -> int:
    rc, steps = service.uninstall(dry_run=getattr(args, "dry_run", False))
    for line in steps:
        print(line)
    if getattr(args, "forget", False) and not getattr(args, "dry_run", False):
        s = Settings.load()
        delete_token(s.machine_id)
        s.machine_id, s.consent = None, None
        s.save()
        print(
            "Token and agent consent removed. (Ask your queue admin to revoke the machine server-side too.)"
        )
    return rc


def cmd_run(args) -> int:
    setup_logging()
    s = Settings.load()
    if not s.consent:
        print(
            "This machine has no agent consent on record. Run `llm-speed agent enroll` first."
        )
        return 1
    if not (s.machine_id and load_token(s.machine_id)):
        print("Not enrolled. Run `llm-speed agent enroll`.")
        return 1
    try:
        client = _client(s, args)
    except QueueError as e:
        print(str(e))
        return 1
    return Runner(client, s).run_forever(once=getattr(args, "once", False))


def cmd_pause(args) -> int:
    ensure_dirs()
    atomic_write_json(
        pause_path(),
        {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reason": getattr(args, "reason", None),
            "now": bool(getattr(args, "now", False)),
        },
    )
    print(
        "Paused. No new jobs will start"
        + (
            " and the current job is being stopped."
            if args.now
            else "; a running job finishes first (use --now to stop it)."
        )
    )
    return 0


def cmd_resume(args) -> int:
    pause_path().unlink(missing_ok=True)
    print("Resumed.")
    return 0


def cmd_status(args) -> int:
    s = Settings.load()
    st = read_status()
    pending = (
        sorted(p.stem for p in approvals_dir().glob("*.pending"))
        if approvals_dir().exists()
        else []
    )
    out = {
        "cli_version": __version__,
        "enrolled": bool(s.machine_id and load_token(s.machine_id)),
        "machine_id": s.machine_id,
        "consent": bool(s.consent),
        "paused": pause_path().exists(),
        "service": service.service_state(),
        "state": st.get("state"),
        "reason": st.get("reason"),
        "gates": st.get("gates"),
        "current_job": st.get("current_job"),
        "last_result": st.get("last_result"),
        "awaiting_approval": pending,
        "updated_at": st.get("updated_at"),
        "settings": {
            k: v for k, v in s.to_dict().items() if k not in ("consent", "extra")
        },
    }
    if getattr(args, "json", False):
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    print(f"llm-speed agent {__version__}")
    print(f"  enrolled : {'yes, ' + str(s.machine_id) if out['enrolled'] else 'no'}")
    print(f"  service  : {out['service']}")
    print(f"  paused   : {'yes' if out['paused'] else 'no'}")
    print(
        f"  state    : {out['state'] or '-'}{(' (' + out['reason'] + ')') if out['reason'] else ''}"
    )
    if out["current_job"]:
        print(f"  job      : {out['current_job']}")
    if out["last_result"]:
        print(f"  last     : {out['last_result']}")
    for aid in pending:
        print(f"  awaiting approval: {aid}  (llm-speed agent approve {aid})")
    return 0


def _decide(args, verdict: str) -> int:
    aid = args.assignment_id
    if not (approvals_dir() / f"{aid}.pending").exists():
        print(f"No job {aid} is waiting for approval.")
        return 1
    (approvals_dir() / f"{aid}.{verdict}").write_text("", encoding="utf-8")
    print(f"{'Approved' if verdict == 'approve' else 'Declined'} {aid}.")
    return 0


def cmd_presets(args) -> int:
    """The (name, version, sha256) triples this machine advertises when it claims."""
    from .allowlist import advertised

    for ident in advertised(Settings.load()):
        print(f"{ident['name']}\tversion {ident['version']}\tsha256 {ident['sha256']}")
    return 0


def cmd_config(args) -> int:
    s = Settings.load()
    if not args.key:
        print(
            json.dumps(
                {k: v for k, v in s.to_dict().items() if k != "extra"},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.key not in SETTABLE:
        print(
            f"unknown or read-only key {args.key!r}; settable: {', '.join(sorted(SETTABLE))}"
        )
        return 2
    if args.value is None:
        print(json.dumps(getattr(s, args.key)))
        return 0
    try:
        val = SETTABLE[args.key](args.value)
        if args.key == "quiet_hours" and val:
            from .gates import parse_quiet_hours

            parse_quiet_hours(val)
    except ValueError as e:
        print(f"bad value: {e}")
        return 2
    setattr(s, args.key, val)
    s.save()
    print(f"{args.key} = {json.dumps(val)}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser(
        "agent",
        help="runner mode: run allowlisted presets from the llm-speed job queue",
    )
    asub = p.add_subparsers(dest="agent_cmd")
    e = asub.add_parser(
        "enroll",
        help="pair this machine (prints a code for the queue admin to approve), store the token",
    )
    e.add_argument("--label", default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--timeout", type=int, default=600)
    i = asub.add_parser(
        "install",
        help="enroll if needed, then install + start the launchd/systemd user service",
    )
    i.add_argument("--label", default=None)
    i.add_argument("--force", action="store_true")
    i.add_argument("--timeout", type=int, default=600)
    i.add_argument(
        "--dry-run", action="store_true", help="print what would be written/run"
    )
    u = asub.add_parser("uninstall", help="stop + remove the service")
    u.add_argument(
        "--forget", action="store_true", help="also delete the token and agent consent"
    )
    u.add_argument("--dry-run", action="store_true")
    r = asub.add_parser("run", help="the poll loop (what the service runs)")
    r.add_argument("--once", action="store_true", help="one claim cycle, then exit")
    pa = asub.add_parser("pause", help="stop taking jobs (local pause file)")
    pa.add_argument("--reason", default=None)
    pa.add_argument(
        "--now", action="store_true", help="also stop the running job and give it back"
    )
    asub.add_parser("resume", help="remove the pause file")
    asub.add_parser(
        "presets", help="list the presets (name, version, sha256) this machine can run"
    )
    asub.add_parser(
        "restart", help="restart the service (after pipx upgrade llm-speed)"
    )
    st = asub.add_parser("status", help="enrollment, service, gates, current/last job")
    st.add_argument("--json", action="store_true")
    for name in ("approve", "decline"):
        a = asub.add_parser(name, help=f"{name} a job held by manual_approve")
        a.add_argument("assignment_id")
    c = asub.add_parser("config", help="show or set local policy")
    c.add_argument("key", nargs="?")
    c.add_argument("value", nargs="?")
    for parser in (p, e, i, r):
        _add_api_base(parser)


API_BASE_HELP = (
    "queue server base URL (default: $LLM_SPEED_API, else https://api.llm-speed.com). "
    "Use it to enroll with a self-hosted queue that implements ops/AGENT_QUEUE_API.md. "
    "The server you enroll with is remembered and the machine token is only ever sent "
    "there, so later commands need no flag"
)


def _add_api_base(parser) -> None:
    # default=SUPPRESS keeps a value given at another level (`llm-speed --api-base URL
    # agent enroll`, `llm-speed agent --api-base URL enroll`, `... enroll --api-base URL`).
    import argparse

    parser.add_argument(
        "--api-base", default=argparse.SUPPRESS, metavar="URL", help=API_BASE_HELP
    )


def dispatch(args) -> int:
    c = getattr(args, "agent_cmd", None)
    table = {
        "enroll": cmd_enroll,
        "install": cmd_install,
        "uninstall": cmd_uninstall,
        "run": cmd_run,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "status": cmd_status,
        "presets": cmd_presets,
        "config": cmd_config,
        "approve": lambda a: _decide(a, "approve"),
        "decline": lambda a: _decide(a, "decline"),
        "restart": lambda a: service.restart(),
    }
    if c not in table:
        print(
            "usage: llm-speed agent {enroll,install,uninstall,run,pause,resume,restart,status,presets,approve,decline,config}"
        )
        return 2
    return table[c](args)
