"""`llm-speed agent ...` subcommands."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import sys
import time
from dataclasses import dataclass

from .. import __version__
from ..api_host import display_host, same_api_base, token_allowed, token_opt_in_hint
from ..config import DEFAULT_API_BASE
from . import service
from .label import LabelError, generate_default_label, is_valid, validate_label
from .runner import Runner, approvals_dir, capabilities, read_status
from .settings import (
    LEGACY_KEYS,
    SETTABLE,
    Settings,
    atomic_write_json,
    ensure_dirs,
    logs_dir,
    pause_path,
)
from .tokens import (
    RedactFilter,
    Stored,
    TokenStoreError,
    delete_token,
    describe_store,
    load_token,
    save_token,
)
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
    if settings.api_base:
        enrolled = settings.api_base.rstrip("/")
    else:
        # Enrolled before the server was remembered: "the default". N-5: a default
        # that only comes from $LLM_SPEED_API and isn't api.llm-speed.com gets the
        # token only if the user opted in to that host.
        enrolled = DEFAULT_API_BASE.rstrip("/")
        if not token_allowed(enrolled, args):
            raise QueueError(
                0,
                "token_host_not_allowed",
                f"not sending this machine's token to {display_host(enrolled)} (from "
                f"$LLM_SPEED_API; it is not api.llm-speed.com). Pass "
                f"{token_opt_in_hint(enrolled)} if that is your queue, or re-enroll "
                "with `llm-speed agent enroll --force --api-base URL`",
            )
    if asked and not same_api_base(asked, enrolled):
        raise QueueError(
            0,
            "api_base_mismatch",
            f"this machine is enrolled with {enrolled}; its token is only sent there "
            "(run `llm-speed agent enroll --force --api-base URL` to switch)",
        )
    return QueueClient(enrolled, load_token(settings.machine_id))


def _enroll_base_refusal(args) -> str | None:
    """N-5: enrolling makes the queue the one host this machine's token is sent to.
    `--api-base URL` on the command line is that explicit choice; a non-default server
    that only comes from $LLM_SPEED_API needs the token opt-in for that exact host
    (an inherited environment variable is not a choice)."""
    if getattr(args, "api_base", None):
        return None
    base = DEFAULT_API_BASE.rstrip("/")
    if token_allowed(base, args):
        return None
    host = display_host(base)
    return (
        f"Not enrolling with {host}: it comes from $LLM_SPEED_API, not "
        f"api.llm-speed.com, and the machine token would be sent there. If {host} is "
        f"your queue, run `llm-speed agent enroll --api-base {base}` or pass "
        f"{token_opt_in_hint(base)}."
    )


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


def _pairing_label(s: Settings, user_label: str | None, caps: dict) -> str:
    """--label wins; else the saved label; else generate `<chip-slug>-<random>` once
    and save it so re-pairing keeps the same name (never hostname/username)."""
    if user_label:
        return user_label
    if s.label and is_valid(s.label):
        return s.label.strip()
    s.label = generate_default_label(caps)
    s.save()
    return s.label


def _store_token(mid: str, token: str, args) -> Stored | None:
    """Keep the once-delivered token: the login Keychain if it works, else the 0600
    file (same process, token only in memory meanwhile). None = no store could keep
    it; the message says what to do. The token is never printed."""
    prefer = getattr(args, "token_store", None)
    try:
        return save_token(mid, token, None if prefer in (None, "auto") else prefer)
    except (TokenStoreError, ValueError) as e:
        print(
            f"Approved, but the machine token could not be stored: {e}.\n"
            "The server hands the token out only once, so this pairing is used up and "
            "nothing was saved.\nNext: fix the cause above, ask your queue admin to "
            f"revoke machine {mid}, then run `llm-speed agent install` again "
            "(it prints a new pairing code to approve)."
        )
        return None


def _explain_fallback(stored: Stored) -> None:
    if not stored.fallback_reason:
        return
    print(
        f"Note: the login Keychain could not be used ({stored.fallback_reason}),\n"
        f"so the token is in {stored.where} instead (directory 0700). The service reads "
        "it from there without an unlocked Keychain; nothing else to do.\n"
        "To keep it in the Keychain instead, run `llm-speed agent enroll --force "
        "--token-store keychain` in Terminal in the logged-in desktop session "
        "(needs a new approval)."
    )


def cmd_enroll(args, *, apply_now: bool = True) -> int:
    """Pair this machine. After a successful pairing, ask the signed-tests setup
    question once (`apply_now`: restart a running agent so a yes applies; `agent
    install` passes False because it starts the service itself)."""
    ensure_dirs()
    s = Settings.load()
    if s.machine_id and load_token(s.machine_id) and not getattr(args, "force", False):
        print(f"Already enrolled as {s.machine_id}. Use --force to enroll again.")
        if _setup_due(s, args) and _flag_or_terminal(args):
            setup_dialog(s, args, apply_now=apply_now, upgrade=True)
        return 0
    refusal = _enroll_base_refusal(args)
    if refusal:
        print(refusal)
        return 2
    from .enrolment import EnrolmentTokenError
    from .enrolment import take as take_enrolment_token

    if getattr(args, "enrolment_token_argv", None) is not None:
        print(
            "error: don't pass the enrolment token on the command line (other users can see "
            "it, and it is now in your shell history; ask your queue admin for a new one). Put "
            "it in a file only you can read (chmod 600) and pass --enrolment-token-file PATH, "
            "or set LLM_SPEED_ENROLMENT_TOKEN.",
            file=sys.stderr,
        )
        return 2
    try:
        # Read (and drop $LLM_SPEED_ENROLMENT_TOKEN) before anything else can start.
        enrol = take_enrolment_token(getattr(args, "enrolment_token_file", None))
    except EnrolmentTokenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    # Where it came from, for the "you can delete it now" note (never the path or value).
    enrol_source = (
        None
        if enrol is None
        else ("file" if getattr(args, "enrolment_token_file", None) else "env")
    )
    user_label = getattr(args, "label", None)
    if user_label is not None:
        try:
            user_label = validate_label(user_label)
        except LabelError as e:
            print(f"Invalid --label: {e}. Nothing was sent.")
            return 2
    if not _ask_consent():
        print("Not enrolled (no consent). Nothing was stored.")
        return 1
    try:
        client = _client(s, args, auth=False)
        caps = capabilities(s)
        label = _pairing_label(s, user_label, caps)
        p = client.pair_start(caps, label, enrol)
    except QueueError as e:
        if e.status == 401 and e.code == "invalid_enrolment":
            print(
                "The queue refused the enrolment token (wrong, expired or already used). "
                "Not retrying without it; ask your queue admin for a new one."
            )
            return 1
        print(f"Could not start pairing: {e}")
        return 1
    finally:
        enrol = None  # forgotten after the pair request (201 or not)
    if enrol_source is not None:
        from .enrolment import DELETE_FILE_NOTE, UNSET_ENV_NOTE

        # The queue took the token (it is single-use); nothing here keeps a copy.
        print(DELETE_FILE_NOTE if enrol_source == "file" else UNSET_ENV_NOTE)
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
                stored = _store_token(mid, token, args)
                token = ""
                if stored is None:
                    return 1
                s.machine_id, s.label = mid, label
                s.api_base = client.api_base
                s.token_store = stored.kind
                s.consent = {
                    "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "cli_version": __version__,
                    "text_sha256": hashlib.sha256(CONSENT_TEXT.encode()).hexdigest(),
                }
                s.save()
                print(f"Approved. Enrolled as {mid}; token stored in {stored.where}.")
                _explain_fallback(stored)
                if _setup_due(s, args, enrolling=True):
                    setup_dialog(s, args, apply_now=apply_now, enrolling=True)
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
        print("\nStopped waiting. No token was stored.")
        return 1
    print("Pairing timed out. Run `llm-speed agent enroll` again.")
    return 1


def cmd_install(args) -> int:
    if service.platform_kind() in ("windows", "unsupported"):
        # before enrolling: a pairing approved for a machine that can't run the service
        # would be wasted
        print(
            service.WINDOWS_MESSAGE
            if service.platform_kind() == "windows"
            else "unsupported platform"
        )
        return 1
    prefer = getattr(args, "token_store", None)
    bad = service.preflight(
        dry_run=getattr(args, "dry_run", False),
        token_store=None if prefer in (None, "auto") else prefer,
    )
    if bad:
        # before enrolling, for the same reason (and a single-use enrolment token)
        print(bad)
        print("Nothing was paired; an enrolment token, if you have one, was not used.")
        return 1
    s = Settings.load()
    dry = getattr(args, "dry_run", False)
    was = _setup_values(s)
    if not dry and not (s.machine_id and load_token(s.machine_id)):
        # The setup question comes after pairing; the service started below reads it.
        rc = cmd_enroll(args, apply_now=False)
        if rc != 0:
            return rc
    elif not dry and _setup_due(s, args) and _flag_or_terminal(args):
        setup_dialog(s, args, apply_now=False, upgrade=True)
    # A changed setting on an agent that is already running needs a restart after the
    # install below (`systemctl --user enable --now` leaves a running unit as it was).
    changed = not dry and _setup_values(Settings.load()) != was
    before = service.service_state() if changed else None
    rc, steps = service.install(dry_run=dry)
    for line in steps:
        print(line)
    if dry:
        print("(dry run: nothing was written or started)")
    elif rc == 0:
        print("Service installed and started.")
        if changed and not _not_running(before):
            _apply_now()
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
    for key in ("busy_util_pct", "busy_window_s", "busy_samples"):
        val = getattr(args, key, None)
        if val is not None:  # this run only; `agent config KEY VALUE` saves it
            try:
                setattr(s, key, SETTABLE[key](str(val)))
            except ValueError as e:
                print(f"bad --{key.replace('_', '-')}: {e}")
                return 2
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


def _status_backends(s: Settings) -> dict | None:
    if not s.machine_id:
        return None
    from . import backends

    try:
        return backends.status_probe(s)
    except OSError:
        return None


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
        "token_store": describe_store(s.token_store) if s.machine_id else None,
        "consent": bool(s.consent),
        "paused": pause_path().exists(),
        "service": service.service_state(),
        "state": st.get("state"),
        "reason": st.get("reason"),
        "gates": st.get("gates"),
        "current_job": st.get("current_job"),
        "last_result": st.get("last_result"),
        "last_preset_refusal": st.get("last_preset_refusal"),
        "signed_presets": s.signed_presets is True,
        "signed_presets_asked": s.signed_presets_asked is True,
        "auto_install_backends": s.auto_install_backends is True,
        "auto_install_asked": s.auto_install_asked is True,
        "backends": _status_backends(s),
        "last_backend_check": st.get("backends"),
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
    if out["token_store"]:
        print(f"  token    : {out['token_store']}")
    print(f"  service  : {out['service']}")
    print(f"  paused   : {'yes' if out['paused'] else 'no'}")
    print(f"  signed presets: {signed_presets_status_line(s)}")
    print(f"  auto-install  : {auto_install_status_line(s)}")
    if out["backends"] is not None:
        from .backends import describe

        print(
            f"  backends : {describe(out['backends'])} (looked up as the service does)"
        )
    print(
        f"  state    : {out['state'] or '-'}{(' (' + out['reason'] + ')') if out['reason'] else ''}"
    )
    if out["current_job"]:
        print(f"  job      : {out['current_job']}")
    if out["last_result"]:
        print(f"  last     : {out['last_result']}")
    ref = out["last_preset_refusal"]
    if ref:
        print(
            f"  refused  : signed preset {ref.get('preset')} v{ref.get('version')} "
            f"({ref.get('reason')}) at {ref.get('at')}; job {ref.get('job_id')} handed back"
        )
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


SIGNED_PRESETS_PROMPT = (
    "Turn on signed presets for this machine?\n"
    "  This machine will run benchmark tests published by llm-speed.com: presets the\n"
    "  queue signed with a key built into this CLI, without a CLI release for each one.\n"
    "  Each one is checked (signature, revocation list, the same schema as packaged\n"
    "  presets) before it runs; it can download models from allowlisted Hugging Face\n"
    "  orgs and run llama.cpp / MLX with allowlisted flags only. Uploads stay\n"
    "  --strict-anon. Turn it off any time: llm-speed agent signed-presets off\n"
)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# 0.0.13 setup dialog (Boss: "set when it installs and by the user willingly"): one
# y/N question per setting, asked once. Wording approved by Boss
# 2026-10-31 (docs/CLI_CHANGELOG.md 0.0.13).
SIGNED_PRESETS_SETUP_TEXT = (
    "Allow signed tests from llm-speed.com on this machine?\n"
    "  The agent may run benchmark tests published and signed by llm-speed.com.\n"
    "  Each test's signature is checked before it runs. Turn it off any time:\n"
    "  llm-speed agent signed-presets off\n"
)
SIGNED_PRESETS_SETUP_QUESTION = "Allow signed tests? [y/N] "
SIGNED_PRESETS_SETUP_NO = (
    "Signed tests stay off. To allow them later: llm-speed agent signed-presets on"
)
SIGNED_PRESETS_NONINTERACTIVE_HINT = (
    "Signed tests from llm-speed.com: off (not asked: no terminal). To allow them: "
    "llm-speed agent signed-presets on"
)
AUTO_INSTALL_SETUP_TEXT = (
    "Let the agent install a missing backend on this machine?\n"
    "  When a test needs llama.cpp or MLX and it isn't installed, the agent may install\n"
    "  it: the official llama.cpp prebuilt into ~/.local/share/llm-speed/llama.cpp (no\n"
    "  sudo), or mlx-lm into the agent's own Python environment. Turn it off any time:\n"
    "  llm-speed agent auto-install off\n"
)
AUTO_INSTALL_SETUP_QUESTION = "Allow automatic backend installs? [y/N] "
AUTO_INSTALL_SETUP_NO = (
    "Automatic backend installs stay off. To allow them later: "
    "llm-speed agent auto-install on"
)
AUTO_INSTALL_NONINTERACTIVE_HINT = (
    "Automatic backend installs: off (not asked: no terminal). To allow them: "
    "llm-speed agent auto-install on"
)
SETUP_UPGRADE_LEAD = "New settings for this machine (each asked once, never again):\n"


@dataclass(frozen=True)
class _Setting:
    attr: str  # Settings field (on/off)
    asked: str  # Settings field (the question was answered)
    flag: str  # argparse dest of --X / --no-X on enroll and install
    label: str  # "signed presets: on"
    command: str  # llm-speed agent COMMAND on|off
    text: str
    question: str
    no: str
    hint: str


SETUP_SETTINGS = (
    _Setting(
        "signed_presets",
        "signed_presets_asked",
        "signed_presets",
        "signed presets",
        "signed-presets",
        SIGNED_PRESETS_SETUP_TEXT,
        SIGNED_PRESETS_SETUP_QUESTION,
        SIGNED_PRESETS_SETUP_NO,
        SIGNED_PRESETS_NONINTERACTIVE_HINT,
    ),
    _Setting(
        "auto_install_backends",
        "auto_install_asked",
        "auto_install_backends",
        "auto-install",
        "auto-install",
        AUTO_INSTALL_SETUP_TEXT,
        AUTO_INSTALL_SETUP_QUESTION,
        AUTO_INSTALL_SETUP_NO,
        AUTO_INSTALL_NONINTERACTIVE_HINT,
    ),
)


def signed_presets_status_line(s: Settings) -> str:
    if s.signed_presets is True:
        return (
            "on (signed tests from llm-speed.com allowed; "
            "turn off: llm-speed agent signed-presets off)"
        )
    return "off (turn on: llm-speed agent signed-presets on)"


def auto_install_status_line(s: Settings) -> str:
    if s.auto_install_backends is True:
        return (
            "on (the agent may install llama.cpp / mlx-lm a job needs; "
            "turn off: llm-speed agent auto-install off)"
        )
    return "off (turn on: llm-speed agent auto-install on)"


def _setup_values(s: Settings) -> tuple:
    return tuple(getattr(s, q.attr) is True for q in SETUP_SETTINGS)


def _not_running(state: str | None) -> bool:
    return (
        state is None
        or state in ("not installed", "unsupported", "inactive", "failed")
        or state.startswith("unknown")
    )


def _apply_now() -> int:
    """Make a setting change take effect: restart an installed agent service (launchd
    `kickstart -k`, systemd `restart`, WSL2 logon task end + run). Nothing is started
    where no service is installed; the agent reads its settings when it starts."""
    state = service.service_state()
    if state in ("not installed", "unsupported") or state.startswith("unknown"):
        print(
            "The agent service is not installed here; the setting applies when the "
            "agent starts."
        )
        return 0
    rc = service.restart()
    if rc == 0:
        print("Agent restarted; the setting applies now.")
    else:
        print(f"Could not restart the agent (exit {rc}). Run: llm-speed agent restart")
    return rc


def _record(s: Settings, q: _Setting, on: bool) -> None:
    setattr(s, q.attr, on is True)
    setattr(s, q.asked, True)
    s.save()


def _due(s: Settings, args, q: _Setting, enrolling: bool) -> bool:
    """Ask (or apply the explicit flag) at enrolment, or once on an enrolled machine that
    was never asked and doesn't have it on. An explicit --[no-]FLAG always applies."""
    if getattr(args, q.flag, None) is not None:
        return True
    if enrolling:
        return getattr(s, q.asked) is not True
    return getattr(s, q.asked) is not True and getattr(s, q.attr) is not True


def _setup_due(s: Settings, args, *, enrolling: bool = False) -> bool:
    return any(_due(s, args, q, enrolling) for q in SETUP_SETTINGS)


def _flag_or_terminal(args) -> bool:
    return (
        any(getattr(args, q.flag, None) is not None for q in SETUP_SETTINGS)
        or _interactive()
    )


def setup_dialog(
    s: Settings,
    args,
    *,
    apply_now: bool = True,
    upgrade: bool = False,
    enrolling: bool = False,
) -> None:
    """The setup questions (Security Desk R-93 rules for each: on only by an explicit
    yes typed in a terminal or an explicit --X flag; Enter = no; no terminal and no
    flag = stays off, not recorded, one-line hint). One restart at the end when a
    value changed and `apply_now`."""
    due = [q for q in SETUP_SETTINGS if _due(s, args, q, enrolling)]
    if not due:
        return
    before = _setup_values(s)
    led = False
    for q in due:
        flag = getattr(args, q.flag, None)
        if flag is True or flag is False:
            _record(s, q, flag)
            print(
                f"{q.label}: {'on' if flag else 'off'} (from --{'' if flag else 'no-'}"
                f"{q.flag.replace('_', '-')})"
            )
            continue
        if not _interactive():
            print(q.hint)
            continue
        print()
        if not led and upgrade:
            print(SETUP_UPGRADE_LEAD, end="")
        led = True
        print(q.text, end="")
        try:
            ans = input(q.question).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nNot answered; it stays off (asked again next time).")
            break
        if ans in ("y", "yes"):
            _record(s, q, True)
            print(f"{q.label}: on")
        else:
            _record(s, q, False)
            print(q.no)
    if apply_now and _setup_values(s) != before:
        _apply_now()


# Interactive commands that ask the setup questions once on an upgraded, enrolled
# machine. Never `run` (what the service starts) and never machine-readable output.
_UPGRADE_ASK_CMDS = (None, "status", "restart", "presets")


def _maybe_ask_upgraded(c, args) -> None:
    if c not in _UPGRADE_ASK_CMDS or getattr(args, "json", False):
        return
    if not _interactive():
        return
    s = Settings.load()
    if not (s.machine_id and s.consent) or not _setup_due(s, args):
        return
    # `agent restart` restarts right after, so no second restart here.
    setup_dialog(s, args, apply_now=c != "restart", upgrade=True)
    print()


def _on_off_command(args, q: _Setting, confirm_text: str, confirm_q: str) -> int:
    """`agent signed-presets|auto-install [on|off|status] [--yes]`. Both on and off
    restart an installed agent so they apply now (R-103)."""
    s = Settings.load()
    want = getattr(args, "state", None)
    if want in (None, "status"):
        print(f"{q.label}: {'on' if getattr(s, q.attr) is True else 'off'}")
        return 0
    if want == "off":
        _record(s, q, False)
        print(
            "signed presets: off (this machine runs packaged presets only)"
            if q is SETUP_SETTINGS[0]
            else f"{q.label}: off (a job whose backend is missing fails backend_missing)"
        )
        _apply_now()
        return 0
    if not getattr(args, "yes", False):
        print(confirm_text)
        if not _interactive():
            print("Not a terminal: re-run with --yes to confirm. Nothing changed.")
            return 2
        try:
            ans = input(confirm_q).strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print(f"Nothing changed ({q.label} stays off).")
            return 1
    _record(s, q, True)
    print(f"{q.label}: on")
    _apply_now()
    return 0


def cmd_signed_presets(args) -> int:
    """Per-machine opt-in for server-signed presets (Security Desk R-93). Off is the
    kill switch."""
    return _on_off_command(
        args, SETUP_SETTINGS[0], SIGNED_PRESETS_PROMPT, "Turn on signed presets? [y/N] "
    )


def cmd_auto_install(args) -> int:
    """Per-machine opt-in for the agent installing a missing backend (0.0.13)."""
    return _on_off_command(
        args, SETUP_SETTINGS[1], AUTO_INSTALL_SETUP_TEXT, AUTO_INSTALL_SETUP_QUESTION
    )


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
    if args.key in LEGACY_KEYS:
        print(f"{args.key} was removed in 0.0.10: {LEGACY_KEYS[args.key]}")
        return 2
    if args.key == "signed_presets" and args.value is not None:
        print("use `llm-speed agent signed-presets on|off` (it asks to confirm)")
        return 2
    if args.key == "auto_install_backends" and args.value is not None:
        print("use `llm-speed agent auto-install on|off` (it asks to confirm)")
        return 2
    if args.key not in SETTABLE and args.key not in (
        "signed_presets",
        "auto_install_backends",
    ):
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
    # A token on the command line is visible to every user (ps) and in shell history: the
    # flag exists only to refuse it without echoing the value (exact match beats the
    # --enrolment-token-file prefix).
    e.add_argument(
        "--enrolment-token",
        dest="enrolment_token_argv",
        default=None,
        help=argparse.SUPPRESS,
    )
    e.add_argument(
        "--enrolment-token-file",
        metavar="PATH",
        default=None,
        help="single-use enrolment token from your queue admin, read from a file only you "
        "can read (chmod 600); or set $LLM_SPEED_ENROLMENT_TOKEN. Never pass it on the command line",
    )
    i = asub.add_parser(
        "install",
        help="enroll if needed, then install + start the launchd / systemd user service (WSL2: a Windows logon task)",
    )
    i.add_argument("--label", default=None)
    i.add_argument("--force", action="store_true")
    i.add_argument("--timeout", type=int, default=600)
    # A token on the command line is visible to every user (ps) and in shell history: the
    # flag exists only to refuse it without echoing the value (exact match beats the
    # --enrolment-token-file prefix).
    i.add_argument(
        "--enrolment-token",
        dest="enrolment_token_argv",
        default=None,
        help=argparse.SUPPRESS,
    )
    i.add_argument(
        "--enrolment-token-file",
        metavar="PATH",
        default=None,
        help="single-use enrolment token from your queue admin, read from a file only you "
        "can read (chmod 600); or set $LLM_SPEED_ENROLMENT_TOKEN. Never pass it on the command line",
    )
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
    r.add_argument(
        "--busy-util-pct",
        type=int,
        default=None,
        help="GPU busy above this utilization %% (this run; default: agent config, 15)",
    )
    r.add_argument(
        "--busy-window-s",
        type=float,
        default=None,
        help="seconds to sample GPU utilization over (this run; default 15)",
    )
    r.add_argument(
        "--busy-samples",
        type=int,
        default=None,
        help="utilization samples per window (this run; default 5)",
    )
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
    sp = asub.add_parser(
        "signed-presets",
        help="on/off: run benchmark tests published (signed) by llm-speed.com on this machine",
    )
    sp.add_argument("state", nargs="?", choices=("on", "off", "status"))
    sp.add_argument(
        "--yes", action="store_true", help="confirm turning it on without a prompt"
    )
    ai = asub.add_parser(
        "auto-install",
        help="on/off: let the agent install a missing llama.cpp (official prebuilt, no "
        "sudo) or mlx-lm (its own venv) when a job needs it",
    )
    ai.add_argument("state", nargs="?", choices=("on", "off", "status"))
    ai.add_argument(
        "--yes", action="store_true", help="confirm turning it on without a prompt"
    )
    c = asub.add_parser("config", help="show or set local policy")
    c.add_argument("key", nargs="?")
    c.add_argument("value", nargs="?")
    for parser in (p, e, i, r):
        _add_api_base(parser)
    for parser in (e, i):
        parser.add_argument(
            "--signed-presets",
            dest="signed_presets",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="allow (or not) signed tests from llm-speed.com on this machine without "
            "asking (for scripted installs; default: ask once in a terminal, Enter = no)",
        )
        parser.add_argument(
            "--auto-install-backends",
            dest="auto_install_backends",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="allow (or not) the agent to install a missing backend a job needs, "
            "without asking (for scripted installs; default: ask once in a terminal, "
            "Enter = no)",
        )
        parser.add_argument(
            "--token-store",
            choices=("auto", "keychain", "file"),
            default="auto",
            help="where to keep the machine token (auto: login Keychain on macOS, "
            "falling back to a 0600 file if the Keychain can't be used; file elsewhere)",
        )


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
        "signed-presets": cmd_signed_presets,
        "auto-install": cmd_auto_install,
        "approve": lambda a: _decide(a, "approve"),
        "decline": lambda a: _decide(a, "decline"),
        "restart": lambda a: service.restart(),
    }
    _maybe_ask_upgraded(c, args)
    if c not in table:
        print(
            "usage: llm-speed agent {enroll,install,uninstall,run,pause,resume,restart,status,presets,approve,decline,config,signed-presets,auto-install}"
        )
        return 2
    return table[c](args)
