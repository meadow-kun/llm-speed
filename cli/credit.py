"""Opt-in contributor credit (0.0.10).

A user may attach a public X or Reddit handle to the runs they upload. It is
off by default, never inferred (not from git, the hostname, the environment or
anything else: the only sources are what the user types at the one-time
question, ``llm-speed config credit HANDLE`` and ``--credit HANDLE``), shown
publicly as *unverified*, and never sent under ``--strict-anon`` / ``--anon``
or from an ``llm-speed agent`` job (those children always run ``--strict-anon``).

Stored in ``~/.config/llm-speed/config.json`` (mode 0600)::

    {"credit": {"platform": "x", "handle": "name"} | null,
     "credit_prompted": true, "credit_prompted_at": "<iso8601 utc>"}

Upload shape (outer POST body, next to the JWS; see cli/upload.py)::

    {"jws": "<token>", "credit": {"platform": "x" | "reddit", "handle": "<name>"}}

``handle`` carries no ``@`` / ``u/`` prefix. Case is kept as typed.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("cli.credit")

PLATFORM_X = "x"
PLATFORM_REDDIT = "reddit"
PLATFORMS = (PLATFORM_X, PLATFORM_REDDIT)

X_HANDLE_RE = re.compile(r"[A-Za-z0-9_]{1,15}")
REDDIT_HANDLE_RE = re.compile(r"[A-Za-z0-9_-]{3,20}")

# Exact wording approved for the one-time question (docs/CLI.md quotes it).
PROMPT_LINE_1 = (
    "Want credit for your runs on llm-speed.com? Enter your X or Reddit handle, "
    "or press Enter to stay anonymous."
)
PROMPT_LINE_2 = (
    "It's shown publicly as unverified next to runs you upload; change or remove "
    "it anytime with: llm-speed config credit"
)
PROMPT_INPUT = "Handle (@name or u/name): "
MAX_PROMPT_TRIES = 3

FORMAT_HELP = (
    "use @name for X (1-15 letters, digits or _) or u/name for Reddit "
    "(3-20 letters, digits, _ or -)"
)

# Env vars that mean "no human at the keyboard". Any non-empty value other than
# 0/false counts.
CI_ENV_VARS = (
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "BITBUCKET_BUILD_NUMBER",
    "CODEBUILD_BUILD_ID",
    "DRONE",
    "APPVEYOR",
    "SEMAPHORE",
    "BUILD_BUILDID",
)


class CreditError(ValueError):
    """Invalid handle or a forbidden flag combination. Message is user-facing."""


@dataclass(frozen=True)
class Credit:
    platform: str
    handle: str

    def to_payload(self) -> dict[str, str]:
        return {"platform": self.platform, "handle": self.handle}

    @property
    def display(self) -> str:
        return f"@{self.handle}" if self.platform == PLATFORM_X else f"u/{self.handle}"

    @classmethod
    def from_dict(cls, d: Any) -> Credit | None:
        if not isinstance(d, dict):
            return None
        platform, handle = d.get("platform"), d.get("handle")
        if not isinstance(platform, str) or not isinstance(handle, str):
            return None
        try:
            c = parse_handle(f"@{handle}" if platform == PLATFORM_X else f"u/{handle}")
        except CreditError:
            return None
        return c if c.platform == platform else None


_X_URL = re.compile(
    r"(?:https?://)?(?:www\.|mobile\.)?(?:x|twitter)\.com/@?([^/?#\s]+)/?", re.I
)
_REDDIT_URL = re.compile(
    r"(?:https?://)?(?:www\.|old\.|new\.)?reddit\.com/(?:u|user)/([^/?#\s]+)/?", re.I
)


def parse_handle(raw: str) -> Credit:
    """Normalize user input to a Credit or raise CreditError.

    Accepts ``@name`` (X), ``u/name`` or ``/u/name`` (Reddit), and the plain
    profile URLs ``x.com/name``, ``twitter.com/name``, ``reddit.com/user/name``,
    ``reddit.com/u/name``. A bare ``name`` raises CreditError here (ambiguous
    platform); interactive callers first ask which one via ``resolve_input``.
    """
    if not isinstance(raw, str):
        raise CreditError(f"invalid handle; {FORMAT_HELP}")
    s = raw.strip()
    if not s:
        raise CreditError(f"empty handle; {FORMAT_HELP}")
    m = _X_URL.fullmatch(s)
    if m:
        platform, name = PLATFORM_X, m.group(1)
    else:
        m = _REDDIT_URL.fullmatch(s)
        if m:
            platform, name = PLATFORM_REDDIT, m.group(1)
        elif s.startswith("@"):
            platform, name = PLATFORM_X, s[1:]
        elif s[:2].lower() == "u/":
            platform, name = PLATFORM_REDDIT, s[2:]
        elif s[:3].lower() == "/u/":
            platform, name = PLATFORM_REDDIT, s[3:]
        else:
            raise CreditError(f"{_show(s)} is not an X or Reddit handle; {FORMAT_HELP}")
    rx = X_HANDLE_RE if platform == PLATFORM_X else REDDIT_HANDLE_RE
    if not rx.fullmatch(name):
        label = "X handle" if platform == PLATFORM_X else "Reddit username"
        raise CreditError(f"{_show(s)} is not a valid {label}; {FORMAT_HELP}")
    if is_reserved(name):
        raise CreditError(
            f"{_show(s)} is a reserved name on llm-speed.com (it reads as official or as "
            "someone else); please use your own handle"
        )
    return Credit(platform, name)


# Reserved and impersonation names: a copy of RESERVED_HANDLES / RESERVED_SUBSTRINGS in
# worker/src/credit.ts (ops/CREDIT.md). Compared after normalize_handle(). The server is
# authoritative (it also has a runtime blocklist); this copy only saves a round trip.
RESERVED_HANDLES = frozenset(
    {
        # us
        "llmspeed", "llmspeedcom", "llmspeedbot", "llmspeedofficial", "llmspeedteam",
        "meadowkun",
        # roles and words that read as official or as a label
        "admin", "administrator", "mod", "moderator", "moderators", "support", "help",
        "official", "staff", "team", "root", "system", "sysadmin", "security", "privacy",
        "abuse", "noreply", "null", "undefined", "none", "anonymous", "anon", "unverified",
        "verified", "deleted", "removed", "automoderator", "reddit", "twitter", "x",
        # AI labs, model and runtime brands
        "openai", "chatgpt", "gpt", "sora", "anthropic", "anthropicai", "claude",
        "claudeai", "google", "googleai", "googledeepmind", "deepmind", "gemini",
        "geminiapp", "meta", "metaai", "aiatmeta", "llama", "xai", "grok", "microsoft",
        "msft", "azure", "nvidia", "nvidiaai", "nvidiageforce", "geforce", "amd", "radeon",
        "intel", "apple", "mistral", "mistralai", "huggingface", "hf", "qwen", "alibaba",
        "alibabaqwen", "alibabacloud", "deepseek", "deepseekai", "moonshot", "kimi",
        "zhipu", "zai", "minimax", "cohere", "perplexity", "perplexityai", "unsloth",
        "unslothai", "bartowski", "thebloke", "ggerganov", "ggml", "ggmlorg", "llamacpp",
        "ollama", "lmstudio", "lmstudioai", "vllm", "vllmproject", "mlx", "mlxcommunity",
        "runpod", "cloudflare", "github",
        # well-known people
        "elonmusk", "sama", "samaltman", "sundarpichai", "satyanadella", "jensenhuang",
        "karpathy", "ylecun", "darioamodei", "demishassabis", "gdb", "ilyasut",
        "mustafasuleyman", "lisasu", "timcook", "zuck", "markzuckerberg", "billgates",
        "jeffbezos",
    }
)  # fmt: skip
RESERVED_SUBSTRINGS = ("llmspeed", "meadowkun")


def normalize_handle(handle: str) -> str:
    """Comparison form (same as the server): lowercase, without _ and -."""
    return handle.lower().replace("_", "").replace("-", "")


# Words that, next to a reserved name, still read as that name (same list as
# web/lib/reserved-handles.ts RESERVED_AFFIXES).
RESERVED_AFFIXES = (
    "official", "team", "support", "ai", "hq", "app", "bot", "dev", "labs", "news",
    "real", "the", "its", "iam",
)  # fmt: skip
_MIN_FUZZY = 4
_SKELETON_MAP = (("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("i", "l"), ("|", "l"),
                 ("3", "e"), ("4", "a"), ("5", "s"), ("7", "t"), ("8", "b"))  # fmt: skip


def _confusable(n: str) -> str:
    for a, b in _SKELETON_MAP:
        n = n.replace(a, b)
    return n


def skeleton(handle: str) -> str:
    """Confusable skeleton (same as the server's): trailing digits dropped, then
    rn->m, vv->w, 0->o, 1/i/|->l, 3->e, 4->a, 5->s, 7->t, 8->b."""
    return _confusable(re.sub(r"\d+$", "", normalize_handle(handle)))


def skeletons(handle: str) -> list[str]:
    """Every reading the reserved check tries (security retest R-20, same as the
    server's skeletons()): trailing digits dropped ("karpathy2") and trailing digits
    read as letters ("OpenA1" -> openal, "c1aud3" -> claude), then any digits the map
    left at the end dropped ("claud32" -> claude, "5am42" -> sama)."""
    n = normalize_handle(handle)
    mapped = _confusable(n)
    return list(
        dict.fromkeys(
            [_confusable(re.sub(r"\d+$", "", n)), mapped, re.sub(r"\d+$", "", mapped)]
        )
    )


_RESERVED_SKELETONS = frozenset(
    skeleton(r) for r in RESERVED_HANDLES if len(r) >= _MIN_FUZZY
)
_AFFIX_SKELETONS = frozenset(skeleton(a) for a in RESERVED_AFFIXES)


def is_reserved(handle: str) -> bool:
    """Same rule as web/lib/reserved-handles.ts reservedReason (security retest R-5):
    exact names, our own names anywhere, look-alikes (skeleton) and a reserved name of
    5+ characters with an official-sounding word before or after it."""
    n = normalize_handle(handle)
    if n in RESERVED_HANDLES or any(sub in n for sub in RESERVED_SUBSTRINGS):
        return True
    for k in skeletons(handle):
        if k in _RESERVED_SKELETONS or any(
            skeleton(sub) in k for sub in RESERVED_SUBSTRINGS
        ):
            return True
        for r in _RESERVED_SKELETONS:
            if len(r) < 5 or len(k) <= len(r):
                continue
            if k.startswith(r) and k[len(r) :] in _AFFIX_SKELETONS:
                return True
            if k.endswith(r) and k[: len(k) - len(r)] in _AFFIX_SKELETONS:
                return True
    return False


PLATFORM_QUESTION = "X or Reddit? [x/r]: "
_BARE_RE = re.compile(r"[A-Za-z0-9_-]+")


class BareNameNeedsPlatform(CreditError):
    """A bare name was given where no one can be asked which platform it is."""


def is_bare_name(raw: str) -> bool:
    """True for input with no @ / u/ prefix and no URL, e.g. ``speedy``."""
    return isinstance(raw, str) and bool(_BARE_RE.fullmatch(raw.strip()))


def bare_name_error(raw: str) -> BareNameNeedsPlatform:
    name = raw.strip()
    return BareNameNeedsPlatform(
        f"{_show(name)} needs a platform: use @{name} for X or u/{name} for Reddit "
        "(there's no terminal to ask which one)"
    )


def ask_platform(name: str, *, stdin=None, stdout=None) -> Credit | None:
    """Ask exactly one follow-up for a bare name. Returns the Credit, or None
    on Enter / EOF / an answer other than x or r. Raises CreditError when the
    name doesn't fit the chosen platform. Ctrl-C propagates."""
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    out.write(PLATFORM_QUESTION)
    out.flush()
    answer = inp.readline().strip().lower()
    if answer == "x":
        return parse_handle(f"@{name.strip()}")
    if answer == "r":
        return parse_handle(f"u/{name.strip()}")
    return None


def resolve_input(raw: str, *, can_ask: bool, stdin=None, stdout=None) -> Credit | None:
    """parse_handle, plus the bare-name follow-up when someone can answer it.
    None means the follow-up got no usable platform answer."""
    if is_bare_name(raw):
        if not can_ask:
            raise bare_name_error(raw)
        return ask_platform(raw, stdin=stdin, stdout=stdout)
    return parse_handle(raw)


def _show(s: str) -> str:
    s = "".join(ch if ch.isprintable() else "?" for ch in s)
    return repr(s[:40] + ("…" if len(s) > 40 else ""))


# ---------------------------------------------------------------- config file


def config_path() -> Path:
    from . import config as _cfg  # read at call time so tests can redirect it

    return Path(_cfg.CONFIG_DIR) / "config.json"


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("config unreadable (%s); ignoring", exc)
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
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


def configured_credit() -> Credit | None:
    return Credit.from_dict(load_config().get("credit"))


def _mark_prompted(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg["credit_prompted"] = True
    cfg.setdefault("credit_prompted_at", dt.datetime.now(dt.timezone.utc).isoformat())
    return cfg


def set_configured_credit(credit: Credit | None) -> None:
    cfg = _mark_prompted(load_config())
    cfg["credit"] = credit.to_payload() if credit else None
    save_config(cfg)


def was_prompted() -> bool:
    return bool(load_config().get("credit_prompted"))


# ------------------------------------------------------------- per-run logic


def check_flags(args, *, stdin=None, stdout=None) -> None:
    """Reject forbidden combinations before any work starts (exit 2 at caller).

    A bare ``--credit NAME`` gets the one "X or Reddit?" follow-up when a human
    is at a terminal (then ``args.credit`` is rewritten to ``@NAME`` / ``u/NAME``);
    otherwise it is an error telling them to use @name or u/name."""
    raw = getattr(args, "credit", None)
    if raw is None:
        return
    if getattr(args, "no_credit", False):
        raise CreditError("--credit and --no-credit can't be used together")
    if getattr(args, "strict_anon", False):
        raise CreditError(
            "--credit can't be used with --strict-anon (strict-anon never sends a handle)"
        )
    if getattr(args, "anon", False):
        raise CreditError(
            "--credit can't be used with --anon (anon uploads never send a handle)"
        )
    if is_bare_name(raw):
        can_ask = can_prompt(args, stdin=stdin, stdout=stdout)
        c = resolve_input(raw, can_ask=can_ask, stdin=stdin, stdout=stdout)
        if c is None:
            raise CreditError(
                f"no platform chosen for --credit {raw.strip()}; use @name for X or u/name for Reddit"
            )
        args.credit = f"@{c.handle}" if c.platform == PLATFORM_X else f"u/{c.handle}"
        return
    parse_handle(raw)


def effective_credit(args) -> Credit | None:
    """The credit for this run's upload, or None. Never under any anon mode."""
    if getattr(args, "strict_anon", False) or getattr(args, "anon", False):
        return None
    if getattr(args, "no_credit", False):
        return None
    raw = getattr(args, "credit", None)
    if raw is not None:
        return parse_handle(raw)
    return configured_credit()


def ci_detected(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    for k in CI_ENV_VARS:
        v = env.get(k)
        if (
            v is not None
            and v.strip()
            and v.strip().lower() not in ("0", "false", "no")
        ):
            return True
    return False


def interactive(stdin=None, stdout=None) -> bool:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        return bool(stdin.isatty()) and bool(stdout.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def can_prompt(args, *, stdin=None, stdout=None) -> bool:
    """A human can answer a question now: both TTYs, no CI, not --yes/--auto."""
    if getattr(args, "yes", False) or getattr(args, "no_prompt", False):
        return False
    if ci_detected():
        return False
    return interactive(stdin, stdout)


def should_ask(args, *, stdin=None, stdout=None) -> bool:
    """True only when a human can answer and we have never asked before."""
    if getattr(args, "strict_anon", False) or getattr(args, "anon", False):
        return False
    if getattr(args, "yes", False) or getattr(args, "no_prompt", False):
        return False
    if getattr(args, "no_credit", False) or getattr(args, "credit", None) is not None:
        return False
    if ci_detected():
        return False
    if not interactive(stdin, stdout):
        return False
    cfg = load_config()
    if cfg.get("credit_prompted") or Credit.from_dict(cfg.get("credit")):
        return False
    return True


def maybe_ask(args, *, stdin=None, stdout=None) -> Credit | None:
    """Ask the one-time credit question right after upload consent.

    Returns the chosen Credit (also saved) or None. Records that it was asked
    whatever the answer, so it is never asked again. Never reads stdin unless
    ``should_ask`` is True. Ctrl-C propagates (the run is already saved
    locally; nothing is uploaded).
    """
    if not should_ask(args, stdin=stdin, stdout=stdout):
        return None
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    out.write("\n" + PROMPT_LINE_1 + "\n" + PROMPT_LINE_2 + "\n")
    chosen: Credit | None = None
    for attempt in range(MAX_PROMPT_TRIES):
        out.write(PROMPT_INPUT)
        out.flush()
        line = inp.readline()
        if not line or not line.strip():  # EOF or Enter: anonymous
            break
        try:
            if is_bare_name(line):
                # One follow-up; no usable answer means anonymous.
                chosen = ask_platform(line, stdin=inp, stdout=out)
                break
            chosen = parse_handle(line)
            break
        except CreditError as exc:
            out.write(f"  {exc}\n")
            if attempt == MAX_PROMPT_TRIES - 1:
                out.write(
                    "  Staying anonymous. Set it later with: llm-speed config credit @name\n"
                )
    set_configured_credit(chosen)
    if chosen:
        out.write(
            f"  Saved. Runs you upload will show {chosen.display} (unverified). "
            "Remove it with: llm-speed config credit --clear\n"
        )
    else:
        out.write("  Staying anonymous. You won't be asked again.\n")
    out.flush()
    return chosen


# ------------------------------------------------------- `llm-speed config`


def cmd_config(args) -> int:
    key = getattr(args, "config_key", None)
    if key != "credit":
        print("usage: llm-speed config credit [HANDLE | --clear]", file=sys.stderr)
        return 2
    value = getattr(args, "value", None)
    clear = getattr(args, "clear", False)
    if clear and value is not None:
        print("error: give a HANDLE or --clear, not both", file=sys.stderr)
        return 2
    if clear:
        had = configured_credit()
        set_configured_credit(None)
        print(
            "Credit removed. New uploads are anonymous."
            if had
            else "No credit was set. Uploads are anonymous."
        )
        if had:
            print(
                "Runs you already uploaded keep the handle until you ask "
                "privacy@llm-speed.com to remove it (see https://llm-speed.com/privacy)."
            )
        return 0
    if value is None:
        cur = configured_credit()
        if cur:
            print(
                f"Credit: {cur.display} ({'X' if cur.platform == PLATFORM_X else 'Reddit'}), "
                "shown publicly as unverified next to runs you upload."
            )
        else:
            print("Credit: none. Uploads are anonymous.")
        print(
            "Change: llm-speed config credit @name | u/name    Remove: llm-speed config credit --clear"
        )
        return 0
    try:
        c = resolve_input(value, can_ask=can_prompt(args))
    except CreditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if c is None:
        print(
            "error: no platform chosen; use @name for X or u/name for Reddit. Nothing changed.",
            file=sys.stderr,
        )
        return 2
    set_configured_credit(c)
    print(
        f"Credit set: {c.display}. It's shown publicly as unverified next to runs you upload "
        "(never with --strict-anon or --anon)."
    )
    return 0
