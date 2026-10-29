"""The allowlist: the only presets and overrides a queued job can use.

A job is data, never a command. It names a preset by (name, version, sha256); the
agent runs it only if that triple matches a preset shipped in the INSTALLED CLI
package (cli/presets/<name>.json, validated against the shared schema
cli/schemas/llm-speed-preset-1.schema.json; or the built-in `selftest`)
and the name is in the reviewed list below. Overrides can only narrow the preset.
The agent builds argv itself; nothing from the server reaches argv except values
that were checked against the preset's own data.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from functools import lru_cache
from typing import Any

from .. import agent_job, preset_schema
from .settings import Settings

# Reviewed presets. Adding one is a code change + review + CLI release (phase 1).
REVIEWED_PRESETS = ("flint-40b", "selftest")

# `selftest`: no model, no download, no upload, no fit rows. Lets the queue operator check a
# newly enrolled machine end to end in seconds. Its identity is the sha256 of this dict's
# canonical bytes (print it with `llm-speed agent presets`).
SELFTEST_PRESET: dict[str, Any] = {
    "schema": "llm-speed-selftest/1",
    "preset": "selftest",
    "version": 1,
    "title": "Agent self-test (no model, no upload)",
    "overrides": {"sleep_s": [0, 60], "outcome": ["succeeded", "failed"]},
}

# Fields of the claimed `job` object (ops/AGENT_QUEUE_API.md section 4), plus the
# few enqueue fields that may be echoed. Anything else -> rejected (contract drift shows up).
JOB_SCHEMA = "llm-speed-job/1"
JOB_FIELDS = {
    "job_id",
    "assignment_id",
    "preset",
    "overrides",
    "launch_id",
    "upload",
    "strict_anon",
    "require_approval",
    "max_runtime_min",
    "attempt",
    "created_at",
    "expires_at",
    "not_before",
    "cost_class",  # how a job is classified is the server's business: accepted, ignored
    "priority",
    "schema",
    # Per-assignment upload nonce (security retest R-9). A queue that sends it to a build
    # that doesn't use it must not fail the claim (Security Desk R-36): accepted, ignored.
    # This build echoes a well-formed one inside each signed upload of the job
    # (cli/agent_job.py, job_context); a missing or malformed one is ignored, never a
    # rejection.
    "upload_nonce",
}
# The server fails an assignment after 3 lapsed leases (queue API spec section 5), so a claim
# never carries attempt > 3; anything else is contract drift -> rejected.
MAX_ATTEMPTS = 3
WORKLOAD_RE = re.compile(r"^[a-z0-9\-]{1,40}$")

# override -> (python type, bench flag). Values are validated against the preset first.
OVERRIDES: dict[str, tuple[type, str]] = {
    "include_control": (bool, "--with-control"),
    "include_fallback": (bool, "--include-fallback"),
    "include_offload_probe": (bool, "--no-offload-probe"),  # emitted only when False
    "entries": (list, "--only"),
    "contexts": (list, "--contexts"),
    "max_ctx": (int, "--max-ctx"),
    "workloads": (list, "--preset-workloads"),
}
# Flags a job can never cause, whatever happens above (belt and braces).
FORBIDDEN_FLAGS = {
    "--yes",
    "--force",
    "--api-base",
    "--api-key",
    "--llama-server",
    "--models-dir",
    "--resume",
}


class JobRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason[:120]


@lru_cache(maxsize=1)
def bench_flags() -> frozenset[str]:
    """Option strings `llm-speed bench` accepts in this install (so a job never sends a
    flag the installed preset command doesn't understand)."""
    from ..__main__ import _build_parser

    parser = _build_parser()
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        sub = action.choices.get("bench") if hasattr(action, "choices") else None
        if sub is not None:
            return frozenset(o for a in sub._actions for o in a.option_strings)
    return frozenset()


def _parse_time(v: Any, field: str) -> dt.datetime | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise JobRejected(f"bad_field:{field}")
    try:
        t = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise JobRejected(f"bad_field:{field}") from None
    if t.tzinfo is None:
        raise JobRejected(f"bad_field:{field}")
    return t


def selftest_identity() -> dict[str, Any]:
    return {
        "name": "selftest",
        "version": SELFTEST_PRESET["version"],
        "sha256": preset_schema.preset_sha256(SELFTEST_PRESET),
    }


def advertised(settings: Settings) -> list[dict[str, Any]]:
    """[{name, version, sha256}] this machine can run: reviewed + installed + valid + allowed locally."""
    names = [
        n
        for n in REVIEWED_PRESETS
        if settings.allow_presets is None or n in settings.allow_presets
    ]
    out = preset_schema.packaged_identities([n for n in names if n != "selftest"])
    if "selftest" in names:
        out.append(selftest_identity())
    return out


def _ctx_values(lst: list) -> set[int]:
    return {c["ctx"] if isinstance(c, dict) else c for c in lst or []}


def _preset_contexts(p: dict) -> set[int]:
    out = _ctx_values(p.get("contexts", []))
    for m in p.get("models", {}).values():
        out |= _ctx_values(m.get("contexts", []))
    for prof in (p.get("profiles") or {}).values():
        for lst in (prof.get("contexts") or {}).values():
            out |= _ctx_values(lst)
    return out


def _preset_workloads(p: dict) -> set[str]:
    out: set[str] = set()
    for v in (p.get("workloads_by_ctx") or {}).values():
        out.update(v.split(","))
    for m in p.get("models", {}).values():
        for c in m.get("contexts", []):
            if isinstance(c, dict) and c.get("workloads"):
                out.update(c["workloads"].split(","))
    return out or {"chat-short"}


def _validate_selftest(ident: dict, ov: dict) -> tuple[dict, dict[str, Any]]:
    mine = selftest_identity()
    if (ident.get("version"), ident.get("sha256")) != (mine["version"], mine["sha256"]):
        raise JobRejected("preset_not_allowed")
    out: dict[str, Any] = {}
    for k, v in ov.items():
        ok_sleep = (
            k == "sleep_s"
            and isinstance(v, int)
            and not isinstance(v, bool)
            and 0 <= v <= 60
        )
        if ok_sleep or (k == "outcome" and v in ("succeeded", "failed")):
            out[k] = v
        elif k in ("sleep_s", "outcome"):
            raise JobRejected(f"bad_override:{k}")
        else:
            raise JobRejected(f"override_not_allowed:{k}")
    return dict(SELFTEST_PRESET), out


def validate_job(
    job: Any,
    settings: Settings,
    *,
    now: dt.datetime | None = None,
    preset_loader=preset_schema.packaged_preset,
    flags: frozenset[str] | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Return (preset, overrides) or raise JobRejected(reason). Reasons use the queue API's codes."""
    if not isinstance(job, dict):
        raise JobRejected("invalid_job")
    extra = set(job) - JOB_FIELDS
    if extra:
        raise JobRejected(f"unknown_field:{min(extra)}")
    if job.get("schema", JOB_SCHEMA) != JOB_SCHEMA:
        raise JobRejected("unsupported_job_schema")
    if job.get("strict_anon", True) is not True:
        raise JobRejected("strict_anon_required")
    now = now or dt.datetime.now(dt.timezone.utc)
    exp = _parse_time(job.get("expires_at"), "expires_at")
    if exp is not None and exp <= now:
        raise JobRejected("expired")
    nb = _parse_time(job.get("not_before"), "not_before")
    if nb is not None and nb > now:
        raise JobRejected(
            "not_before"
        )  # the runner releases (not an attempt) instead of reporting
    for b in ("upload", "require_approval"):
        if not isinstance(job.get(b, False), bool):
            raise JobRejected(f"bad_field:{b}")
    att = job.get("attempt", 1)
    if (
        not isinstance(att, int)
        or isinstance(att, bool)
        or not 1 <= att <= MAX_ATTEMPTS
    ):
        raise JobRejected("bad_field:attempt")
    mr = job.get("max_runtime_min")
    if mr is not None and (
        not isinstance(mr, int) or isinstance(mr, bool) or not 1 <= mr <= 720
    ):
        raise JobRejected("bad_field:max_runtime_min")

    ident = job.get("preset")
    if not isinstance(ident, dict) or set(ident) - {"name", "version", "sha256"}:
        raise JobRejected("bad_preset")
    name, version, sha = ident.get("name"), ident.get("version"), ident.get("sha256")
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9.\-]{0,62}", name
    ):
        raise JobRejected("bad_preset")
    if name not in REVIEWED_PRESETS:
        raise JobRejected("preset_not_allowed")
    if settings.allow_presets is not None and name not in settings.allow_presets:
        raise JobRejected("preset_not_allowed_on_this_machine")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(sha, str)
    ):
        raise JobRejected("bad_preset")
    ov = job.get("overrides") or {}
    if not isinstance(ov, dict):
        raise JobRejected("bad_overrides")
    if name == "selftest":
        return _validate_selftest(ident, ov)
    preset = preset_loader(name)
    if preset is None:
        raise JobRejected("preset_not_installed")
    try:
        preset_schema.validate_preset(preset)
    except preset_schema.PresetInvalid:
        raise JobRejected("installed_preset_invalid") from None
    mine = preset_schema.identity(preset)
    if (mine["version"], mine["sha256"]) != (version, sha):
        raise JobRejected("preset_not_allowed")  # same name, different bytes/version

    flags = bench_flags() if flags is None else flags
    if "--preset" not in flags:
        raise JobRejected("preset_command_not_available")
    out: dict[str, Any] = {}
    models = set(preset.get("models", {}))
    ctxs = _preset_contexts(preset)
    wls = _preset_workloads(preset)
    for k, v in ov.items():
        if k not in OVERRIDES:
            raise JobRejected(f"override_not_allowed:{k}")
        typ, flag = OVERRIDES[k]
        if typ is bool and not isinstance(v, bool):
            raise JobRejected(f"bad_override:{k}")
        if typ is int and (not isinstance(v, int) or isinstance(v, bool)):
            raise JobRejected(f"bad_override:{k}")
        if typ is list:
            if (
                not isinstance(v, list)
                or not v
                or len(v) > 64
                or len(set(map(repr, v))) != len(v)
            ):
                raise JobRejected(f"bad_override:{k}")
            if k == "entries" and not all(
                isinstance(x, str) and x in models for x in v
            ):
                raise JobRejected(
                    "override_not_allowed:entries"
                )  # would widen / unknown entry
            if k == "contexts" and not all(
                isinstance(x, int) and not isinstance(x, bool) and x in ctxs for x in v
            ):
                raise JobRejected("override_not_allowed:contexts")
            if k == "workloads" and not all(
                isinstance(x, str) and WORKLOAD_RE.match(x) and x in wls for x in v
            ):
                raise JobRejected("override_not_allowed:workloads")
        if k == "max_ctx" and not (1024 <= v <= max(ctxs)):
            raise JobRejected("override_not_allowed:max_ctx")
        noop = (k == "include_offload_probe" and v is True) or (
            typ is bool and k != "include_offload_probe" and v is False
        )
        if not noop and flag not in flags:
            raise JobRejected(f"override_not_supported:{k}")
        out[k] = v
    return preset, out


def job_context(job: dict, api_base: str) -> dict[str, str] | None:
    """{assignment_id, upload_nonce, api_base} for a validated job's uploads, or None when
    the claim carried no usable nonce (older server, a pre-release build, or a malformed
    value with a malformed assignment_id): the job then runs exactly as before. Never a
    rejection (Security Desk R-36)."""
    aid, nonce = job.get("assignment_id"), job.get("upload_nonce")
    if not agent_job.valid(aid, nonce) or not api_base:
        return None
    return {"assignment_id": aid, "upload_nonce": nonce, "api_base": api_base}


def build_argv(
    name: str, overrides: dict[str, Any], settings: Settings, *, upload: bool = True
) -> list[str]:
    """argv for the child: the same `llm-speed bench --preset` a person would paste."""
    if name not in REVIEWED_PRESETS:
        raise JobRejected("preset_not_allowlisted")
    argv = [sys.executable, "-m", "cli", "bench", "--preset", name]
    for k, v in overrides.items():
        _, flag = OVERRIDES[k]
        assert flag not in FORBIDDEN_FLAGS
        if k == "include_offload_probe":
            if v is False:
                argv.append(flag)
        elif isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, list):
            argv += [flag, ",".join(str(x) for x in v)]
        else:
            argv += [flag, str(int(v))]
    # Local-only settings.
    if settings.models_dir:
        argv += ["--models-dir", settings.models_dir]
    if settings.llama_server:
        argv += ["--llama-server", settings.llama_server]
    if not (upload and settings.upload):
        argv.append("--no-upload")
    argv.append("--strict-anon")  # agent uploads are always strict-anon
    return argv
