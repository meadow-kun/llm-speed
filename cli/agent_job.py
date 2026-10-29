"""Per-assignment upload nonce for `llm-speed agent` jobs (security retest R-9 / F5).

The queue's claim gives each assignment an `upload_nonce` (`lsn_` + 32 lowercase hex).
Every result upload made FOR THAT JOB carries, inside the signed JWS payload,

    "agent_job": {"assignment_id": "asg_<16 hex>", "upload_nonce": "lsn_<32 hex>"}

and nothing else under that key. The Worker records the run as the job's own only when
the nonce matches (docs/agent_queue_nonce_enrolment.md). Normal `llm-speed bench`,
`--resume` and hand-made uploads never carry it.

Handling (spec: treat it like the lease id):
- memory only: the agent hands it to its preset child through an inherited pipe (never
  argv, never an environment value, never a file). The child reads it once at start,
  closes the pipe and drops the fd number from its environment, so the per-cell
  `llm-speed bench` grandchildren and llama-server never see it;
- never logged, printed, or written to the job log, result JSON, local saves or history:
  the saved per-cell JWS stays nonce-free and is re-signed in memory just before POST;
- sent only to the queue that issued it (the agent's enrolled API base); an upload aimed
  at any other host goes without it (N-5 spirit).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# The agent sets this in its preset child's environment to the number of an inherited
# read-only pipe fd. It holds a number, never the nonce. Names under LLM_SPEED_AGENT_ are
# stripped from the agent's own environment before the child's is built.
FD_ENV = "LLM_SPEED_AGENT_JOB_FD"
ASSIGNMENT_RE = re.compile(r"^asg_[0-9a-f]{16}$")
NONCE_RE = re.compile(r"^lsn_[0-9a-f]{32}$")
_MAX_BYTES = 1024

# Security Desk R-36: the queue sends `upload_nonce` only to a claim that declares it can use
# it, as `"features": ["upload_nonce"]` in the claim request body (final since the queue's
# R-36 change: a list of at most 16 strings matching ^[a-z0-9_]{1,32}$, unknown values
# ignored, a malformed list refused with 400). Declared by cli/agent/transport.py claim()
# because this build handles the nonce (this module).
CLAIM_FEATURES_KEY = "features"
UPLOAD_NONCE_FEATURE = "upload_nonce"

_current: dict[str, str] | None = None  # {"assignment_id", "upload_nonce", "api_base"}


def valid(assignment_id: Any, upload_nonce: Any) -> bool:
    return (
        isinstance(assignment_id, str)
        and isinstance(upload_nonce, str)
        and bool(ASSIGNMENT_RE.fullmatch(assignment_id))
        and bool(NONCE_RE.fullmatch(upload_nonce))
    )


def payload_field(ctx: dict[str, str] | None) -> dict[str, str] | None:
    """The exact `agent_job` value for the signed payload: two keys, both strings."""
    if not ctx or not valid(ctx.get("assignment_id"), ctx.get("upload_nonce")):
        return None
    return {"assignment_id": ctx["assignment_id"], "upload_nonce": ctx["upload_nonce"]}


def encode(ctx: dict[str, str]) -> bytes:
    return json.dumps(
        {
            "assignment_id": ctx["assignment_id"],
            "upload_nonce": ctx["upload_nonce"],
            "api_base": ctx["api_base"],
        },
        separators=(",", ":"),
    ).encode("ascii")


def take_from_env(environ: dict[str, str] | None = None) -> dict[str, str] | None:
    """Child side: read the job context once from the inherited pipe, close it, remove
    the fd number from the environment. Anything malformed -> None (upload without it,
    exactly as before the nonce existed)."""
    global _current
    env = os.environ if environ is None else environ
    raw_fd = env.pop(FD_ENV, None)
    if raw_fd is None:
        return _current
    try:
        fd = int(raw_fd)
    except ValueError:
        return None
    if fd < 3:  # never stdin/stdout/stderr
        return None
    data = b""
    try:
        while len(data) <= _MAX_BYTES:
            chunk = os.read(fd, _MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    except OSError:
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if len(data) > _MAX_BYTES:
        return None
    try:
        obj = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != {"assignment_id", "upload_nonce", "api_base"}
        or not valid(obj.get("assignment_id"), obj.get("upload_nonce"))
        or not isinstance(obj.get("api_base"), str)
        or not obj["api_base"]
    ):
        return None
    _current = {k: obj[k] for k in ("assignment_id", "upload_nonce", "api_base")}
    return _current


def current() -> dict[str, str] | None:
    return _current


def for_upload(api_base: str, *, strict_anon: bool) -> dict[str, str] | None:
    """The `agent_job` field for an upload to `api_base`, or None. Only inside an agent
    preset child (context read from the pipe), only for strict-anon uploads (agent
    uploads always are), and only to the queue that issued the nonce."""
    from .api_host import same_api_base

    ctx = _current
    if not ctx or not strict_anon or not same_api_base(api_base, ctx["api_base"]):
        return None
    return payload_field(ctx)


def clear() -> None:
    global _current
    _current = None
