"""`llm-speed bench --lab-session`: attributed provenance for Launch Lab sessions (rev 3.3, C4).

A Launch Lab session runs one bench per one-time run token. The token reaches the
CLI only through an inherited pipe. `LLM_SPEED_LAB_FD` holds that pipe's file
descriptor number and nothing else.

What this module does, in order:

1. `take_lab_credentials()` runs in `cli.__main__.main` right after argument parsing,
   only when `--lab-session` was passed, and before the background updater or any other
   child process, driver or llama-server can start. It removes `LLM_SPEED_LAB_FD` from
   the environment, checks that the value is a plain integer naming a pipe, reads one
   line from it, closes the fd, and keeps the per-run credentials in memory only.
   Without `--lab-session` nothing here runs: the variable and the pipe are not touched.
2. The bench itself runs in strict-anon mode (ephemeral key, no `fingerprint_hash`, no
   storage block, no llm-speed-cli User-Agent, no API key, no credit). The local save is the ordinary
   strict-anon JWS with no lab field inside it, plus a top-level `"lab_session": true`
   marker so `submit` and `bench --resume` in 0.0.24 refuse it (it is not
   re-uploaded by this CLI).
3. `upload_lab_report()` signs the strict-anon payload plus the claim
   `lab_session: {sid, ath, seq}` with a fresh ephemeral key, in memory, runs
   `saved_run_is_identified` on exactly that payload and token, and POSTs
   `{"jws": ..., "lab_attestation": "lla_<64hex>"}` to the issuing host only, with no
   `X-LLM-Speed-Anon` header. That JWS and the token are never written to disk,
   logged, or printed.

Pipe format (one line, UTF-8 JSON, at most 1024 bytes, read until newline or EOF):

    {"sid": "lss_<16hex>", "seq": <1..16>, "lab_attestation": "lla_<64hex>"}

The key names are the rev 3.3 names: `sid` and `seq` from the `lab_session` claim and
`lab_attestation` for the envelope field that carries the raw token.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import re
import stat
from typing import Any

from . import api_host

LAB_FD_ENV = "LLM_SPEED_LAB_FD"
LAB_FLAG = "--lab-session"

# The host that mints lab sessions (POST /v1/lab/sessions) and so the only host a lab
# run token may be sent to. Hard-coded, never read from $LLM_SPEED_API or --api-base.
LAB_ISSUING_API_BASE = api_host.CANONICAL_API_BASE

# rev 3.3 C1/C2: sid "lss_<16hex>", token "lla_<64hex>", seq 1..max_runs (cap 16).
_SID_RE = re.compile(r"lss_[0-9a-f]{16}", re.ASCII)
_TOKEN_RE = re.compile(r"lla_[0-9a-f]{64}", re.ASCII)
_FD_RE = re.compile(r"[0-9]{1,9}", re.ASCII)
MAX_RUNS = 16
_MAX_PIPE_BYTES = 1024
_PIPE_TIMEOUT_S = 30.0

# Bench flags that cannot be combined with --lab-session. One token is one upload of
# one fresh run, so nothing that skips the upload, re-uploads a saved run, uploads
# several runs, prints the signed body, or adds identity is allowed.
INCOMPATIBLE_FLAGS: tuple[tuple[str, str], ...] = (
    ("resume", "--resume"),
    ("preset", "--preset"),
    ("no_upload", "--no-upload"),
    ("upload_failures", "--upload-failures"),
    ("json", "--json"),
    ("dry_run", "--dry-run"),
    ("print_payload", "--print-payload"),
    ("anon", "--anon"),
    ("api_key", "--api-key"),
    ("credit", "--credit"),
    ("share_storage_info", "--share-storage-info"),
)


class LabSessionError(ValueError):
    """A lab-session problem. The message never contains a token value."""


@dataclasses.dataclass(frozen=True)
class LabCredentials:
    """The per-run lab credentials, in memory only. `repr` never shows the token."""

    sid: str
    seq: int
    lab_attestation: str = dataclasses.field(repr=False)

    def claim(self) -> dict[str, Any]:
        """The signed claim `lab_session: {sid, ath, seq}` (keys in sorted order)."""
        return {"ath": ath_of(self.lab_attestation), "seq": self.seq, "sid": self.sid}


def ath_of(token: str) -> str:
    """b64url(sha256(token)), unpadded (43 characters)."""
    digest = hashlib.sha256(token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def incompatible_flags(args) -> list[str]:
    """Flags passed together with --lab-session that it refuses."""
    return [flag for attr, flag in INCOMPATIBLE_FLAGS if getattr(args, attr, None)]


def _parse_fd(raw: str | None) -> int:
    if raw is None or raw == "":
        raise LabSessionError(
            f"{LAB_FLAG} needs {LAB_FD_ENV} set to the number of an inherited pipe "
            "file descriptor; it is not set"
        )
    if raw.startswith("lla_") or _TOKEN_RE.search(raw):
        raise LabSessionError(
            f"{LAB_FD_ENV} must hold only a pipe file descriptor number, never a token; "
            "refusing it (the value is not shown)"
        )
    if not _FD_RE.fullmatch(raw):
        raise LabSessionError(
            f"{LAB_FD_ENV} must be a plain decimal file descriptor number (for example "
            "3); refusing a non-integer value (the value is not shown)"
        )
    fd = int(raw)
    if fd < 3:
        raise LabSessionError(
            f"{LAB_FD_ENV}={fd} names stdin, stdout or stderr; it must be an inherited "
            "pipe file descriptor (3 or higher)"
        )
    return fd


def _require_pipe(fd: int) -> None:
    try:
        st = os.fstat(fd)
    except OSError:
        raise LabSessionError(
            f"{LAB_FD_ENV}={fd} is not an open file descriptor; it must be an "
            "inherited pipe"
        ) from None
    if not stat.S_ISFIFO(st.st_mode):
        raise LabSessionError(
            f"{LAB_FD_ENV}={fd} is not a pipe; refusing to read lab credentials from "
            "a regular file, terminal, socket or device"
        )


def _read_line(fd: int) -> bytes:
    """Up to one line (or EOF) from the pipe, at most `_MAX_PIPE_BYTES`."""
    buf = b""
    while b"\n" not in buf:
        if os.name == "posix":
            import select

            ready, _, _ = select.select([fd], [], [], _PIPE_TIMEOUT_S)
            if not ready:
                raise LabSessionError(
                    f"timed out after {int(_PIPE_TIMEOUT_S)} s waiting for lab "
                    f"credentials on {LAB_FD_ENV}={fd}"
                )
        chunk = os.read(fd, _MAX_PIPE_BYTES + 1 - len(buf))
        if not chunk:
            break
        buf += chunk
        if len(buf) > _MAX_PIPE_BYTES:
            raise LabSessionError(
                f"lab credentials on {LAB_FD_ENV}={fd} are longer than "
                f"{_MAX_PIPE_BYTES} bytes; refusing them"
            )
    return buf.split(b"\n", 1)[0]


def parse_credentials(line: bytes) -> LabCredentials:
    """Strict parse of the pipe line. Error messages never echo the input."""
    bad = LabSessionError(
        "lab credentials on the pipe are not "
        '{"sid": "lss_<16hex>", "seq": 1..16, "lab_attestation": "lla_<64hex>"} '
        "(the value is not shown)"
    )
    try:
        obj = json.loads(line.decode("utf-8").strip())
    except (UnicodeDecodeError, ValueError):
        raise bad from None
    if not isinstance(obj, dict) or sorted(obj) != ["lab_attestation", "seq", "sid"]:
        raise bad
    sid, seq, token = obj["sid"], obj["seq"], obj["lab_attestation"]
    if not isinstance(sid, str) or not _SID_RE.fullmatch(sid):
        raise bad
    if isinstance(seq, bool) or not isinstance(seq, int) or not 1 <= seq <= MAX_RUNS:
        raise bad
    if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
        raise bad
    return LabCredentials(sid=sid, seq=seq, lab_attestation=token)


def take_lab_credentials(environ: Any = None) -> LabCredentials:
    """Read the per-run credentials once, then close the pipe and clear the variable.

    Call only when --lab-session was passed, before any child process starts. The
    variable is removed from the environment first, whatever happens next. The fd is
    closed once it has been confirmed to be a pipe, including when reading or parsing
    fails, so a child never inherits it.
    """
    env = os.environ if environ is None else environ
    raw = env.pop(LAB_FD_ENV, None)
    fd = _parse_fd(raw)
    _require_pipe(fd)
    try:
        line = _read_line(fd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    return parse_credentials(line)


def require_issuing_host(api_base: str) -> None:
    """A lab run token goes only to the host that issued it."""
    if not api_host.same_api_base(api_base, LAB_ISSUING_API_BASE):
        raise LabSessionError(
            f"{LAB_FLAG} uploads only to the issuing host "
            f"{api_host.display_host(LAB_ISSUING_API_BASE)}; refusing "
            f"{api_host.display_host(api_base)} (check --api-base and LLM_SPEED_API)"
        )
    api_host.check_api_base(api_base)


# Payload keys a lab upload must not carry (rev 3.3 Worker ingest: strict-anon-shaped).
_FORBIDDEN_TOP = ("identity", "attribution", "agent_job", "credit")
_FORBIDDEN_FP = ("fingerprint_hash", "storage")


def payload_problem(payload: Any) -> str | None:
    """Why this payload is not strict-anon-shaped, or None. Key presence counts."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    for key in _FORBIDDEN_TOP:
        if key in payload:
            return f"payload carries `{key}`"
    fp = payload.get("fingerprint")
    if isinstance(fp, dict):
        for key in _FORBIDDEN_FP:
            if key in fp:
                return f"payload fingerprint carries `{key}`"
    return None


@dataclasses.dataclass(frozen=True)
class LabUploadResult:
    """What the controller needs for close: no token, no JWS."""

    url: str
    run_id: str | None
    seq: int
    public_key: str  # standard base64 of the raw Ed25519 key, as the Worker stores it
    jws_sha256: str  # sha256 hex of the compact JWS that was posted


def upload_lab_report(
    report,
    creds: LabCredentials,
    *,
    api_base: str,
    timeout: float = 30.0,
) -> LabUploadResult:
    """Sign the strict-anon payload with the lab claim and POST it to the issuing host.

    Every check runs before the POST and raises `UploadError` / `LabSessionError`:
    issuing host, strict-anon shape (an injected `fingerprint_hash` or `storage` is
    refused), and `saved_run_is_identified`, which runs unconditionally. One attempt
    only: the token is single use, and a retry after a lost response would be a 409.
    """
    from . import signing, upload

    require_issuing_host(api_base)
    payload = signing.build_uploadable_report(report, strict_anon=True)
    problem = payload_problem(payload)
    if problem:
        raise LabSessionError(
            f"refusing the lab upload before POST: {problem} (a lab upload is "
            "strict-anon shaped)"
        )
    if "lab_session" in payload:
        raise LabSessionError("refusing the lab upload before POST: claim already set")
    payload["lab_session"] = creds.claim()
    signing.assert_no_long_strings(payload)
    token = signing.sign_payload_jws(payload, strict_anon=True)
    # rev 3.3 C4: unconditional, on exactly what is about to be posted.
    if upload.saved_run_is_identified(payload, token):
        raise LabSessionError(
            "refusing the lab upload before POST: the run is identified (fingerprint "
            "hash, storage block, or this machine's long-term key)"
        )
    jwk = signing.header_jwk(token) or {}
    x = jwk.get("x")
    if not isinstance(x, str):
        raise LabSessionError("refusing the lab upload before POST: no public key")
    public_key = base64.b64encode(
        base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
    ).decode("ascii")
    jws_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    body = {"jws": token, "lab_attestation": creds.lab_attestation}
    # Exactly the strict-anon headers: no llm-speed-cli User-Agent (httpx's
    # default is sent), no Authorization, and no X-LLM-Speed-Anon (rev 3.3 C8 §3).
    headers = {"Content-Type": "application/json"}
    url = f"{api_base.rstrip('/')}/v1/results"
    data = upload._post_with_retries(
        url, body, headers=headers, timeout=timeout, max_retries=1
    )
    rid = data.get("id") if isinstance(data, dict) else None
    return LabUploadResult(
        url=upload._result_url_from_response(data),
        run_id=rid if isinstance(rid, str) else None,
        seq=creds.seq,
        public_key=public_key,
        jws_sha256=jws_sha256,
    )


__all__ = [
    "INCOMPATIBLE_FLAGS",
    "LAB_FD_ENV",
    "LAB_FLAG",
    "LAB_ISSUING_API_BASE",
    "LabCredentials",
    "LabSessionError",
    "LabUploadResult",
    "ath_of",
    "incompatible_flags",
    "parse_credentials",
    "payload_problem",
    "require_issuing_host",
    "take_lab_credentials",
    "upload_lab_report",
]
