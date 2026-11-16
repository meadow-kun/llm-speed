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
   storage block, no llm-speed User-Agent (httpx sends its generic default), no API key,
   no credit).
3. `lab_pins()` adds the three pins the Worker's `pinsMatch` checks against the
   session row minted for this session (rev 3.3 C8 §6 names; R-195): top-level
   `llama_build` (the llama.cpp build number this run used), top-level `gpu_class`
   (the site's hardware slug of the signed fingerprint GPU name) and, in every
   `results[].model.digest`, the full-file 64-hex sha256 of the model file (the value
   minted into `lab_sessions.model_sha256s[].sha256`). Without `--lab-session` none of
   this runs and `model.digest` stays the 16-hex prefix digest.
4. The local save of a lab run is NOT a JWS (R-194). It is the unsigned payload
   (pins included, no `lab_session` claim, no token), so no CLI of any version can
   post it: `submit` and `bench --resume` need a verifiable `jws` and there is none.
   This does not rest on any unsigned marker.
5. `upload_lab_report()` signs the strict-anon payload plus the pins and the claim
   `lab_session: {sid, ath, seq}` with a fresh ephemeral key, in memory, runs
   `saved_run_is_identified` on exactly that payload and token, reports the
   close-step values (seq, public_key, jws_sha256) BEFORE the POST (R-196), and POSTs
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
from collections.abc import Callable
from pathlib import Path
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


# --------------------------------------------------------------------- pins (R-195)
#
# The Worker (T101a, worker/src/lab-sessions.ts pinsMatch) compares, with strict
# equality and failing closed:
#   payload.cli_version                    == lab_sessions.cli_version
#   payload.llama_build (first of 4 names) == lab_sessions.llama_build (as strings)
#   payload.gpu_class   (first of 2 names) == lab_sessions.gpu_class
#   every results[].model.{digest, quant}  == some lab_sessions.model_sha256s[].{sha256, quant}
# The CLI uses only the rev 3.3 names (C8 §6: `llama_build`, `gpu_class`,
# `model_sha256s`) and the existing `model.digest` field, at the placement pinsMatch
# reads first. It never sends `exact_llama_build` or a fingerprint-level copy.

_GPU_CLASS_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}", re.ASCII)  # Worker GPU_RE
_QUANT_RE = re.compile(r"[A-Za-z0-9._+-]{1,32}", re.ASCII)  # Worker QUANT_RE
_LLAMA_BUILD_RE = re.compile(r"b([0-9]{1,9})", re.ASCII)


@dataclasses.dataclass(frozen=True)
class LabPins:
    """The lab-only pins, computed once per run. Not secrets."""

    llama_build: int
    gpu_class: str
    model_sha256s: tuple[str, ...]  # full-file sha256, one per report result, in order


def gpu_class_of(gpu_names: list[str]) -> str:
    """`gpu_class` from the signed fingerprint GPU names.

    The site's hardware slug (`canonicalHardwareSlug` in web/lib/seo.ts, mirrored in
    cli/mcp/slugs.py) of the one GPU model on the machine: "RTX 4090" -> "rtx-4090",
    "RTX 3090 Ti" -> "rtx-3090-ti", the same keys `tools/cloud_bench.py` uses for
    hardware. No GPU, or GPUs of different models, is refused."""
    from .mcp.slugs import canonical_hardware_slug

    names = [n.strip() for n in gpu_names if isinstance(n, str) and n.strip()]
    if not names or len(names) != len(gpu_names) or len(set(names)) != 1:
        raise LabSessionError(
            "refusing the lab upload before POST: gpu_class needs exactly one GPU "
            "model in the fingerprint"
        )
    slug = canonical_hardware_slug(names[0])
    if not _GPU_CLASS_RE.fullmatch(slug):
        raise LabSessionError(
            "refusing the lab upload before POST: the GPU name does not give a valid "
            "gpu_class"
        )
    return slug


def llama_build_of(results) -> int:
    """`llama_build`: the one llama.cpp build number every result ran on.

    From the `llama_cpp_build` label the llama.cpp driver records for each run
    (`b<N>`, from llama-server /props or --version). Sent as the integer N, the form
    preset `exact_llama_build` pins use. Missing or disagreeing builds are refused."""
    builds: set[int] = set()
    for r in results:
        if getattr(r, "backend", None) != "llama.cpp":
            raise LabSessionError(
                "refusing the lab upload before POST: lab sessions pin a llama.cpp "
                "build; every result must be a llama.cpp run"
            )
        label = (getattr(r, "extras", None) or {}).get("llama_cpp_build")
        m = _LLAMA_BUILD_RE.fullmatch(label) if isinstance(label, str) else None
        if not m:
            raise LabSessionError(
                "refusing the lab upload before POST: llama-server did not report its "
                "build number, so llama_build cannot be pinned"
            )
        builds.add(int(m.group(1)))
    if len(builds) != 1:
        raise LabSessionError(
            "refusing the lab upload before POST: the results do not share one "
            "llama.cpp build"
        )
    return builds.pop()


def _model_file(model) -> str:
    from .gguf_split import shard_set_for

    ident = getattr(model, "identifier", None)
    if not isinstance(ident, str) or not ident:
        raise LabSessionError(
            "refusing the lab upload before POST: the model has no local file to pin"
        )
    if shard_set_for(Path(ident)) is not None:
        # rev 3.3 pins one full-file sha256 per model; a split GGUF has no single
        # file. Not guessed (first shard only would not pin the model). Open item.
        raise LabSessionError(
            "refusing the lab upload before POST: a split (multi-file) GGUF has no "
            "single full-file sha256 to pin"
        )
    return ident


def lab_pins(report) -> LabPins:
    """Compute the pins for this run, or raise `LabSessionError` (nothing is posted).

    Hashes each distinct model file once (streamed, cached by path, size and mtime;
    the duration is logged at info level, see cli/model_hash.py)."""
    from . import model_hash
    from .fingerprint import to_uploadable_dict

    results = list(getattr(report, "results", None) or [])
    if not results:
        raise LabSessionError("refusing the lab upload before POST: no results")
    fp = to_uploadable_dict(report.fingerprint, strict_anon=True)
    gpu = gpu_class_of([g.get("name") for g in fp.get("gpus") or []])
    build = llama_build_of(results)
    by_file: dict[str, str] = {}
    digests: list[str] = []
    for r in results:
        quant = getattr(r.model, "quant", None)
        if not isinstance(quant, str) or not _QUANT_RE.fullmatch(quant):
            raise LabSessionError(
                "refusing the lab upload before POST: the model quant is unknown, so "
                "the model pin cannot match"
            )
        path = _model_file(r.model)
        if path not in by_file:
            try:
                fh = model_hash.full_sha256(path)
            except model_hash.ModelHashError as exc:
                raise LabSessionError(
                    f"refusing the lab upload before POST: {exc}"
                ) from None
            by_file[path] = fh.sha256
        digests.append(by_file[path])
    return LabPins(llama_build=build, gpu_class=gpu, model_sha256s=tuple(digests))


def build_lab_payload(
    report, pins: LabPins, *, include_raw_timings: bool = False
) -> dict[str, Any]:
    """The strict-anon payload plus the pins. No `lab_session` claim, no token."""
    from . import signing

    payload = signing.build_uploadable_report(
        report, strict_anon=True, include_raw_timings=include_raw_timings
    )
    problem = payload_problem(payload)
    if problem:
        raise LabSessionError(
            f"refusing the lab upload before POST: {problem} (a lab upload is "
            "strict-anon shaped)"
        )
    for key in ("lab_session", "llama_build", "gpu_class"):
        if key in payload:
            raise LabSessionError(
                f"refusing the lab upload before POST: `{key}` already set"
            )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(pins.model_sha256s):
        raise LabSessionError(
            "refusing the lab upload before POST: results do not line up with the pins"
        )
    for row, digest in zip(rows, pins.model_sha256s, strict=True):
        model = row.get("model")
        if not isinstance(model, dict):
            raise LabSessionError("refusing the lab upload before POST: no model")
        model["digest"] = digest
    payload["llama_build"] = pins.llama_build
    payload["gpu_class"] = pins.gpu_class
    return payload


@dataclasses.dataclass(frozen=True)
class LabSigned:
    """The close-step values, known before the POST (R-196). No token, no JWS."""

    seq: int
    public_key: str  # standard base64 of the raw Ed25519 key, as the Worker stores it
    jws_sha256: str  # sha256 hex of the compact JWS that is about to be posted


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
    pins: LabPins | None = None,
    on_signed: Callable[[LabSigned], None] | None = None,
    timeout: float = 30.0,
) -> LabUploadResult:
    """Sign the strict-anon payload with the pins and the lab claim and POST it to
    the issuing host.

    Every check runs before the POST and raises `UploadError` / `LabSessionError`:
    issuing host, pins (computed here when not given), strict-anon shape (an injected
    `fingerprint_hash` or `storage` is refused), and `saved_run_is_identified`, which
    runs unconditionally. `on_signed` is called with (seq, public_key, jws_sha256)
    after every check and immediately before the POST, so the controller has the
    close-step values even when the response is lost (R-196). One attempt only: the
    token is single use, and a retry after a lost response would be a 409.
    """
    from . import signing, upload

    require_issuing_host(api_base)
    if pins is None:
        pins = lab_pins(report)
    payload = build_lab_payload(report, pins)
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
    # Exactly the strict-anon headers: no llm-speed User-Agent (httpx's
    # default is sent), no Authorization, and no X-LLM-Speed-Anon (rev 3.3 C8 §3).
    headers = {"Content-Type": "application/json"}
    url = f"{api_base.rstrip('/')}/v1/results"
    if on_signed is not None:
        on_signed(
            LabSigned(seq=creds.seq, public_key=public_key, jws_sha256=jws_sha256)
        )
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
    "LabPins",
    "LabSessionError",
    "LabSigned",
    "LabUploadResult",
    "ath_of",
    "build_lab_payload",
    "gpu_class_of",
    "incompatible_flags",
    "lab_pins",
    "llama_build_of",
    "parse_credentials",
    "payload_problem",
    "require_issuing_host",
    "take_lab_credentials",
    "upload_lab_report",
]
