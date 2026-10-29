"""JWS upload + offline save.

POST body shape: `{"jws": "<compact-token>"}`.
The server decodes + verifies + extracts the payload from the protected JWS.
Local saves are the same shape — `--resume` re-POSTs the saved JWS as-is.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .config import CACHE_DIR, DEFAULT_API_BASE
from .signing import sign_payload_jws, sign_report_jws, verify_jws_token
from .types import RunReport

log = logging.getLogger("cli.upload")

RUNS_DIR = CACHE_DIR / "runs"


def warn_if_insecure_api_base(
    api_base: str, api_key: str | None, *, source: str | None = None
) -> None:
    """Stderr warning before a bearer token or a result goes to a risky URL.

    Thin wrapper over `cli.api_host` (Security Desk L-1). The yardstick is the
    hard-coded canonical host, never `DEFAULT_API_BASE` (that one is read from
    $LLM_SPEED_API, so a hostile override used to compare equal to itself and stay
    silent). Plain http to a non-loopback host is refused by `check_api_base`, which
    callers run first; if one didn't, this still says so loudly.
    """
    from . import api_host

    try:
        api_host.check_api_base(api_base)
    except api_host.InsecureApiBase as exc:
        import sys as _sys

        _sys.stderr.write(f"warning: {exc}\n")
        return
    api_host.warn_if_noncanonical(
        api_base,
        source=source or api_host.api_base_source(),
        bearer=bool(api_key),
    )


def _require_safe_base(api_base: str) -> None:
    """Defence in depth: the upload functions themselves never POST to a URL
    `cli.api_host.check_api_base` refuses (plain http off loopback, etc.)."""
    from .api_host import InsecureApiBase, check_api_base

    try:
        check_api_base(api_base)
    except InsecureApiBase as exc:
        raise UploadError(str(exc)) from None


# ---------------------------------------------------------------------------
# Offline save
# ---------------------------------------------------------------------------


def default_offline_path(report: RunReport) -> Path:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fp = report.fingerprint.fingerprint_hash or "nofp"
    return RUNS_DIR / f"{ts}-{fp}.json"


def save_offline(
    report: RunReport,
    path: Path | None = None,
    *,
    strict_anon: bool = False,
    include_raw_timings: bool = True,
) -> Path:
    """Sign with `include_raw_timings=True` (local artifact keeps full data) and write.

    Run artifacts are written 0600 inside a 0700 parent directory. The local
    file is a superset of the upload payload (it keeps raw timings the
    upload would strip), so on a multi-user host a default-umask 0644 leaks
    benchmark internals to every other user — including the public key
    that ties all of this user's runs together. Tighten the modes
    explicitly so the privacy contract holds at the filesystem boundary.
    """
    target = path or default_offline_path(report)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Catch the case where the directory existed pre-fix at 0755.
    try:
        os.chmod(target.parent, 0o700)
    except OSError:
        # FUSE / network mounts may refuse chmod; best-effort.
        log.debug("could not chmod %s to 0o700", target.parent)
    token = sign_report_jws(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
    )
    # Write at 0600 atomically: open with restrictive mode, then write.
    fd = os.open(
        str(target),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"jws": token}, indent=2))
    except Exception:
        # `os.fdopen` takes ownership of fd on success; on failure we close.
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    # Belt-and-braces: also chmod after write in case the file existed
    # pre-fix at 0644 (os.O_CREAT respects existing mode unless O_EXCL).
    try:
        os.chmod(target, 0o600)
    except OSError:
        log.debug("could not chmod %s to 0o600", target)
    log.debug("saved offline run to %s", target)
    return target


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class UploadError(RuntimeError):
    pass


class UploadRejected(UploadError):
    """The server answered 4xx. ``status`` is the HTTP status code; ``code`` is the
    JSON body's ``code`` when there is one (e.g. ``credit_reserved``)."""

    def __init__(
        self,
        message: str,
        status: int,
        code: str | None = None,
        error: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.error = error  # the JSON body's ``error`` text, e.g. "invalid agent_job"

    @property
    def agent_job_refused(self) -> bool:
        """The queue refused the signed `agent_job` field (security retest R-9). The job
        fails; the run is never sent again without it."""
        return self.status == 400 and self.error == "invalid agent_job"


# What the CLI says when llm-speed.com refuses the `credit` next to the JWS (400 with one
# of these codes, ops/CREDIT.md). The run is then uploaded once more, same JWS, no credit.
_CHANGE = (
    "set a different one with `llm-speed config credit @name` (or u/name), or remove it "
    "with `llm-speed config credit --clear`."
)
CREDIT_REJECTED_NOTICES = {
    "credit_reserved": (
        "note: {handle} is a reserved name on llm-speed.com, so this run was uploaded "
        "without it. Please don't keep using it: " + _CHANGE
    ),
    "credit_invalid_handle": (
        "note: llm-speed.com didn't accept {handle} as a valid handle, so this run was "
        "uploaded without it. Please change it: " + _CHANGE
    ),
    "credit_invalid_platform": (
        "note: llm-speed.com didn't recognise the platform for {handle}, so this run was "
        "uploaded without it. Your handle stays saved; if this keeps happening, update "
        "llm-speed or " + _CHANGE
    ),
    "credit_invalid": (
        "note: llm-speed.com couldn't read the credit sent with {handle}, so this run was "
        "uploaded without it. Your handle stays saved; if this keeps happening, update "
        "llm-speed."
    ),
    "credit_not_allowed": (
        "note: llm-speed.com doesn't take credit on anonymous uploads, so this run was "
        "uploaded without {handle}. Your handle stays saved for normal uploads."
    ),
}
CREDIT_REJECTED_FALLBACK = (
    "note: llm-speed.com didn't accept {handle} ({code}), so this run was uploaded "
    "without it. Your handle stays saved; to change it: " + _CHANGE
)


def credit_rejected_notice(code: str, credit: dict[str, Any] | None) -> str:
    c = credit or {}
    handle = (
        f"@{c.get('handle')}" if c.get("platform") == "x" else f"u/{c.get('handle')}"
    )
    tmpl = CREDIT_REJECTED_NOTICES.get(code, CREDIT_REJECTED_FALLBACK)
    return tmpl.format(handle=handle, code=code)


def _error_text(resp: Any) -> str | None:
    try:
        data = resp.json()
    except ValueError:
        return None
    err = data.get("error") if isinstance(data, dict) else None
    return err if isinstance(err, str) and len(err) <= 64 else None


def _error_code(resp: Any) -> str | None:
    try:
        data = resp.json()
    except ValueError:
        return None
    code = data.get("code") if isinstance(data, dict) else None
    return code if isinstance(code, str) and len(code) <= 64 else None


def _with_credit(
    body: dict[str, Any], credit: Any, strict_anon: bool
) -> dict[str, Any]:
    """Attach the optional `credit` field next to the JWS. Never under strict-anon."""
    if credit is None or strict_anon:
        return body
    payload = credit.to_payload() if hasattr(credit, "to_payload") else dict(credit)
    return {
        **body,
        "credit": {"platform": payload["platform"], "handle": payload["handle"]},
    }


def build_upload_payload(
    report: RunReport,
    *,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
    credit: Any = None,
) -> dict[str, Any]:
    """Returns the exact body that will be POSTed: `{"jws": "<token>"}`, plus
    `"credit": {"platform", "handle"}` when the user opted in (never under
    strict-anon). The credit sits outside the signed JWS so the local save and
    `--resume` artifacts are unchanged and an unknown-field 400 can be retried
    with the identical token."""
    token = sign_report_jws(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
    )
    return _with_credit({"jws": token}, credit, strict_anon)


def _post_with_retries(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    from .upload_guard import UploadBlocked, check_upload_allowed

    try:
        check_upload_allowed(url)
    except UploadBlocked as exc:
        raise UploadError(str(exc)) from None
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=body, headers=headers)
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    return {}
            if 400 <= resp.status_code < 500:
                raise UploadRejected(
                    f"server rejected upload: {resp.status_code} {resp.text[:200]}",
                    resp.status_code,
                    _error_code(resp),
                    _error_text(resp),
                )
            last_exc = UploadError(f"server error {resp.status_code}")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            log.debug("upload attempt %d failed: %s", attempt, exc)
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))
    raise UploadError(f"upload failed after {max_retries} attempts: {last_exc}")


def _post_body(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    notify=None,
) -> dict[str, Any]:
    """POST. If llm-speed.com refuses the `credit` next to the JWS (400 with a JSON
    `code` starting with `credit_`, ops/CREDIT.md), say why and retry once with the same
    JWS and no credit. Any other 400 / 4xx is not retried."""
    try:
        return _post_with_retries(
            url, body, headers=headers, timeout=timeout, max_retries=max_retries
        )
    except UploadRejected as exc:
        code = exc.code or ""
        if exc.status != 400 or "credit" not in body or not code.startswith("credit_"):
            raise
        log.debug("credit refused (%s); retrying once without it", code)
        bare = {k: v for k, v in body.items() if k != "credit"}
        data = _post_with_retries(
            url, bare, headers=headers, timeout=timeout, max_retries=max_retries
        )
        msg = credit_rejected_notice(code, body.get("credit"))
        if notify is None:
            import sys as _sys

            _sys.stderr.write(msg + "\n")
        else:
            notify(msg)
        return data


# Logged (no values) when an upload falls back to the "EdDSA" header name.
OLD_ALG_SERVER_NOTE = (
    'the server is older than the Worker that accepts JWS alg "Ed25519" (a763204); '
    're-signed once as "EdDSA" and sent again'
)


def _token_alg(token: str) -> str | None:
    import base64

    try:
        seg = token.split(".")[0]
        return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))).get(
            "alg"
        )
    except Exception:  # noqa: BLE001
        return None


def _resign_with_fallback_alg(token: str, *, strict_anon: bool) -> str | None:
    """The same payload signed with the older "EdDSA" header name, or None when that is not
    possible without changing who signed it (a non-strict token signed by another key)."""
    from .signing import (
        FALLBACK_ALG,
        get_or_create_keypair,
        public_jwk,
        sign_payload_jws,
    )

    ok, _, payload = verify_jws_token(token)
    if not ok or not isinstance(payload, dict):
        return None
    if not strict_anon:
        import base64

        seg = token.split(".")[0]
        header = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
        if header.get("jwk") != public_jwk(get_or_create_keypair()[1]):
            return None
    return sign_payload_jws(payload, strict_anon=strict_anon, alg=FALLBACK_ALG)


def _post_signed(
    url: str,
    body: dict[str, Any],
    *,
    strict_anon: bool,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    notify=None,
) -> dict[str, Any]:
    """POST; if the server refuses the "Ed25519" header name ("unexpected alg", an older
    Worker), re-sign the same payload once with "EdDSA" and POST that (Security Desk fix 5).
    Nothing is saved or printed along the way."""
    from .signing import ALG, FALLBACK_ALG

    try:
        return _post_body(
            url,
            body,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            notify=notify,
        )
    except UploadRejected as exc:
        if (
            exc.status not in (400, 401)
            or "unexpected alg" not in str(exc)
            or ALG == FALLBACK_ALG
            or _token_alg(body.get("jws", "")) != ALG
        ):
            raise
        token = _resign_with_fallback_alg(body["jws"], strict_anon=strict_anon)
        if token is None:
            raise
        log.warning(OLD_ALG_SERVER_NOTE)
        return _post_body(
            url,
            {**body, "jws": token},
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            notify=notify,
        )


def _result_url_from_response(data: dict[str, Any]) -> str:
    url = data.get("url") or data.get("result_url")
    if url:
        return url
    rid = data.get("id") or data.get("result_id")
    if rid:
        return f"https://llm-speed.com/r/{rid}"
    raise UploadError("server response missing url/id")


def upload_report(
    report: RunReport,
    *,
    api_base: str = DEFAULT_API_BASE,
    api_key: str | None = None,
    anon: bool = False,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
    timeout: float = 30.0,
    max_retries: int = 3,
    credit: Any = None,
    notify=None,
) -> str:
    if strict_anon or anon:
        credit = None  # belt and braces: anon modes never carry a handle
    body = build_upload_payload(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
        credit=credit,
    )
    if strict_anon:
        # Be indistinguishable from a generic httpx client.
        headers: dict[str, str] = {"Content-Type": "application/json"}
    else:
        headers = {
            "User-Agent": f"llm-speed-cli/{report.cli_version}",
            "Content-Type": "application/json",
        }
        if anon:
            headers["X-LLM-Speed-Anon"] = "1"
    # `--anon` (soft anon) and `--strict-anon` BOTH suppress the bearer token.
    # A user passing `--anon` reasonably expects no identifying credential to
    # leave the machine; only the unmarked default mode forwards the api_key.
    if api_key and not strict_anon and not anon:
        headers["Authorization"] = f"Bearer {api_key}"

    _require_safe_base(api_base)
    url = f"{api_base.rstrip('/')}/v1/results"
    data = _post_signed(
        url,
        body,
        strict_anon=strict_anon,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        notify=notify,
    )
    return _result_url_from_response(data)


def upload_or_save(
    report: RunReport,
    *,
    api_base: str = DEFAULT_API_BASE,
    api_key: str | None = None,
    anon: bool = False,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
    credit: Any = None,
    notify=None,
) -> tuple[str | None, Path | None]:
    try:
        return (
            upload_report(
                report,
                api_base=api_base,
                api_key=api_key,
                anon=anon,
                strict_anon=strict_anon,
                include_raw_timings=include_raw_timings,
                credit=credit,
                notify=notify,
            ),
            None,
        )
    except UploadError as exc:
        log.warning("upload failed (%s); saving locally", exc)
        path = save_offline(report, strict_anon=strict_anon)
        return None, path


# ---------------------------------------------------------------------------
# Resume: re-upload a previously-saved JWS
# ---------------------------------------------------------------------------


def upload_saved_payload(
    saved: dict[str, Any],
    *,
    api_base: str = DEFAULT_API_BASE,
    api_key: str | None = None,
    anon: bool = False,
    strict_anon: bool = False,
    timeout: float = 30.0,
    max_retries: int = 3,
    credit: Any = None,
    notify=None,
    agent_job: dict[str, str] | None = None,
) -> str:
    """Re-upload a saved `{"jws": "..."}` blob.

    Verifies the JWS locally first so a tampered file fails fast instead of
    being rejected at the server.

    `agent_job` (security retest R-9, only from an `llm-speed agent` preset child, always
    strict-anon): the verified payload gets `"agent_job": {assignment_id, upload_nonce}`
    and is re-signed in memory with a fresh strict-anon key just before the POST, so the
    nonce sits inside the signature and never in the saved file. Without it, a saved
    payload that already carries `agent_job` is refused (a nonce is never reused).
    """
    token = saved.get("jws") if isinstance(saved, dict) else None
    if not isinstance(token, str) or not token:
        raise UploadError(
            "saved payload has no `jws` field; not a llm-speed offline run"
        )
    ok, reason, saved_payload = verify_jws_token(token)
    if not ok:
        raise UploadError(
            f"saved JWS does not verify locally ({reason}); aborting "
            f"(file may have been edited or generated by an old client)"
        )
    if isinstance(saved_payload, dict) and saved_payload.get("agent_job") is not None:
        raise UploadError(
            "saved payload carries an agent job field; it is not re-uploaded"
        )
    if agent_job is not None:
        if not strict_anon:
            raise UploadError("agent job uploads are strict-anon only")
        from .agent_job import payload_field

        field = payload_field(agent_job)
        if field is None or not isinstance(saved_payload, dict):
            raise UploadError("invalid agent job context")
        token = sign_payload_jws(
            {**saved_payload, "agent_job": field}, strict_anon=True
        )
    if strict_anon:
        # Same as upload_report: indistinguishable from a generic client.
        headers: dict[str, str] = {"Content-Type": "application/json"}
    else:
        headers = {
            "User-Agent": "llm-speed-cli/resume",
            "Content-Type": "application/json",
        }
    if anon and not strict_anon:
        headers["X-LLM-Speed-Anon"] = "1"
    # Match the `upload_report` policy: any anon flag suppresses the bearer.
    if api_key and not anon and not strict_anon:
        headers["Authorization"] = f"Bearer {api_key}"
    if strict_anon or anon:
        credit = None
    _require_safe_base(api_base)
    url = f"{api_base.rstrip('/')}/v1/results"
    data = _post_signed(
        url,
        _with_credit({"jws": token}, credit, strict_anon),
        strict_anon=strict_anon,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        notify=notify,
    )
    return _result_url_from_response(data)
