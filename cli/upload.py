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
from .signing import sign_report_jws, verify_jws_token
from .types import RunReport

log = logging.getLogger("cli.upload")

RUNS_DIR = CACHE_DIR / "runs"


def warn_if_insecure_api_base(api_base: str, api_key: str | None) -> None:
    """Emit a stderr warning if a bearer token would be sent to a risky URL.

    Two risks the user picks the URL for; we refuse to do either silently:

    1. **Plaintext.** ``http://`` always warns — token visible on the wire.
    2. **Off-default host.** Even ``https://``, sending the token to a host
       other than the default API base is a credential exfil if --api-base
       was supplied (or env-injected) by an attacker. Warn so the user
       sees ``https://evil.example`` flagged before bearer goes out.

    Skip when there's no api_key (no credential at risk) or when --api-base
    is the canonical default (the case for normal users).
    """
    import sys as _sys
    from urllib.parse import urlparse

    if not api_key:
        return
    base_raw = (api_base or "").strip()
    base = base_raw.lower()
    if base.startswith("http://"):
        _sys.stderr.write(
            "warning: API key will be sent over plaintext to "
            f"{api_base}; this is unsafe. "
            "Pass --api-base with https:// or run with --strict-anon to drop the bearer.\n"
        )
        return
    # HTTPS — but is it the default host?
    try:
        host = (urlparse(base_raw).hostname or "").lower()
        default_host = (urlparse(DEFAULT_API_BASE).hostname or "").lower()
    except ValueError:
        host = ""
        default_host = ""
    if host and default_host and host != default_host:
        _sys.stderr.write(
            f"warning: API key will be sent to non-default host {host} "
            f"(default: {default_host}). If you did not configure --api-base "
            "yourself, abort and inspect your environment for a malicious "
            "LLMS_API_BASE / LLM_SPEED_API_BASE override. Run with "
            "--strict-anon to drop the bearer.\n"
        )


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


def build_upload_payload(
    report: RunReport,
    *,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
) -> dict[str, Any]:
    """Returns the exact body that will be POSTed: `{"jws": "<token>"}`."""
    token = sign_report_jws(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
    )
    return {"jws": token}


def _post_with_retries(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
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
                raise UploadError(
                    f"server rejected upload: {resp.status_code} {resp.text[:200]}"
                )
            last_exc = UploadError(f"server error {resp.status_code}")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            log.debug("upload attempt %d failed: %s", attempt, exc)
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))
    raise UploadError(f"upload failed after {max_retries} attempts: {last_exc}")


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
) -> str:
    body = build_upload_payload(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
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

    url = f"{api_base.rstrip('/')}/v1/results"
    data = _post_with_retries(
        url, body, headers=headers, timeout=timeout, max_retries=max_retries
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
    timeout: float = 30.0,
    max_retries: int = 3,
) -> str:
    """Re-upload a saved `{"jws": "..."}` blob.

    Verifies the JWS locally first so a tampered file fails fast instead of
    being rejected at the server.
    """
    token = saved.get("jws") if isinstance(saved, dict) else None
    if not isinstance(token, str) or not token:
        raise UploadError(
            "saved payload has no `jws` field; not a llm-speed offline run"
        )
    ok, reason, _ = verify_jws_token(token)
    if not ok:
        raise UploadError(
            f"saved JWS does not verify locally ({reason}); aborting "
            f"(file may have been edited or generated by an old client)"
        )
    headers = {"User-Agent": "llm-speed-cli/resume", "Content-Type": "application/json"}
    if anon:
        headers["X-LLM-Speed-Anon"] = "1"
    # Match the `upload_report` policy: any anon flag suppresses the bearer.
    if api_key and not anon:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{api_base.rstrip('/')}/v1/results"
    data = _post_with_retries(
        url, {"jws": token}, headers=headers, timeout=timeout, max_retries=max_retries
    )
    return _result_url_from_response(data)
