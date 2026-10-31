"""Thin HTTP transport for the agent queue. This file is the ONLY place that knows
the wire format: the settled spec, ops/AGENT_QUEUE_API.md (v1, 2026-10-22), which
supersedes the earlier proposal in docs/agent_queue_contract.md. Everything else talks
to `QueueClient`'s methods and the small result types."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from .. import __version__
from .. import agent_job as _agent_job
from ..api_host import CANONICAL_API_BASE, InsecureApiBase, warn_if_noncanonical
from ..api_host import check_api_base as _check_upload_base
from ..config import DEFAULT_API_BASE

CANONICAL_API = CANONICAL_API_BASE


log = logging.getLogger("cli.agent")


class QueueError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(f"{status} {code} {message}".strip())
        self.status, self.code, self.retry_after = status, code, retry_after


@dataclass
class Claim:
    # {"job_id", "assignment_id", "preset": {name, version, sha256}, "overrides", "launch_id",
    #  "upload", "strict_anon", "require_approval", "max_runtime_min", "attempt", "created_at",
    #  "expires_at"}
    job: dict | None
    lease: dict | None  # {"lease_id", "expires_at", "lease_s", "heartbeat_s"}
    control: dict
    resumed: bool = False


# Test-suite hook, not a setting: only a test may flip it (monkeypatch.setattr). Nothing in
# a real run sets it, and no environment variable or flag reaches it (0.0.9's
# LLM_SPEED_AGENT_INSECURE_OK=1 is gone and is ignored if set).
_INSECURE_OK_FOR_TESTS = False


def check_api_base(api_base: str) -> None:
    """Any https queue (the official one or self-hosted); plain http only to this
    machine's loopback. No credentials, query or fragment in the URL."""
    if _INSECURE_OK_FOR_TESTS:
        return
    try:
        u = urlparse(api_base)
    except ValueError:
        u = urlparse("")
    try:
        # Same scheme / loopback / credentials rule as every CLI upload (cli.api_host).
        _check_upload_base(api_base)
        bad = bool(u.query or u.fragment)
    except InsecureApiBase:
        bad = True
    if bad:
        raise QueueError(
            0,
            "insecure_api_base",
            f"refusing to send the machine token to {u.scheme}://{u.netloc}",
        )


class QueueClient:
    def __init__(
        self,
        api_base: str | None = None,
        token: str | None = None,
        *,
        timeout: float = 30.0,
    ):
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        check_api_base(self.api_base)
        # L-1: a self-hosted queue is supported, but say so (once per process) before
        # the machine token or a pairing secret goes there.
        warn_if_noncanonical(
            self.api_base,
            source="the agent's queue (--api-base / enrolled base)",
            bearer=True,
            what="this machine's agent token and job results",
        )
        self.token = token
        # R-36: declare the upload-nonce feature in claims until a queue refuses the key.
        self._declare_features = True
        self._http = httpx.Client(
            timeout=timeout, headers={"User-Agent": f"llm-speed-agent/{__version__}"}
        )

    def close(self) -> None:
        self._http.close()

    # -- plumbing
    def _req(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
        json: Any = None,
        headers: dict | None = None,
        want_headers: bool = False,
    ):
        h = dict(headers or {})
        if auth:
            if not self.token:
                raise QueueError(401, "invalid_token", "not enrolled")
            h["Authorization"] = f"Bearer {self.token}"
        try:
            r = self._http.request(method, self.api_base + path, json=json, headers=h)
        except httpx.HTTPError as exc:
            raise QueueError(0, "network", type(exc).__name__) from None
        body: dict = {}
        if r.content:
            try:
                body = r.json()
            except ValueError:
                body = {}
        if r.status_code >= 400 and r.status_code != 410:
            err = (body.get("error") or {}) if isinstance(body, dict) else {}
            ra = r.headers.get("Retry-After")
            raise QueueError(
                r.status_code,
                err.get("code") or f"http_{r.status_code}",
                str(err.get("message") or "")[:200],
                float(ra) if ra and ra.replace(".", "", 1).isdigit() else None,
            )
        out = body if isinstance(body, dict) else {}
        if want_headers:
            return r.status_code, out, r.headers
        return r.status_code, out

    # -- pairing (no token yet)
    def pair_start(
        self,
        capabilities: dict,
        label: str | None,
        enrolment_token: str | None = None,
    ) -> dict:
        # Queues before the 2026-10-22 server fix rejected a JSON null label
        # (agent-queue.ts handlePair), so the key is left out when unset.
        body: dict[str, Any] = {
            "cli_version": __version__,
            "capabilities": capabilities,
            **({"label": label} if label else {}),
        }
        if enrolment_token:
            body["enrolment_token"] = enrolment_token  # R-8: sent once, never logged
        _, b = self._req("POST", "/v1/agent/pair", auth=False, json=body)
        return b

    def pair_poll(self, pairing_id: str, poll_secret: str) -> tuple[str, dict]:
        """('pending'|'approved'|'expired'|'denied'|'used'|'unknown', body).
        Queue API spec section 2: 202 pending, 200 approved with the token exactly once, then
        410 pairing_used / pairing_expired / pairing_denied; 404 unknown id or secret."""
        try:
            st, b = self._req(
                "GET",
                f"/v1/agent/pair/{pairing_id}",
                auth=False,
                headers={"X-Pairing-Secret": poll_secret},
            )
        except QueueError as e:
            if e.status == 404:
                return "unknown", {}
            raise
        if st == 410:
            code = (b.get("error") or {}).get("code", "pairing_expired")
            return {"pairing_denied": "denied", "pairing_used": "used"}.get(
                code, "expired"
            ), b
        if st == 200 and b.get("status") == "approved":
            return "approved", b
        if st == 202 or b.get("status") == "pending":
            return "pending", b
        raise QueueError(st, "unexpected_pairing_response")

    # -- agent routes
    def status(self) -> dict:
        _, b = self._req("GET", "/v1/agent/status")
        return b

    def claim(self, capabilities: dict, features: list[str] | None = None) -> Claim:
        """capabilities includes "presets": [{name, version, sha256}] (queue API spec section 4).
        `features` adds claim features (0.0.12: "signed_presets" = this machine also takes
        jobs pinning a published, signed preset it doesn't package)."""
        body: dict[str, Any] = {
            "cli_version": __version__,
            "capabilities": capabilities,
        }
        key = _agent_job.CLAIM_FEATURES_KEY
        if self._declare_features:
            # This build echoes the nonce (cli/agent_job.py), so it asks for it (R-36).
            body[key] = [_agent_job.UPLOAD_NONCE_FEATURE, *(features or [])]
        try:
            st, b, h = self._req(
                "POST", "/v1/agent/claim", json=body, want_headers=True
            )
        except QueueError as e:
            # A queue from before R-36 refuses unknown claim keys ("unknown field: features").
            # Drop the key for this process and ask once more; nothing else is retried.
            if not (
                key in body and e.status == 400 and f"unknown field: {key}" in str(e)
            ):
                raise
            self._declare_features = False
            log.info("the queue doesn't take claim %s yet; claiming without it", key)
            body.pop(key)
            st, b, h = self._req(
                "POST", "/v1/agent/claim", json=body, want_headers=True
            )
        if st == 204 or not b.get("job"):
            control = dict(b.get("control") or {})
            pa = h.get("X-Poll-After-S")
            if pa and pa.isdigit():
                control["poll_after_s"] = int(pa)
            if h.get("X-Agent-Paused") == "1":
                control["paused"] = True
            return Claim(None, None, control)
        return Claim(
            b["job"],
            b.get("lease") or {},
            b.get("control") or {},
            bool(b.get("resumed")),
        )

    def heartbeat(
        self, job_id: str, lease_id: str, phase: str, progress: dict | None = None
    ) -> dict:
        _, b = self._req(
            "POST",
            f"/v1/agent/jobs/{job_id}/heartbeat",
            json={"lease_id": lease_id, "phase": phase, "progress": progress or {}},
        )
        return b

    def release(self, job_id: str, lease_id: str, reason: str) -> None:
        self._req(
            "POST",
            f"/v1/agent/jobs/{job_id}/release",
            json={"lease_id": lease_id, "reason": reason},
        )

    def result(self, job_id: str, report: dict, idempotency_key: str) -> dict:
        _, b = self._req(
            "POST",
            f"/v1/agent/jobs/{job_id}/result",
            json=report,
            headers={"Idempotency-Key": idempotency_key},
        )
        return b

    # -- phase 2
    # Response caps (Security Desk R-98): a bundle is <= 64 KiB of bytes (the Worker's
    # MAX_BUNDLE), so its base64url envelope stays well under 128 KiB; a revocation list is
    # small. Anything larger is refused before it is parsed.
    MAX_PRESET_RESPONSE = 128 * 1024
    MAX_REVOCATIONS_RESPONSE = 256 * 1024

    def _get_capped(self, path: str, cap: int) -> dict:
        """Unauthenticated GET on the enrolled api_base, read at most `cap` bytes."""
        try:
            with self._http.stream("GET", self.api_base + path) as r:
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > cap:
                    raise QueueError(r.status_code, "response_too_large")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > cap:
                        raise QueueError(r.status_code, "response_too_large")
                status = r.status_code
        except httpx.HTTPError as exc:
            raise QueueError(0, "network", type(exc).__name__) from None
        try:
            body = json.loads(bytes(buf)) if buf else {}
        except ValueError:
            body = {}
        if status >= 400:
            err = (body.get("error") or {}) if isinstance(body, dict) else {}
            raise QueueError(status, err.get("code") or f"http_{status}")
        return body if isinstance(body, dict) else {}

    def get_preset(self, name: str, version: int | None = None) -> dict:
        path = f"/v1/presets/{name}" + (
            f"/{int(version)}" if version is not None else ""
        )
        return self._get_capped(path, self.MAX_PRESET_RESPONSE)

    def get_revocations(self) -> dict:
        return self._get_capped(
            "/v1/presets/revocations", self.MAX_REVOCATIONS_RESPONSE
        )
