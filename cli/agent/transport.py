"""Thin HTTP transport for the agent queue. This file is the ONLY place that knows
the wire format: the settled spec, ops/AGENT_QUEUE_API.md (v1, 2026-10-22), which
supersedes the earlier proposal in docs/agent_queue_contract.md. Everything else talks
to `QueueClient`'s methods and the small result types."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from .. import __version__
from ..config import DEFAULT_API_BASE

CANONICAL_API = "https://api.llm-speed.com"


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


def check_api_base(api_base: str) -> None:
    """Any https queue (the official one or self-hosted); plain http only to this
    machine's loopback. No credentials, query or fragment in the URL. Tests opt out
    explicitly with LLM_SPEED_AGENT_INSECURE_OK=1."""
    if os.environ.get("LLM_SPEED_AGENT_INSECURE_OK") == "1":
        return
    try:
        u = urlparse(api_base)
        host = u.hostname
    except ValueError:
        u, host = urlparse(""), None
    loopback = host in ("localhost", "127.0.0.1", "::1")
    if (
        not host
        or u.scheme not in ("https", "http")
        or (u.scheme == "http" and not loopback)
        or u.username is not None
        or u.password is not None
        or u.query
        or u.fragment
    ):
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
        self.token = token
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
    def pair_start(self, capabilities: dict, label: str | None) -> dict:
        _, b = self._req(
            "POST",
            "/v1/agent/pair",
            auth=False,
            json={
                "cli_version": __version__,
                "label": label,
                "capabilities": capabilities,
            },
        )
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

    def claim(self, capabilities: dict) -> Claim:
        """capabilities includes "presets": [{name, version, sha256}] (queue API spec section 4)."""
        st, b, h = self._req(
            "POST",
            "/v1/agent/claim",
            json={"cli_version": __version__, "capabilities": capabilities},
            want_headers=True,
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
    def get_preset(self, name: str, version: int | None = None) -> dict:
        path = f"/v1/presets/{name}" + (
            f"/{int(version)}" if version is not None else ""
        )
        _, b = self._req("GET", path, auth=False)
        return b

    def get_revocations(self) -> dict:
        _, b = self._req("GET", "/v1/presets/revocations", auth=False)
        return b
