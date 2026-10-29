"""Which API host a credential or an upload is about to go to (Security Desk L-1).

`DEFAULT_API_BASE` in `cli.config` is itself read from `$LLM_SPEED_API`, so it can never
be the yardstick for "is this the official host?": with a hostile `LLM_SPEED_API` the old
check compared the attacker's host with itself and stayed silent. Everything here
compares against the hard-coded `CANONICAL_API_BASE` instead.

Rules, applied before any request that carries a bearer token or uploads a result:

* plain ``http://`` is refused unless the host is this machine's loopback
  (``localhost``, ``127.0.0.1``, ``::1``); so are other schemes and URLs with
  credentials in them (`InsecureApiBase`, a clear error and a non-zero exit);
* any https host other than the canonical one gets a one-line stderr warning that
  names the host and where the URL came from (``--api-base``, ``$LLM_SPEED_API``,
  the agent's enrolled queue). It is printed once per process per host, so the agent
  and its preset child don't repeat it for every request.

The comparison is normalized: scheme and host case, a trailing dot or slash and the
default port (443 for https, 80 for http) don't make a URL "different".
"""

from __future__ import annotations

import os
import sys
from typing import TextIO
from urllib.parse import urlparse

CANONICAL_API_BASE = "https://api.llm-speed.com"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_PORTS = {"https": 443, "http": 80}

# (scheme, host, port) already warned about in this process.
_warned: set[tuple[str, str, int | None]] = set()


class InsecureApiBase(ValueError):
    """The API base must not receive a token or an upload."""


def _parts(api_base: str) -> tuple[str, str, int | None] | None:
    """(scheme, host, port) with case, trailing dot and default port normalized, or
    None when the URL can't be parsed or has no host."""
    try:
        u = urlparse((api_base or "").strip())
        host = (u.hostname or "").rstrip(".").lower()
        port = u.port
    except ValueError:
        return None
    if not host:
        return None
    scheme = (u.scheme or "").lower()
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return scheme, host, port


_CANONICAL = _parts(CANONICAL_API_BASE)


def is_canonical_api_base(api_base: str) -> bool:
    """True only for https://api.llm-speed.com (any case, trailing slash, :443)."""
    return _parts(api_base) == _CANONICAL


def same_api_base(a: str, b: str) -> bool:
    """Same scheme, host and port after normalization (path ignored)."""
    pa = _parts(a)
    return pa is not None and pa == _parts(b)


def is_loopback(api_base: str) -> bool:
    p = _parts(api_base)
    return bool(p and p[1] in LOOPBACK_HOSTS)


def display_host(api_base: str) -> str:
    p = _parts(api_base)
    if not p:
        return repr(api_base)
    scheme, host, port = p
    shown = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        shown += f":{port}"
    return shown


def check_api_base(api_base: str) -> None:
    """Raise `InsecureApiBase` unless the URL may carry a token / upload: https to any
    host, or plain http to this machine's loopback only. No credentials in the URL."""
    p = _parts(api_base)
    if p is None:
        raise InsecureApiBase(f"invalid API base URL {api_base!r} (no host)")
    scheme, host, _port = p
    try:
        u = urlparse(api_base.strip())
    except ValueError:  # pragma: no cover - _parts already parsed it
        raise InsecureApiBase(f"invalid API base URL {api_base!r}") from None
    if u.username is not None or u.password is not None:
        raise InsecureApiBase(
            "refusing an API base URL with credentials in it; pass the key with --api-key"
        )
    if scheme == "https":
        return
    if scheme == "http":
        if host in LOOPBACK_HOSTS:
            return
        raise InsecureApiBase(
            f"refusing to send results or an API key over plain http to "
            f"{display_host(api_base)}; use https:// (plain http is only allowed "
            "to localhost / 127.0.0.1 / ::1)"
        )
    raise InsecureApiBase(
        f"unsupported API base scheme {scheme or '(none)'!r} in {api_base!r}; use https://"
    )


def api_base_source(args=None) -> str:
    """Where the API base for this command came from, for the warning text."""
    if args is not None and getattr(args, "api_base", None):
        return "--api-base"
    if os.environ.get("LLM_SPEED_API"):
        return "the LLM_SPEED_API environment variable"
    return "the default"


def warn_if_noncanonical(
    api_base: str,
    *,
    source: str,
    bearer: bool,
    what: str | None = None,
    stream: TextIO | None = None,
) -> bool:
    """One stderr line when `api_base` isn't the canonical host. Returns True when a
    warning was printed (False for the canonical host or a repeat in this process)."""
    if is_canonical_api_base(api_base):
        return False
    p = _parts(api_base)
    if p is None:
        return False
    if p in _warned:
        return False
    _warned.add(p)
    what = what or ("your API key and results" if bearer else "results")
    (stream or sys.stderr).write(
        f"warning: sending {what} to non-default API host {display_host(api_base)} "
        f"(from {source}), not {CANONICAL_API_BASE}. If you did not set this "
        "yourself, stop and check your environment for an LLM_SPEED_API override "
        "or the --api-base you passed.\n"
    )
    return True


def guard_api_base(
    api_base: str,
    *,
    source: str,
    bearer: bool,
    stream: TextIO | None = None,
) -> None:
    """`check_api_base` then `warn_if_noncanonical`. Raises `InsecureApiBase`."""
    check_api_base(api_base)
    warn_if_noncanonical(api_base, source=source, bearer=bearer, stream=stream)


# --- N-5: the bearer token only goes to a non-canonical host the user opted in to -------

ALLOW_TOKEN_HOST_ENV = "LLM_SPEED_ALLOW_TOKEN_HOST"
ALLOW_TOKEN_HOST_FLAG = "--allow-token-to-host"


def normalize_host_spec(spec: str | None) -> str | None:
    """Normalize an opt-in value to the form `display_host` prints: lowercase host,
    no trailing dot, ``:port`` only when it isn't the scheme's default, IPv6 in
    brackets. Accepts ``host``, ``host:port``, ``[::1]:8787`` or a full URL."""
    spec = (spec or "").strip()
    if not spec:
        return None
    url = spec if "://" in spec else f"https://{spec}"
    p = _parts(url)
    if p is None:
        return None
    try:
        u = urlparse(url)
        if u.path not in ("", "/") or u.query or u.fragment or "@" in u.netloc:
            return None  # a host, not a URL with a path / query / credentials
    except ValueError:
        return None
    return display_host(url)


def token_opt_in_hosts(args=None) -> set[str]:
    """Hosts the user explicitly allowed to receive the API key / bearer token, from
    ``--allow-token-to-host HOST`` and ``$LLM_SPEED_ALLOW_TOKEN_HOST``."""
    out: set[str] = set()
    for raw in (
        getattr(args, "allow_token_to_host", None) if args is not None else None,
        os.environ.get(ALLOW_TOKEN_HOST_ENV),
    ):
        h = normalize_host_spec(raw)
        if h:
            out.add(h)
    return out


def token_allowed(api_base: str, args=None) -> bool:
    """May a bearer token be attached to requests to `api_base`? Always for the
    canonical host; for any other host only when it exactly matches an opt-in."""
    if is_canonical_api_base(api_base):
        return True
    if _parts(api_base) is None:
        return False
    return display_host(api_base) in token_opt_in_hosts(args)


def token_opt_in_hint(api_base: str) -> str:
    h = display_host(api_base)
    return f"{ALLOW_TOKEN_HOST_FLAG} {h} or {ALLOW_TOKEN_HOST_ENV}={h}"


def _reset_for_tests() -> None:
    _warned.clear()


__all__ = [
    "CANONICAL_API_BASE",
    "InsecureApiBase",
    "api_base_source",
    "check_api_base",
    "display_host",
    "guard_api_base",
    "is_canonical_api_base",
    "is_loopback",
    "normalize_host_spec",
    "same_api_base",
    "token_allowed",
    "token_opt_in_hint",
    "token_opt_in_hosts",
    "warn_if_noncanonical",
]
