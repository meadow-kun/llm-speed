"""Hard guard: no upload from a mock / fake / test / demo environment, and a global
offline switch.

`check_upload_allowed(api_base)` is called right before every network upload
(cli.upload._post_with_retries). It raises
`UploadBlocked` when:

  * LLM_SPEED_OFFLINE=1 (or true/yes) is set - honoured by every upload path; or
  * the environment looks synthetic (see `synthetic_reason`) and the target is not a
    reserved test host (localhost, 127.0.0.1, *.test, *.example, *.invalid, *.localhost),
    so tests can exercise upload code against mocked HTTP but can never reach a real site.
"""

from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse

_TRUE = {"1", "true", "yes", "on"}
_MOCK_PATH_MARKERS = ("mock", "fixture", "fake")
_RESERVED_SUFFIXES = (".test", ".example", ".invalid", ".localhost")
_RESERVED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "example.com",
    "example.org",
    "example.net",
}


class UploadBlocked(RuntimeError):
    pass


def offline() -> bool:
    return os.environ.get("LLM_SPEED_OFFLINE", "").strip().lower() in _TRUE


def _looks_mock(path: str | None) -> bool:
    if not path:
        return False
    p = path.replace("\\", "/").lower()
    return any(
        f"/{m}" in p or f"{m}/" in p or f"_{m}" in p or f"{m}_" in p
        for m in _MOCK_PATH_MARKERS
    )


def synthetic_reason(extra_binaries: tuple[str | None, ...] = ()) -> str | None:
    """Why this environment is not real hardware/backends, or None."""
    if os.environ.get("LLM_SPEED_SYNTHETIC", "").strip().lower() in _TRUE:
        return "LLM_SPEED_SYNTHETIC is set (demo/synthetic mode)"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "running under a test runner"
    mock_env = sorted(k for k in os.environ if k.startswith("MOCK_"))
    if mock_env:
        return f"mock environment variables set ({', '.join(mock_env[:3])})"
    for name in ("nvidia-smi", "llama-server"):
        found = shutil.which(name)
        if _looks_mock(found):
            return f"{name} on PATH is a mock/test fixture"
    for b in extra_binaries:
        if _looks_mock(b):
            return "the backend binary is a mock/test fixture"
    srv = os.environ.get("LLAMA_SERVER")
    if _looks_mock(srv):
        return "LLAMA_SERVER points at a mock/test fixture"
    return None


def _reserved(api_base: str) -> bool:
    host = (urlparse(api_base).hostname or "").lower()
    return host in _RESERVED_HOSTS or host.endswith(_RESERVED_SUFFIXES)


def upload_block_reason(
    api_base: str, extra_binaries: tuple[str | None, ...] = ()
) -> str | None:
    if offline():
        return "LLM_SPEED_OFFLINE=1"
    why = synthetic_reason(extra_binaries)
    if why and not _reserved(api_base):
        return f"{why}; uploads from mock/test/demo runs are refused"
    return None


def check_upload_allowed(
    api_base: str, extra_binaries: tuple[str | None, ...] = ()
) -> None:
    why = upload_block_reason(api_base, extra_binaries)
    if why:
        raise UploadBlocked(f"upload refused: {why}")
