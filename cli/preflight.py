"""Dependency preflight engine.

Pure, side-effect-free checks that classify the host into "ready" vs the
specific non-happy path it's in. The interactive ``llm-speed doctor`` command
(and the bench preflight) consume the resulting :class:`Check` list to either
OFFER a fix (when ``Check.fix`` is an auto-installable plan) or PRINT guidance
(``Check.remediation``).

Nothing here prompts, installs, or mutates state — every external probe is
injectable so the whole engine is unit-testable without a network, a backend,
or the real filesystem. The doctor command owns all the prompting.

Status vocabulary:
  ok    — satisfied
  warn  — degraded but the CLI still works (e.g. offline, old Python)
  fail  — blocks the thing the check guards; if ``required`` it blocks `bench`
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import auto_install
from .auto_install import InstallPlan
from .config import CACHE_DIR, CONFIG_DIR, DEFAULT_API_BASE
from .registry import get_driver

log = logging.getLogger("cli.preflight")

OK = "ok"
WARN = "warn"
FAIL = "fail"

REQUIRED_PYTHON = (3, 10)
CORE_DEPS = ("httpx", "psutil", "rich", "cryptography", "joserfc")

# The backends the doctor walks the user through. Mirrors auto_install's
# supported set (detect_missing): these are the ones we have install plans for.
_BACKEND_CHECK_ORDER = ("llama.cpp", "ollama", "mlx")


@dataclass
class Check:
    """One preflight result.

    ``fix`` is present iff we can OFFER to run a command (an auto-installable
    :class:`~cli.auto_install.InstallPlan`). ``remediation`` is the guide-only
    fallback text. ``required`` marks checks whose failure blocks `bench`.
    """

    id: str
    title: str
    status: str
    detail: str = ""
    remediation: str | None = None
    fix: InstallPlan | None = None
    required: bool = False


# ---------------------------------------------------------------------------
# runtime checks
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_runtime_python(version_info: tuple[int, int] | None = None) -> Check:
    vi = version_info or (sys.version_info.major, sys.version_info.minor)
    if tuple(vi[:2]) >= REQUIRED_PYTHON:
        return Check(
            id="runtime.python",
            title="Python >= 3.10",
            status=OK,
            detail=f"Python {vi[0]}.{vi[1]}",
        )
    return Check(
        id="runtime.python",
        title="Python >= 3.10",
        status=WARN,
        detail=f"Python {vi[0]}.{vi[1]} is older than the required 3.10",
        remediation=(
            "Reinstall on a newer Python. The one-shot installer handles this: "
            "`curl -fsSL https://llm-speed.com/install.sh | sh` "
            "(installs uv + Python 3.12), or `uv python install 3.12`."
        ),
    )


def check_core_deps(is_available: Callable[[str], bool] = _module_available) -> Check:
    missing = [d for d in CORE_DEPS if not is_available(d)]
    if not missing:
        return Check(
            id="runtime.core_deps",
            title="Core Python dependencies",
            status=OK,
            detail=", ".join(CORE_DEPS) + " present",
            required=True,
        )
    return Check(
        id="runtime.core_deps",
        title="Core Python dependencies",
        status=FAIL,
        detail="missing: " + ", ".join(missing),
        required=True,
        remediation=(
            "The install is broken/incomplete. Reinstall: "
            "`pipx install --force llm-speed` (or `pip install -e .` in a dev checkout)."
        ),
    )


def _probe_writable(path: Path) -> str | None:
    """Return None if ``path`` is creatable+writable, else an error string."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".llm-speed-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return f"{path}: {exc}"
    return None


def check_cache_writable(paths: tuple[Path, ...] | None = None) -> Check:
    paths = paths if paths is not None else (CACHE_DIR, CONFIG_DIR)
    problems = [err for p in paths if (err := _probe_writable(p)) is not None]
    if not problems:
        return Check(
            id="runtime.cache_writable",
            title="Cache + config writable",
            status=OK,
            detail=f"{paths[0]} writable",
        )
    return Check(
        id="runtime.cache_writable",
        title="Cache + config writable",
        status=WARN,
        detail="; ".join(problems),
        remediation=(
            "Point LLM_SPEED_CACHE / LLM_SPEED_CONFIG at a writable directory, "
            "or fix the permissions on the paths above. Without it, runs can't "
            "be saved offline."
        ),
    )


# ---------------------------------------------------------------------------
# network check
# ---------------------------------------------------------------------------


def _probe_http(base: str) -> bool:
    try:
        import httpx

        r = httpx.get(f"{base.rstrip('/')}/healthz", timeout=4.0)
        return r.status_code < 500
    except Exception as exc:  # noqa: BLE001 — any failure means "unreachable"
        log.debug("network probe to %s failed: %s", base, exc)
        return False


def check_network(
    api_base: str | None = None, probe: Callable[[str], bool] | None = None
) -> Check:
    base = api_base or DEFAULT_API_BASE
    reach = probe or _probe_http
    if reach(base):
        return Check(
            id="net.api",
            title="Reach the upload API",
            status=OK,
            detail=f"{base} reachable",
        )
    return Check(
        id="net.api",
        title="Reach the upload API",
        status=WARN,
        detail=f"{base} not reachable",
        remediation=(
            "Offline works fine: `llm-speed bench --no-upload` saves locally and "
            "`--json out.json` writes the payload to upload later with "
            "`llm-speed bench --resume out.json`. Behind a TLS-intercepting "
            "corporate proxy, set SSL_CERT_FILE / REQUESTS_CA_BUNDLE to your root CA."
        ),
    )


# ---------------------------------------------------------------------------
# backend checks (the core "required dependency" gate)
# ---------------------------------------------------------------------------


def _backend_installed(name: str) -> bool:
    if name in ("llama.cpp", "llama_cpp", "llamacpp"):
        return auto_install.is_llama_cpp_installed()
    if name == "ollama":
        return auto_install.is_ollama_installed()
    if name in ("mlx", "mlx_lm"):
        return auto_install.is_mlx_installed()
    return False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _fix_or_guide(plan: InstallPlan) -> tuple[InstallPlan | None, str | None]:
    """Split a plan into (offerable fix, guide-only remediation)."""
    if plan.available and plan.auto_installable:
        return plan, None
    return None, plan.skip_reason


def classify_backend(
    name: str,
    *,
    installed: bool | None = None,
    get_driver_fn: Callable[[str], object] | None = None,
) -> Check:
    """Classify one backend into its non-happy-path sub-state.

    Sub-states (in order of severity):
      not installed              → FAIL, fix = install plan
      installed but unreachable  → FAIL, fix = start plan (e.g. ollama daemon)
      available but no model      → WARN, fix = pull-model plan
      available with a model      → OK
    """
    cid = f"backend.{name}"
    is_installed = _backend_installed(name) if installed is None else installed

    # mlx only exists on Apple Silicon — elsewhere it's an informational skip,
    # never a failure (the host simply can't run it).
    if name in ("mlx", "mlx_lm") and not _is_apple_silicon():
        return Check(
            id=cid,
            title=f"{name} backend",
            status=WARN,
            detail="MLX requires Apple Silicon; not applicable on this platform",
        )

    if not is_installed:
        fix, guide = _fix_or_guide(auto_install.plan_for(name))
        return Check(
            id=cid,
            title=f"{name} backend",
            status=FAIL,
            detail="not installed",
            fix=fix,
            remediation=guide,
        )

    get_drv = get_driver_fn or get_driver
    try:
        driver = get_drv(name)
        det = driver.detect()
        available = det.available
        notes = det.notes
    except Exception as exc:  # noqa: BLE001 — drivers self-isolate; treat as down
        log.debug("detect(%s) failed: %s", name, exc)
        driver = None
        available = False
        notes = str(exc)

    if not available:
        fix, guide = _fix_or_guide(auto_install.start_plan_for(name))
        return Check(
            id=cid,
            title=f"{name} backend",
            status=FAIL,
            detail=notes or "installed but not available",
            fix=fix,
            remediation=guide,
        )

    try:
        models = driver.list_models() if driver is not None else []
    except Exception as exc:  # noqa: BLE001
        log.debug("list_models(%s) failed: %s", name, exc)
        models = []

    if not models:
        fix, guide = _fix_or_guide(auto_install.pull_model_plan_for(name))
        return Check(
            id=cid,
            title=f"{name} backend",
            status=WARN,
            detail="available, but no model is cached",
            fix=fix,
            remediation=guide,
        )

    return Check(
        id=cid,
        title=f"{name} backend",
        status=OK,
        detail=f"{len(models)} model(s) available",
    )


def check_backends(
    *, get_driver_fn: Callable[[str], object] | None = None
) -> list[Check]:
    """Classify each candidate backend + synthesize the required `backend.any`."""
    checks = [
        classify_backend(n, get_driver_fn=get_driver_fn) for n in _BACKEND_CHECK_ORDER
    ]
    usable = any(c.status == OK for c in checks)
    summary = Check(
        id="backend.any",
        title="At least one usable backend",
        status=OK if usable else FAIL,
        detail=(
            "a backend is installed with a model"
            if usable
            else "no backend is ready to benchmark"
        ),
        required=True,
        remediation=(
            None
            if usable
            else "Install or start a backend below, or pull a model for one that's installed."
        ),
    )
    return checks + [summary]


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------


def run_checks(
    *,
    api_base: str | None = None,
    get_driver_fn: Callable[[str], object] | None = None,
    include_network: bool = True,
) -> list[Check]:
    """Run the full preflight and return checks in display order."""
    # Make sure drivers are registered (idempotent; self-isolating).
    try:
        from . import drivers as _drivers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.debug("driver import failed during preflight: %s", exc)

    checks: list[Check] = [
        check_runtime_python(),
        check_core_deps(),
        check_cache_writable(),
    ]
    checks += check_backends(get_driver_fn=get_driver_fn)
    if include_network:
        checks.append(check_network(api_base))
    return checks


def required_failed(checks: list[Check]) -> bool:
    """True if any ``required`` check is in FAIL — i.e. `bench` cannot run."""
    return any(c.required and c.status == FAIL for c in checks)
