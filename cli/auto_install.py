"""Best-effort installer for missing backend dependencies.

The CLI should never silently mutate user state. Every install path here:

1. Detects whether the backend is already available (returns ``installed``).
2. Returns the *exact* shell command we would run (``proposed_command``).
3. Requires the caller to confirm via the y/n prompt before invocation.

The wizard surfaces the proposed command verbatim and asks for confirmation
per backend. Nothing in this module runs on import - all probes are pure
functions and ``offer_install`` is the only side-effecting entry point.

Supported install routes (best effort - we explicitly do not handle every
distro):

* ``ollama``    - macOS: ``brew install ollama``; Linux: official one-liner;
  Windows: skipped (no headless install path the user trusts).
* ``mlx_lm``    - ``pip install --user mlx-lm`` (Apple Silicon only).
* ``llama.cpp`` - macOS: ``brew install llama.cpp``; Linux: ``apt-get install
  llama.cpp`` if apt is present.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("cli.auto_install")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _which(binary: str) -> bool:
    return shutil.which(binary) is not None


def is_ollama_installed() -> bool:
    return _which("ollama")


def is_llama_cpp_installed() -> bool:
    return _which("llama-server") or _which("llama-cli")


def is_mlx_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mlx_lm") is not None
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Install plans
# ---------------------------------------------------------------------------


@dataclass
class InstallPlan:
    """A single install offer the user must confirm before we run it.

    ``command`` is rendered verbatim to the user before any execution.
    ``argv`` is the actual list passed to ``subprocess.run`` (split-out so we
    don't go through a shell unless the install really requires one).
    ``shell`` controls whether ``argv[0]`` is fed through ``/bin/sh -c``.
    """

    backend: str
    description: str
    command: str
    argv: list[str]
    shell: bool = False
    available: bool = True
    skip_reason: str | None = None
    # ``license`` and ``auto_installable`` gate whether the doctor may OFFER to
    # run the command. We only auto-run installs for permissively-licensed
    # tooling the user can redistribute/build freely; anything else is
    # "guide-only" (we print instructions + a link, never invoke). Unknown
    # backends default to ``auto_installable=False`` so the safe behaviour is
    # to guide, not to run.
    license: str | None = None
    auto_installable: bool = False


def plan_for(backend: str) -> InstallPlan:
    """Return a fresh InstallPlan for the named backend on this OS.

    Returns ``available=False`` with a ``skip_reason`` when we don't have a
    safe install path on the current platform.
    """
    name = backend.lower()
    osname = platform.system()

    if name == "ollama":
        if osname == "Darwin" and _which("brew"):
            return InstallPlan(
                backend="ollama",
                description="Install Ollama via Homebrew",
                command="brew install ollama",
                argv=["brew", "install", "ollama"],
                license="MIT",
                auto_installable=True,
            )
        if osname == "Linux":
            return InstallPlan(
                backend="ollama",
                description="Install Ollama via the official upstream installer (curl | sh)",
                command="curl -fsSL https://ollama.com/install.sh | sh",
                argv=["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                shell=True,
                license="MIT",
                auto_installable=True,
            )
        return InstallPlan(
            backend="ollama",
            description="Install Ollama",
            command="(no automated install path on this OS)",
            argv=[],
            available=False,
            skip_reason=f"no automated install for {osname}; see https://ollama.com/download",
        )

    if name in ("llama.cpp", "llama_cpp", "llamacpp"):
        if osname == "Darwin" and _which("brew"):
            return InstallPlan(
                backend="llama.cpp",
                description="Install llama.cpp via Homebrew",
                command="brew install llama.cpp",
                argv=["brew", "install", "llama.cpp"],
                license="MIT",
                auto_installable=True,
            )
        if osname == "Linux" and _which("apt-get"):
            return InstallPlan(
                backend="llama.cpp",
                description="Install llama.cpp via apt-get (requires sudo)",
                command="sudo apt-get update && sudo apt-get install -y llama.cpp",
                argv=[
                    "sh",
                    "-c",
                    "sudo apt-get update && sudo apt-get install -y llama.cpp",
                ],
                shell=True,
                license="MIT",
                auto_installable=True,
            )
        return InstallPlan(
            backend="llama.cpp",
            description="Install llama.cpp",
            command="(no automated install path on this OS)",
            argv=[],
            available=False,
            skip_reason="no brew/apt found; build from source at https://github.com/ggerganov/llama.cpp",
        )

    if name in ("mlx", "mlx_lm"):
        if osname != "Darwin" or platform.machine() != "arm64":
            return InstallPlan(
                backend="mlx",
                description="Install mlx-lm",
                command="(unavailable - MLX requires Apple Silicon)",
                argv=[],
                available=False,
                skip_reason="MLX requires macOS on arm64",
            )
        return InstallPlan(
            backend="mlx",
            description="Install mlx-lm via pip (--user)",
            command=f"{sys.executable} -m pip install --user mlx-lm",
            argv=[sys.executable, "-m", "pip", "install", "--user", "mlx-lm"],
            license="MIT",
            auto_installable=True,
        )

    return InstallPlan(
        backend=backend,
        description=f"Install {backend}",
        command="(unknown backend)",
        argv=[],
        available=False,
        skip_reason=f"unknown backend {backend!r}",
    )


def start_plan_for(backend: str) -> InstallPlan:
    """Plan to START an already-installed-but-not-running backend.

    Today only ``ollama`` has a daemon. On macOS+brew we can safely
    ``brew services start ollama`` (returns immediately). Everywhere else
    starting a server is guide-only: ``ollama serve`` blocks the terminal and
    ``systemctl start ollama`` needs root + systemd, so we never auto-run it.
    """
    name = backend.lower()
    osname = platform.system()
    if name == "ollama":
        if osname == "Darwin" and _which("brew"):
            return InstallPlan(
                backend="ollama",
                description="Start the Ollama daemon via Homebrew services",
                command="brew services start ollama",
                argv=["brew", "services", "start", "ollama"],
                license="MIT",
                auto_installable=True,
            )
        return InstallPlan(
            backend="ollama",
            description="Start the Ollama daemon",
            command="ollama serve   # (run in another terminal, or: systemctl start ollama)",
            argv=[],
            available=False,
            skip_reason=(
                "start the daemon yourself: run `ollama serve` in another terminal, "
                "or `sudo systemctl start ollama` if installed as a service"
            ),
        )
    return InstallPlan(
        backend=backend,
        description=f"Start {backend}",
        command="(no daemon to start)",
        argv=[],
        available=False,
        skip_reason=f"{backend} has no daemon to start",
    )


def pull_model_plan_for(backend: str, model: str | None = None) -> InstallPlan:
    """Plan to fetch a first model for an available backend.

    ``ollama pull`` is a safe, idempotent network download we can offer to run.
    For llama.cpp / mlx the user picks a GGUF / HF model themselves, so those
    are guide-only.
    """
    name = backend.lower()
    if name == "ollama":
        tag = model or "llama3.2:1b"
        return InstallPlan(
            backend="ollama",
            description=f"Pull a small starter model ({tag})",
            command=f"ollama pull {tag}",
            argv=["ollama", "pull", tag],
            license="MIT",
            auto_installable=True,
        )
    if name in ("llama.cpp", "llama_cpp", "llamacpp"):
        return InstallPlan(
            backend="llama.cpp",
            description="Download a GGUF model",
            command="(download a .gguf, e.g. from https://huggingface.co/models?library=gguf)",
            argv=[],
            available=False,
            skip_reason=(
                "llama.cpp needs a local .gguf file. Download one (e.g. from "
                "huggingface.co) and pass it with --model /path/to/model.gguf"
            ),
        )
    if name in ("mlx", "mlx_lm"):
        return InstallPlan(
            backend="mlx",
            description="Download an MLX model",
            command="(models resolve from the HuggingFace cache, e.g. mlx-community/*)",
            argv=[],
            available=False,
            skip_reason=(
                "MLX loads from the HuggingFace cache; pass --model "
                "mlx-community/<repo> and it will be fetched on first run"
            ),
        )
    return InstallPlan(
        backend=backend,
        description=f"Fetch a model for {backend}",
        command="(no automated model fetch)",
        argv=[],
        available=False,
        skip_reason=f"no automated model fetch for {backend}",
    )


# Optional Python extras → the pip package that provides them.
_EXTRA_PACKAGES = {
    "mlx": "mlx-lm",
    "mlx_lm": "mlx-lm",
    "vllm": "vllm",
    "exllamav2": "exllamav2",
}


def extra_plan_for(extra: str) -> InstallPlan:
    """Plan to ``pip install`` an optional accelerator extra (mlx-lm/vllm/...).

    All in-scope extras are permissively licensed, so we offer them. vLLM /
    ExLlamaV2 only make sense with an NVIDIA GPU + CUDA; we still offer the pip
    install and let the package's own wheels/checks complain if the GPU is
    absent.
    """
    pkg = _EXTRA_PACKAGES.get(extra.lower())
    if pkg is None:
        return InstallPlan(
            backend=extra,
            description=f"Install {extra}",
            command="(unknown extra)",
            argv=[],
            available=False,
            skip_reason=f"unknown extra {extra!r}",
        )
    return InstallPlan(
        backend=extra,
        description=f"Install {pkg} via pip (--user)",
        command=f"{sys.executable} -m pip install --user {pkg}",
        argv=[sys.executable, "-m", "pip", "install", "--user", pkg],
        license="Apache-2.0/MIT",
        auto_installable=True,
    )


# Manual-instruction text for guide-only situations (no safe auto-install).
_GUIDES = {
    "corporate-ca": (
        "Behind a TLS-intercepting proxy? Point the CLI at your corporate root "
        "CA bundle: export SSL_CERT_FILE=/path/to/ca.pem (or REQUESTS_CA_BUNDLE). "
        "We never disable certificate verification for you."
    ),
    "vllm": (
        "vLLM needs an NVIDIA GPU + CUDA. See https://docs.vllm.ai for the "
        "install matrix, then `pip install 'llm-speed[vllm]'`."
    ),
    "gguf": (
        "llama.cpp benchmarks a local .gguf file. Grab one from "
        "https://huggingface.co/models?library=gguf and pass --model /path.gguf."
    ),
}


def guide_for(topic: str) -> str | None:
    """Return human-readable manual guidance for a guide-only topic, or None."""
    return _GUIDES.get(topic.lower())


# ---------------------------------------------------------------------------
# Install runner
# ---------------------------------------------------------------------------


def confirm_and_install(
    plan: InstallPlan,
    *,
    confirm: Callable[[str], bool],
    runner: Callable[[InstallPlan], int] | None = None,
) -> bool:
    """Show the plan, ask the caller-supplied confirm() for explicit y/n, run.

    Returns True iff the install command exited with status 0. Refusal
    (confirm returned False) returns False with no side effects.
    """
    if not plan.available:
        return False
    prompt = (
        f"\nProposed install for {plan.backend}:\n"
        f"  $ {plan.command}\n"
        f"\nRun this now? [y/N] "
    )
    if not confirm(prompt):
        return False
    runner_fn = runner or _default_runner
    rc = runner_fn(plan)
    return rc == 0


def _default_runner(plan: InstallPlan) -> int:
    if not plan.argv:
        return 1
    log.info("running install: %s", plan.command)
    try:
        proc = subprocess.run(plan.argv, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("install failed: %s", exc)
        return 1
    return proc.returncode


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


def detect_missing() -> list[str]:
    """Return the list of supported backends not currently available locally.

    The wizard offers installs for these in priority order. ``llama.cpp`` is
    listed before ``ollama`` because it's a smaller install on macOS and
    happens to be the friendliest first-time backend; the wizard only offers
    the first available plan, not all of them.
    """
    missing = []
    if not is_llama_cpp_installed():
        missing.append("llama.cpp")
    if not is_ollama_installed():
        missing.append("ollama")
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and not is_mlx_installed()
    ):
        missing.append("mlx")
    return missing
