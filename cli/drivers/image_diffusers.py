"""Diffusers text-to-image driver (Apple Silicon MPS / CUDA / CPU).

The image-generation analogue of `cli.drivers.mlx`: instead of streaming tokens,
it runs a fixed text-to-image diffusion pipeline and reports how long a
generation took, so a workload can turn that into images/second (the image-gen
answer to tok/s).

Heavy imports of `torch` and `diffusers` are deferred to method bodies after a
feature check, so this module imports cleanly on machines without them installed
(mirroring every other driver). Models download to the Hugging Face cache and
therefore honour `HF_HOME` / `HF_HUB_CACHE` — nothing is written next to the code.

`diffusers` was chosen over `mflux` because the fast/distilled turbo models this
runner targets (SD-Turbo, SDXL-Turbo, SD 1.5) are diffusers-native and small,
whereas mflux is FLUX-only and would force a multi-GB download that competes with
the MLX LLM sweep for the same Metal device and external drive.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..registry import register_driver
from ..timing import stopwatch
from ..types import BackendDetection, ChatRunOutcome, ModelRef

LOG = logging.getLogger("cli.drivers.diffusers")

_BACKEND_NAME = "diffusers"

# Curated txt2img models known to load through `AutoPipelineForText2Image`.
# The turbo/distilled entries render in 1-4 steps — ideal for a quick,
# GPU-light speed bench. (hf_id, human_name, note).
_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("stabilityai/sd-turbo", "SD-Turbo", "distilled SD 2.1, 1-4 steps"),
    ("stabilityai/sdxl-turbo", "SDXL-Turbo", "distilled SDXL, 1-4 steps"),
    (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "SDXL-Base-1.0",
        "SDXL, ~25-40 steps",
    ),
    ("runwayml/stable-diffusion-v1-5", "SD-1.5", "SD 1.5, ~20-30 steps"),
)


@dataclass
class ImageGenOutcome:
    """Low-level result of a single text-to-image call. Workloads compose these
    the way chat workloads compose `ChatRunOutcome`."""

    success: bool
    width: int
    height: int
    steps: int
    num_images: int
    wall_ms: float
    peak_memory_gb: float | None = None
    device: str | None = None
    dtype: str | None = None
    backend_extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _hf_hub_cache_dir() -> Path:
    """Where diffusers/huggingface_hub will read+write model snapshots."""
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


class DiffusersDriver:
    """Text-to-image driver. Caches a loaded pipeline per (model, device, dtype)."""

    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._pipes: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect(self) -> BackendDetection:
        torch_spec = importlib.util.find_spec("torch")
        diffusers_spec = importlib.util.find_spec("diffusers")
        missing = [
            n
            for n, s in (("torch", torch_spec), ("diffusers", diffusers_spec))
            if s is None
        ]
        if missing:
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes=(
                    f"not installed: {', '.join(missing)} "
                    "(pipx inject llm-speed torch diffusers transformers accelerate)"
                ),
            )

        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
        except Exception as exc:  # noqa: BLE001
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes=f"import failed: {exc}",
            )

        device, flags = self._pick_device(torch)
        runtime = {
            "torch": getattr(torch, "__version__", "unknown") or "unknown",
            "diffusers": getattr(diffusers, "__version__", "unknown") or "unknown",
            "device": device,
            "hf_hub_cache": str(_hf_hub_cache_dir()),
        }
        return BackendDetection(
            available=True,
            name=_BACKEND_NAME,
            version=getattr(diffusers, "__version__", None),
            build_flags=flags,
            runtime_versions=runtime,
            notes=f"diffusers text-to-image on {device}",
        )

    # ------------------------------------------------------------------
    # model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[ModelRef]:
        downloaded = self._scan_cache_pipelines()
        out: list[ModelRef] = []
        seen: set[str] = set()

        for hf_id, human, note in _CATALOG:
            seen.add(hf_id)
            snapshot = downloaded.get(hf_id)
            out.append(
                ModelRef(
                    backend=_BACKEND_NAME,
                    identifier=hf_id,
                    name=human,
                    extras={
                        "note": note,
                        "downloaded": snapshot is not None,
                        **({"snapshot_path": snapshot} if snapshot else {}),
                    },
                )
            )

        # Surface any other diffusion pipelines already in the cache.
        for hf_id, snapshot in downloaded.items():
            if hf_id in seen:
                continue
            out.append(
                ModelRef(
                    backend=_BACKEND_NAME,
                    identifier=hf_id,
                    name=hf_id.split("/")[-1],
                    extras={"downloaded": True, "snapshot_path": snapshot},
                )
            )
        return out

    @staticmethod
    def _scan_cache_pipelines() -> dict[str, str]:
        """Map hf_id -> snapshot path for cached repos that look like a
        diffusers pipeline (a snapshot containing `model_index.json`)."""
        cache = _hf_hub_cache_dir()
        found: dict[str, str] = {}
        if not cache.exists():
            return found
        try:
            repo_dirs = list(cache.iterdir())
        except OSError:
            return found
        for repo_dir in repo_dirs:
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue
            snapshots = repo_dir / "snapshots"
            if not snapshots.exists():
                continue
            for snap in snapshots.iterdir():
                if snap.is_dir() and (snap / "model_index.json").exists():
                    tail = repo_dir.name[len("models--") :]
                    hf_id = tail.replace("--", "/", 1).replace("--", "/")
                    found[hf_id] = str(snap)
                    break
        return found

    # ------------------------------------------------------------------
    # image generation (the image-gen analogue of run_chat)
    # ------------------------------------------------------------------
    def generate_image(
        self,
        model: ModelRef,
        prompt: str,
        *,
        steps: int,
        width: int,
        height: int,
        guidance_scale: float = 0.0,
        seed: int = 0,
        num_images: int = 1,
        extras: dict[str, Any] | None = None,
    ) -> ImageGenOutcome:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("diffusers") is None
        ):
            return ImageGenOutcome(
                success=False,
                width=width,
                height=height,
                steps=steps,
                num_images=0,
                wall_ms=0.0,
                error="torch/diffusers not installed",
            )

        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
        except Exception as exc:  # noqa: BLE001
            return ImageGenOutcome(
                success=False,
                width=width,
                height=height,
                steps=steps,
                num_images=0,
                wall_ms=0.0,
                error=f"import failed: {exc}",
            )

        device, _ = self._pick_device(torch)
        dtype_name, dtype = self._pick_dtype(torch, device, extras)

        try:
            pipe = self._load(model.identifier, device, dtype, dtype_name)
        except Exception as exc:  # noqa: BLE001
            return ImageGenOutcome(
                success=False,
                width=width,
                height=height,
                steps=steps,
                num_images=0,
                wall_ms=0.0,
                device=device,
                dtype=dtype_name,
                error=f"pipeline load failed: {exc}",
            )

        # Seed on CPU: diffusers accepts a CPU generator even for an MPS/CUDA
        # pipeline and it is the portable, reproducible choice across devices.
        generator: Any | None = None
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        except Exception:  # noqa: BLE001
            generator = None

        backend_extras = {
            "torch_version": getattr(torch, "__version__", None),
            "diffusers_version": getattr(diffusers, "__version__", None),
            "pipeline_class": type(pipe).__name__,
            "guidance_scale": guidance_scale,
            "seed": int(seed),
        }

        self._sync(torch, device)
        try:
            with stopwatch() as wall:
                pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images,
                    generator=generator,
                )
                # MPS/CUDA are async; force completion before the clock stops.
                self._sync(torch, device)
        except Exception as exc:  # noqa: BLE001
            return ImageGenOutcome(
                success=False,
                width=width,
                height=height,
                steps=steps,
                num_images=0,
                wall_ms=0.0,
                device=device,
                dtype=dtype_name,
                backend_extras=backend_extras,
                error=f"generation failed: {exc}",
            )

        return ImageGenOutcome(
            success=True,
            width=width,
            height=height,
            steps=steps,
            num_images=num_images,
            wall_ms=wall[0],
            peak_memory_gb=self._peak_memory_gb(torch, device),
            device=device,
            dtype=dtype_name,
            backend_extras=backend_extras,
        )

    # ------------------------------------------------------------------
    # protocol conformance: this backend does images, not chat
    # ------------------------------------------------------------------
    def run_chat(
        self,
        model: ModelRef,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool = True,
        extras: dict[str, Any] | None = None,
    ) -> ChatRunOutcome:
        return ChatRunOutcome(
            success=False,
            output_text="",
            prompt_tokens=0,
            output_tokens=0,
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=0.0,
            error=(
                "diffusers is a text-to-image backend; run an image-gen workload "
                "(txt2img-512 / sdxl-1024), not a chat workload"
            ),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load(self, identifier: str, device: str, dtype: Any, dtype_name: str) -> Any:
        key = f"{identifier}|{device}|{dtype_name}"
        if key in self._pipes:
            return self._pipes[key]
        diffusers = importlib.import_module("diffusers")
        auto_pipe = diffusers.AutoPipelineForText2Image
        LOG.info(
            "diffusers loading %s (device=%s dtype=%s)", identifier, device, dtype_name
        )
        pipe = auto_pipe.from_pretrained(identifier, torch_dtype=dtype)
        pipe = pipe.to(device)
        # Drop the safety checker if present: it perturbs timing and can blank
        # frames, which would corrupt a throughput measurement.
        if getattr(pipe, "safety_checker", None) is not None:
            pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:  # noqa: BLE001
            pass
        self._pipes[key] = pipe
        return pipe

    @staticmethod
    def _pick_device(torch: Any) -> tuple[str, list[str]]:
        try:
            if torch.backends.mps.is_available():
                return "mps", ["MPS", "Metal"]
        except Exception:  # noqa: BLE001
            pass
        try:
            if torch.cuda.is_available():
                return "cuda", ["CUDA"]
        except Exception:  # noqa: BLE001
            pass
        return "cpu", ["CPU"]

    @staticmethod
    def _pick_dtype(
        torch: Any, device: str, extras: dict[str, Any] | None
    ) -> tuple[str, Any]:
        override = (extras or {}).get("dtype")
        if override in ("float32", "fp32"):
            return "float32", torch.float32
        if override in ("float16", "fp16"):
            return "float16", torch.float16
        if override in ("bfloat16", "bf16"):
            return "bfloat16", torch.bfloat16
        # fp16 is unsupported / very slow for many ops on CPU; keep it in fp32.
        if device == "cpu":
            return "float32", torch.float32
        return "float16", torch.float16

    @staticmethod
    def _sync(torch: Any, device: str) -> None:
        try:
            if (
                device == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "synchronize")
            ):
                torch.mps.synchronize()
            elif device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _peak_memory_gb(torch: Any, device: str) -> float | None:
        try:
            if device == "mps" and hasattr(torch, "mps"):
                for fn in ("driver_allocated_memory", "current_allocated_memory"):
                    f = getattr(torch.mps, fn, None)
                    if callable(f):
                        return float(f()) / (1024**3)
            elif device == "cuda" and torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated()) / (1024**3)
        except Exception:  # noqa: BLE001
            pass
        return None


register_driver(_BACKEND_NAME, lambda: DiffusersDriver())


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    drv = DiffusersDriver()
    det = drv.detect()
    if not det.available:
        print(f"diffusers not available: {det.notes}")
    else:
        print(
            f"diffusers available: version={det.version} "
            f"flags={det.build_flags} runtime={det.runtime_versions}"
        )
        models = drv.list_models()
        print(f"Catalog + cached models ({len(models)}):")
        for m in models[:12]:
            dl = m.extras.get("downloaded")
            print(f"  - {m.identifier}  name={m.name} downloaded={dl}")
