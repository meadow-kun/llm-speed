"""Image-generation workloads — text-to-image throughput.

The image-gen analogue of `chat_short`: a frozen prompt, a fixed resolution and
step count, one warmup pass, then N timed generations. Where the LLM runner
reports decode tok/s, this reports **images per second** (higher is better — the
same polarity as tok/s, which is exactly why the property factory picked image-gen
as vertical #1).

The metric does not need a column: it is written into `WorkloadResult.extras`,
which the API's `workload_results.extras` JSON absorbs with no schema migration.
`seconds_per_image`, `steps_per_second`, and peak memory ride along in the same
dict.

Two workloads are registered:
  * ``txt2img-512``  — 512x512, 4 steps (turbo-friendly, GPU-light)
  * ``sdxl-1024``    — 1024x1024, 4 steps

Neither is in ``cli.config.DEFAULT_WORKLOADS`` (the LLM flagship suite); they
belong to the image-gen vertical's own suite.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import SUITE_VERSION
from ..registry import register_workload
from ..types import ModelRef, WorkloadResult

LOG = logging.getLogger("cli.workloads.txt2img")

# Frozen prompt for the image-gen suite. Editing it is a breaking change and
# requires bumping `cli.SUITE_VERSION` (same contract as the chat fixtures).
IMAGE_PROMPT = (
    "a photorealistic still life of a ripe orange, a glass of water, and a "
    "sprig of rosemary on a weathered wooden table, soft window light, shallow "
    "depth of field, high detail"
)


class Txt2ImgWorkload:
    """A fixed-resolution, fixed-step text-to-image throughput scenario.

    Runs ``num_warmup`` untimed generations (to pay one-time load/compile/cache
    costs) then ``num_timed`` timed ones, and reports the aggregate rate.
    """

    suite_version = SUITE_VERSION

    def __init__(
        self,
        name: str,
        *,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float = 0.0,
        num_warmup: int = 1,
        num_timed: int = 3,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.num_warmup = num_warmup
        self.num_timed = num_timed
        self.seed = seed

    def _failed(self, model: ModelRef, err: str) -> WorkloadResult:
        return WorkloadResult(
            workload=self.name,
            suite_version=SUITE_VERSION,
            backend=model.backend,
            backend_version=None,
            model=model,
            error=err,
        )

    def run(self, driver: Any, model: ModelRef) -> WorkloadResult:
        gen = getattr(driver, "generate_image", None)
        if not callable(gen):
            return self._failed(
                model,
                f"backend {model.backend!r} has no generate_image(); "
                "txt2img workloads require an image-gen driver (diffusers)",
            )

        # Warmup — first call also triggers the pipeline download/load.
        for _ in range(self.num_warmup):
            try:
                outcome = gen(
                    model,
                    IMAGE_PROMPT,
                    steps=self.steps,
                    width=self.width,
                    height=self.height,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed,
                    num_images=1,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("%s warmup raised", self.name)
                return self._failed(model, f"warmup driver-exception: {exc!r}")
            if not outcome.success:
                return self._failed(model, outcome.error or "warmup failed")

        total_wall_ms = 0.0
        total_images = 0
        peak_gb: float | None = None
        device: str | None = None
        dtype: str | None = None
        backend_extras: dict[str, Any] = {}

        for i in range(self.num_timed):
            try:
                outcome = gen(
                    model,
                    IMAGE_PROMPT,
                    steps=self.steps,
                    width=self.width,
                    height=self.height,
                    guidance_scale=self.guidance_scale,
                    seed=self.seed + 1 + i,  # vary seed so outputs differ
                    num_images=1,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("%s timed run raised", self.name)
                return self._failed(model, f"driver-exception: {exc!r}")
            if not outcome.success:
                return self._failed(model, outcome.error or "generation failed")

            total_wall_ms += outcome.wall_ms
            total_images += outcome.num_images
            if outcome.peak_memory_gb is not None:
                peak_gb = (
                    outcome.peak_memory_gb
                    if peak_gb is None
                    else max(peak_gb, outcome.peak_memory_gb)
                )
            device = outcome.device
            dtype = outcome.dtype
            backend_extras = outcome.backend_extras

        return self._build_result(
            model,
            total_wall_ms=total_wall_ms,
            total_images=total_images,
            peak_gb=peak_gb,
            device=device,
            dtype=dtype,
            backend_extras=backend_extras,
        )

    def _build_result(
        self,
        model: ModelRef,
        *,
        total_wall_ms: float,
        total_images: int,
        peak_gb: float | None,
        device: str | None,
        dtype: str | None,
        backend_extras: dict[str, Any],
    ) -> WorkloadResult:
        total_s = total_wall_ms / 1000.0
        images_per_second = total_images / total_s if total_s > 0 else None
        seconds_per_image = total_s / total_images if total_images > 0 else None
        total_steps = total_images * self.steps
        # End-to-end steps/s (includes text-encode + VAE decode). On async MPS a
        # per-step callback would only capture enqueue time, so a wall-clock,
        # synchronized figure is the honest one to report.
        steps_per_second = total_steps / total_s if total_s > 0 else None

        return WorkloadResult(
            workload=self.name,
            suite_version=SUITE_VERSION,
            backend=model.backend,
            backend_version=None,  # bench fills this from the driver's detect()
            model=model,
            batch_size=1,
            wall_ms=total_wall_ms,
            error=None,
            flags=[],
            extras={
                # The headline metric for this vertical (higher is better).
                "metric": "images_per_second",
                "images_per_second": images_per_second,
                "seconds_per_image": seconds_per_image,
                "steps_per_second": steps_per_second,
                "steps_per_second_basis": "end_to_end_wall",
                "peak_memory_gb": peak_gb,
                # Suite parameters (part of what makes runs comparable).
                "num_inference_steps": self.steps,
                "width": self.width,
                "height": self.height,
                "guidance_scale": self.guidance_scale,
                "num_warmup_runs": self.num_warmup,
                "num_timed_runs": self.num_timed,
                "images_generated": total_images,
                # Environment / provenance.
                "device": device,
                "dtype": dtype,
                "backend_extras": dict(backend_extras) if backend_extras else {},
            },
        )


register_workload(
    "txt2img-512",
    lambda: Txt2ImgWorkload("txt2img-512", width=512, height=512, steps=4),
)
register_workload(
    "sdxl-1024",
    lambda: Txt2ImgWorkload("sdxl-1024", width=1024, height=1024, steps=4),
)


if __name__ == "__main__":
    wl = Txt2ImgWorkload("txt2img-512", width=512, height=512, steps=4)
    print(f"workload={wl.name} {wl.width}x{wl.height} steps={wl.steps}")
    print(f"prompt: {IMAGE_PROMPT!r}")
