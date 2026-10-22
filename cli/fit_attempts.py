"""fit_attempts records: one row per (model file, quant, context, GPU layers) attempt.

The shape is the D1 table `fit_attempts` (worker/migrations/0007_fit_attempts.sql,
read via GET /v1/fit-attempts), extended only with the Flint launch columns listed in
`FLINT_COLUMNS` (worker/migrations/0010_fit_attempts_flint.sql; the agent result route
refuses them until the server's route accepts them, so `cli.agent.runner` strips them).

Shared by:
  * the operator pod sweep (tools/fit_sweep.py, llama-bench lines -> `classify`)
  * the CLI fit sweep / `llm-speed bench --preset` (sweep attempts -> `from_sweep_attempt`)

Every string column passes through `cli.pathscrub.scrub_paths`, so no row (saved or
uploaded) carries a filesystem path. There is no public write route (no POST
/v1/fit-attempts): rows reach D1 only inside an `llm-speed agent` job result
(POST /v1/agent/jobs/:job_id/result, ops/AGENT_QUEUE_API.md section 5).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .pathscrub import scrub_paths

PAYLOAD_SCHEMA = "fit-attempts/1"

# Columns of the D1 table, in table order (minus id/created_at).
COLUMNS = [
    "sweep_id",
    "model_slug",
    "model_file",
    "hardware_slug",
    "gpu_name",
    "quant",
    "context_tokens",
    "n_gpu_layers",
    "n_layers",
    "cpu_offload",
    "status",
    "error_type",
    "error_detail",
    "decode_tps",
    "prefill_tps",
    "vram_total_mib",
    "vram_peak_mib",
    "file_bytes",
    "backend",
    "backend_version",
    "measured_at",
]

# Flint launch extensions (all nullable). See docs/fit_attempts_payload.md.
FLINT_COLUMNS = [
    "context_requested",  # the step's planned context; context_tokens is what really ran
    "n_cpu_moe",  # llama-server --n-cpu-moe (experts of the first N layers on the CPU)
    "ttft_ms",  # first streamed token of any kind
    "ttfa_ms",  # first answer token (after reasoning)
    "reasoning_tokens",  # estimated reasoning tokens before the answer
    "backend_build",  # "b11803" (null when the build number is not trustworthy)
    "backend_commit",  # short llama.cpp commit
    "driver",  # NVIDIA driver major ("580") or "macos-26"
    "memory_kind",  # "vram" | "unified"
    "memory_baseline_mib",  # memory in use before the model loaded
    "gpu_in_use_at_start_mib",  # other processes' GPU memory at start (never killed)
    "workload",  # CLI workload the tok/s came from
    "source",  # "llm-speed-cli/<version>" | "pod-llama-bench"
]

ALL_COLUMNS = COLUMNS + FLINT_COLUMNS
STATUSES = ("ok", "oom", "failed", "skipped")
ERROR_DETAIL_MAX = 200

OOM_PATTERNS = (
    "out of memory",
    "cudamalloc failed",
    "failed to allocate",
    "unable to allocate",
    "not enough memory",
    "alloc_buffer: allocating",
    "error_out_of_memory",
)

_SLUG_RE = re.compile(r"[^a-z0-9.]+")


def slugify(s: str | None) -> str:
    return _SLUG_RE.sub("-", (s or "").lower()).strip("-.")[:120]


def _clean(v: Any) -> Any:
    if isinstance(v, str):
        return scrub_paths(v)
    return v


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep known columns only, scrub every string, cap error_detail."""
    out = {k: _clean(row.get(k)) for k in ALL_COLUMNS if k in row}
    if out.get("model_file"):
        out["model_file"] = Path(str(out["model_file"]).replace("\\", "/")).name
    if isinstance(out.get("error_detail"), str):
        out["error_detail"] = out["error_detail"].strip()[:ERROR_DETAIL_MAX] or None
    if out.get("status") not in STATUSES:
        out["status"] = "failed"
    return out


# --- pod sweep (llama-bench) -> row. Moved from tools/fit_sweep.py in 0.0.8. -----------


def _lower(lines: Iterable[str]) -> str:
    return "\n".join(lines).lower()


def _oom_type(tail: list[str]) -> tuple[str, str] | None:
    text = _lower(tail)
    hit = next((ln for ln in tail if any(p in ln.lower() for p in OOM_PATTERNS)), None)
    if hit is None:
        return None
    ctx_markers = (
        "failed to create context",
        "llama_init_from_model",
        "kv",
        "compute buffer",
        "llama_context",
    )
    if any(m in text for m in ctx_markers) and "failed to load model" not in text:
        et = "oom_context"
    elif (
        "failed to load model" in text
        or "error loading model" in text
        or "model buffer" in text
    ):
        et = "oom_weights"
    else:
        et = "oom"
    return et, hit.strip()[:ERROR_DETAIL_MAX]


def classify(
    raw: dict[str, Any], *, hardware_slug: str, sweep_id: str
) -> dict[str, Any]:
    """One FIT_ATTEMPT record (tools/fit_sweep.py pod log) -> one fit_attempts row."""
    n_layers = raw.get("n_layers")
    ngl = raw.get("n_gpu_layers")
    row: dict[str, Any] = {
        "sweep_id": sweep_id,
        "model_slug": raw.get("model_slug") or "",
        "model_file": raw.get("model_file") or "",
        "hardware_slug": hardware_slug,
        "gpu_name": raw.get("gpu_name"),
        "quant": raw.get("quant") or "",
        "context_tokens": int(raw.get("context_tokens") or 0),
        "n_gpu_layers": ngl,
        "n_layers": n_layers,
        "cpu_offload": 1
        if (
            raw.get("offload")
            or (isinstance(ngl, int) and isinstance(n_layers, int) and ngl < n_layers)
        )
        else 0,
        "status": "failed",
        "error_type": None,
        "error_detail": None,
        "decode_tps": None,
        "prefill_tps": None,
        "vram_total_mib": raw.get("vram_total_mib"),
        "vram_peak_mib": raw.get("vram_peak_mib"),
        "file_bytes": raw.get("file_bytes"),
        "backend": "llama.cpp",
        "backend_version": None,
        "measured_at": raw.get("ts") or "",
    }
    kind = raw.get("kind")
    if kind == "skipped":
        row.update(
            status="skipped",
            error_type="weights_exceed_vram",
            error_detail=f"file {raw.get('file_bytes')} bytes > {raw.get('vram_total_mib')} MiB VRAM",
        )
        return row
    if kind == "download_failed":
        row.update(status="failed", error_type="download_failed")
        return row
    rows = raw.get("rows") or []
    tg = [r for r in rows if int(r.get("n_gen") or 0) > 0]
    pp = [
        r
        for r in rows
        if int(r.get("n_prompt") or 0) > 0 and int(r.get("n_gen") or 0) == 0
    ]
    if rows:
        row["backend_version"] = (
            rows[0].get("build_commit") and f"commit {rows[0]['build_commit']}"
        )
        if isinstance(rows[0].get("n_gpu_layers"), int):
            row["n_gpu_layers"] = rows[0]["n_gpu_layers"]
    if raw.get("exit") == 0 and tg:
        row.update(
            status="ok",
            decode_tps=round(float(tg[0].get("avg_ts") or 0), 3),
            prefill_tps=round(float(pp[0].get("avg_ts") or 0), 3) if pp else None,
        )
        if not row["decode_tps"]:
            row.update(status="failed", error_type="no_decode", decode_tps=None)
        return row
    tail = [str(x) for x in raw.get("stderr_tail") or []]
    text = _lower(tail)
    oom = _oom_type(tail)
    if oom is not None:
        row.update(status="oom", error_type=oom[0], error_detail=oom[1])
        return row
    if raw.get("exit") == -9 or "timeout" in text:
        row.update(error_type="timeout")
    elif "unknown model architecture" in text:
        row.update(error_type="unsupported_architecture")
    elif "failed to load model" in text or "error loading model" in text:
        row.update(error_type="load_failed")
    else:
        row.update(error_type="crash")
    last = next((ln for ln in reversed(tail) if ln.strip()), "")
    row["error_detail"] = last.strip()[:ERROR_DETAIL_MAX] or None
    return row


def parse_log(text: str, *, hardware_slug: str, sweep_id: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        if not line.startswith("FIT_ATTEMPT "):
            continue
        try:
            raw = json.loads(line[len("FIT_ATTEMPT ") :])
        except json.JSONDecodeError:
            continue
        out.append(classify(raw, hardware_slug=hardware_slug, sweep_id=sweep_id))
    return out


def _sql_value(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def to_sql(rows: list[dict[str, Any]]) -> str:
    """INSERT OR IGNORE statements for the base table columns (operator ingest)."""
    lines = []
    for r in rows:
        vals = ", ".join(_sql_value(r.get(c)) for c in COLUMNS)
        lines.append(
            f"INSERT OR IGNORE INTO fit_attempts ({', '.join(COLUMNS)}) VALUES ({vals});"
        )
    return "\n".join(lines) + ("\n" if lines else "")


# --- CLI sweep attempt -> row -----------------------------------------------------------

# fit_sweep status -> (fit_attempts status, error_type)
_STATUS_MAP: dict[str, tuple[str, str | None]] = {
    "ok": ("ok", None),
    "partial": ("ok", "partial"),
    "oom": ("oom", "oom"),
    "skipped_ram": ("skipped", "weights_exceed_memory"),
    "skipped_vram": ("skipped", "weights_exceed_vram"),
    "skipped_disk": ("skipped", "insufficient_disk"),
    "skipped_after_failure": ("skipped", "smaller_context_failed"),
    "skipped_backend": ("skipped", "backend_unavailable"),
    "missing": ("skipped", "model_missing"),
    "download_failed": ("failed", "download_failed"),
    "unsupported_model": ("failed", "unsupported_architecture"),
    "bad_server_args": ("failed", "bad_server_args"),
    "load_failed": ("failed", "load_failed"),
    "load_timeout": ("failed", "timeout"),
    "bench_timeout": ("failed", "timeout"),
    "cli_read_timeout": ("failed", "timeout"),
    "server_crashed": ("failed", "crash"),
    "bench_failed": ("failed", "bench_failed"),
}


def _driver_label(facts: dict[str, Any]) -> str | None:
    if facts.get("kind") == "nvidia" and facts.get("driver_version"):
        return str(facts["driver_version"]).split(".")[0]
    if facts.get("kind") == "apple" and facts.get("macos"):
        return "macos-" + str(facts["macos"]).split(".")[0]
    return None


def _first_result(rec: dict[str, Any]) -> dict[str, Any]:
    res = [r for r in rec.get("cli_results") or [] if not r.get("error")]
    return res[0] if res else ((rec.get("cli_results") or [{}])[0] or {})


def from_sweep_attempt(
    rec: dict[str, Any],
    facts: dict[str, Any],
    *,
    sweep_id: str,
    hardware_slug: str,
    cli_version: str | None = None,
) -> dict[str, Any]:
    """One `cli.fit_sweep` attempt record -> one fit_attempts row (+ Flint columns)."""
    status, error_type = _STATUS_MAP.get(
        str(rec.get("status")), ("failed", rec.get("status"))
    )
    r = _first_result(rec)
    mem = rec.get("memory") or {}
    llama = rec.get("llama_cpp") or {}
    mf = rec.get("model_file") or {}
    backend = rec.get("backend") or "llama.cpp"
    kind = facts.get("kind")
    # Error type refinement for OOM from the server log (weights vs context).
    if status == "oom":
        tail = [
            str(x) for x in (rec.get("server_log_tail") or rec.get("log_signals") or [])
        ]
        o = _oom_type(tail)
        if o:
            error_type = o[0]
    detail = None
    if status != "ok":
        detail = rec.get("note") or r.get("error") or mf.get("error")
        if not detail:
            tail = rec.get("server_log_tail") or rec.get("bench_log_tail") or []
            detail = next((ln for ln in reversed(tail) if str(ln).strip()), None)
    peak = (
        mem.get("peak_mib")
        if mem.get("peak_mib") is not None
        else r.get("peak_vram_mib")
    )
    ngl = rec.get("ngl") if backend == "llama.cpp" else None
    backend_version = r.get("backend_version")
    if (
        backend == "llama.cpp"
        and not backend_version
        and (llama.get("build") or llama.get("commit"))
    ):
        backend_version = " ".join(
            x
            for x in (
                llama.get("build"),
                llama.get("commit") and f"({llama['commit']})",
            )
            if x
        )
    row = {
        "sweep_id": sweep_id,
        "model_slug": slugify(rec.get("model") or rec.get("entry_id")),
        "model_file": rec.get("file") or "",
        "hardware_slug": hardware_slug,
        "gpu_name": facts.get("gpu") or facts.get("chip"),
        "quant": rec.get("quant") or "",
        # The context the attempt really ran with (server n_ctx / MLX tokens processed),
        # falling back to the planned step when nothing ran (load failures, skips).
        "context_tokens": int(rec.get("ctx_actual") or rec.get("ctx") or 0),
        "context_requested": int(rec.get("ctx") or 0) or None,
        "n_gpu_layers": ngl,
        "n_layers": None,
        "cpu_offload": 1 if rec.get("n_cpu_moe") else 0,
        "status": status,
        "error_type": error_type,
        "error_detail": detail,
        "decode_tps": r.get("decode_tps") if status == "ok" else None,
        "prefill_tps": r.get("prefill_tps") if status == "ok" else None,
        "vram_total_mib": round(facts["gpu_memory_total_mib"])
        if facts.get("gpu_memory_total_mib")
        else (
            round(facts["ram_gb"] * 1024)
            if kind == "apple" and facts.get("ram_gb")
            else None
        ),
        "vram_peak_mib": round(peak) if isinstance(peak, (int, float)) else None,
        "file_bytes": mf.get("bytes") or mf.get("expected_bytes"),
        "backend": backend,
        "backend_version": backend_version,
        "measured_at": rec.get("finished_at") or rec.get("started_at") or "",
        "n_cpu_moe": rec.get("n_cpu_moe"),
        "ttft_ms": r.get("ttft_ms") if status == "ok" else None,
        "ttfa_ms": r.get("time_to_first_answer_ms") if status == "ok" else None,
        "reasoning_tokens": r.get("reasoning_tokens_est") if status == "ok" else None,
        "backend_build": (llama.get("build") or r.get("llama_cpp_build"))
        if backend == "llama.cpp"
        else None,
        "backend_commit": (llama.get("commit") or r.get("llama_cpp_commit"))
        if backend == "llama.cpp"
        else None,
        "driver": _driver_label(facts),
        "memory_kind": "vram"
        if kind == "nvidia"
        else ("unified" if kind == "apple" else None),
        "memory_baseline_mib": round(mem["baseline_mib"])
        if isinstance(mem.get("baseline_mib"), (int, float))
        else None,
        "gpu_in_use_at_start_mib": rec.get("gpu_memory_in_use_at_start_mib"),
        "workload": r.get("workload"),
        "source": f"llm-speed-cli/{cli_version}" if cli_version else "llm-speed-cli",
    }
    return sanitize_row(row)
