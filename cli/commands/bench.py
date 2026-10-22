"""`llm-speed bench` — the main user flow.

Default order (privacy-first reframe):
  1. Run benchmarks.
  2. Save to ~/.cache/llm-speed/runs/<isoformat>-<fp_hash>.json (the local artifact;
     always written unless --dry-run or --json picks a different target).
  3. If consent given AND not --no-upload AND not --dry-run AND not --json: upload.
  4. Print "Saved locally: <path>" always; print "Submitted: <url>" only on success.

Special modes:
  - --resume PATH: skip everything, verify signature on PATH, POST as-is.
  - --dry-run: run workloads, print would-be payload, no save, no upload.
  - --print-payload: print payload to stdout (works alongside or without --dry-run).
  - --strict-anon: ephemeral keypair per run, no fingerprint_hash sent, no
    persistent identity. Implies --anon. Skips the consent prompt (the user
    explicitly chose the most-private option).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from .. import SUITE_VERSION, __version__
from ..config import DEFAULT_API_BASE, DEFAULT_WORKLOADS, QUICK_WORKLOADS
from ..consent import has_consented, prompt_for_consent
from ..fingerprint import detect_fingerprint
from ..launch_telemetry import (
    FAILURE_SKIPPED,
    NvidiaVramSampler,
    classify_failure,
    is_load_failure,
    parse_server_args,
    round_mib,
)
from ..output import get_console, print_error, print_run_summary, print_warning
from ..registry import all_driver_names, all_workload_names, get_driver, get_workload
from ..signing import sign_report_jws as _sign_validate
from ..types import ModelRef, RunReport, WorkloadResult
from ..upload import (
    build_upload_payload,
    save_offline,
    upload_or_save,
    upload_saved_payload,
    warn_if_insecure_api_base,
)
from ..upload_guard import offline, upload_block_reason

log = logging.getLogger("cli.commands.bench")


# Auto-pick order: prefer fast local backends, hosted-api as last resort.
_BACKEND_PRIORITY = ("vllm", "llama.cpp", "ollama", "mlx", "exllamav2", "hosted-api")


def _pick_backend(explicit: str | None) -> str | None:
    available = all_driver_names()
    if explicit:
        if explicit not in available:
            print_error(
                f"backend {explicit!r} is not registered. registered: {available}"
            )
            return None
        return explicit
    if not available:
        return None

    # Probe each candidate in priority order; pick first available local one.
    available_set = set(available)
    ordered = [b for b in _BACKEND_PRIORITY if b in available_set]
    # Append any registered backends not in the priority list.
    ordered += [b for b in available if b not in _BACKEND_PRIORITY]

    fallback: str | None = None
    for name in ordered:
        try:
            drv = get_driver(name)
            det = drv.detect()
        except Exception as exc:
            log.debug("detect %s failed: %s", name, exc)
            continue
        if det.available:
            if name == "hosted-api":
                # Save it as a fallback only.
                fallback = fallback or name
                continue
            return name
    return fallback


def _resolve_model(driver, explicit: str | None) -> ModelRef | None:
    if explicit:
        # Build a minimal ModelRef. Drivers that need more should resolve via list_models.
        try:
            models = driver.list_models()
        except Exception as exc:
            log.debug("list_models failed: %s", exc)
            models = []
        for m in models:
            if explicit in (m.identifier, m.name):
                return m
        # Couldn't match in catalog; trust the user, but never use a path as the
        # display name (0.0.8 privacy fix: model.name used to carry the full path).
        return _model_ref_from_path(getattr(driver, "name", "?"), explicit)

    try:
        models = driver.list_models()
    except Exception as exc:
        print_error(f"driver list_models() failed: {exc}")
        return None
    if not models:
        print_error(
            f"no models available for backend {getattr(driver, 'name', '?')}; "
            "specify one with --model or install a model for that backend."
        )
        return None
    return models[0]


def _model_ref_from_path(backend: str, explicit: str) -> ModelRef:
    from pathlib import PurePath

    from ..pathscrub import looks_like_path

    is_path = looks_like_path(explicit) or explicit.lower().endswith(".gguf")
    base = PurePath(explicit.replace("\\", "/")).name if is_path else explicit
    name, quant = base, None
    if base.lower().endswith(".gguf"):
        from ..drivers.llama_cpp import _quant_from_name, _strip_quant

        name, quant = _strip_quant(base) or base, _quant_from_name(base)
    elif is_path and backend == "mlx":
        from ..drivers.mlx import _parse_name_size_quant

        name, _size, quant = _parse_name_size_quant(base)
    return ModelRef(backend=backend, identifier=explicit, name=name, quant=quant)


def _select_workloads(arg_workload: str | None, quick: bool) -> list[str]:
    if arg_workload:
        return [w.strip() for w in arg_workload.split(",") if w.strip()]
    if quick:
        return list(QUICK_WORKLOADS)
    return list(DEFAULT_WORKLOADS)


def _failed_result(
    name: str, model: ModelRef, backend: str, err: str
) -> WorkloadResult:
    return WorkloadResult(
        workload=name,
        suite_version=SUITE_VERSION,
        backend=backend,
        backend_version=None,
        model=model,
        error=err,
    )


def _llama_args(args) -> list[str]:
    raw: list[str] = []
    env = os.environ.get("LLM_SPEED_LLAMA_ARGS", "").strip()
    if env:
        raw.append(env)
    raw += list(getattr(args, "llama_args", None) or [])
    return parse_server_args(raw)


def _start_memory_probe(driver, fp) -> NvidiaVramSampler | None:
    """Reset the driver's own peak counter (MLX) and/or start sampling nvidia-smi."""
    reset = getattr(driver, "reset_peak_memory", None)
    if callable(reset):
        try:
            reset()
        except Exception as exc:  # noqa: BLE001
            log.debug("reset_peak_memory failed: %s", exc)
        return None
    has_nvidia = any(getattr(g, "kind", "") == "nvidia" for g in (fp.gpus or []))
    if has_nvidia and NvidiaVramSampler.available():
        try:
            return NvidiaVramSampler().start()
        except Exception as exc:  # noqa: BLE001
            log.debug("vram sampler failed to start: %s", exc)
    return None


def _finish_memory_probe(driver, sampler: NvidiaVramSampler | None) -> dict:
    if sampler is not None:
        peak = sampler.stop()
        if peak is not None:
            return {
                "peak_vram_mib": round_mib(peak),
                "peak_vram_source": NvidiaVramSampler.SOURCE,
            }
        return {}
    getter = getattr(driver, "peak_memory_mib", None)
    if callable(getter):
        try:
            peak = getter()
        except Exception as exc:  # noqa: BLE001
            log.debug("peak_memory_mib failed: %s", exc)
            peak = None
        if peak:
            return {
                "peak_vram_mib": round_mib(peak),
                "peak_vram_source": "mlx-peak-memory",
            }
    return {}


def _attach_launch_telemetry(res: WorkloadResult, driver, model, mem: dict) -> None:
    """Fold 0.0.8 launch fields into res.extras (see launch_telemetry.UPLOADED_EXTRAS_KEYS)."""
    extras = dict(res.extras or {})
    extras.update(mem)
    pop = getattr(driver, "pop_run_telemetry", None)
    telemetry = pop() if callable(pop) else []
    # Only single-stream workloads have one unambiguous "first answer".
    if res.error is None and len(telemetry) == 1:
        extras.update(telemetry[0])
    labels = getattr(driver, "run_labels", None)
    if callable(labels):
        try:
            extras.update(labels(model))
        except Exception as exc:  # noqa: BLE001
            log.debug("run_labels failed: %s", exc)
    if res.error:
        extras["failure_kind"] = extras.get("failure_kind") or classify_failure(
            res.error
        )
    res.extras = extras


def _handle_resume(args) -> int:
    """--resume PATH: read a saved offline run, verify, upload."""
    console = get_console()
    path = Path(args.resume)
    if not path.exists():
        print_error(f"resume path not found: {path}")
        return 1
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print_error(f"failed to read {path}: {exc}")
        return 1

    api_base = getattr(args, "api_base", None) or DEFAULT_API_BASE
    api_key = getattr(args, "api_key", None)
    anon = getattr(args, "anon", False) or getattr(args, "strict_anon", False)

    blocked = (
        "--no-upload"
        if getattr(args, "no_upload", False)
        else upload_block_reason(api_base)
    )
    if blocked:
        print_error(f"not uploading {path.name}: {blocked}")
        return 1
    if not anon:
        warn_if_insecure_api_base(api_base, api_key)

    try:
        url = upload_saved_payload(
            saved,
            api_base=api_base,
            api_key=api_key,
            anon=anon,
            strict_anon=getattr(args, "strict_anon", False),
        )
    except Exception as exc:
        print_error(f"resume upload failed: {exc}")
        return 1

    console.print(f"[green]Submitted:[/green] {url}")
    return 0


def cmd_bench(args) -> int:
    # --resume short-circuits everything else.
    if getattr(args, "resume", None):
        return _handle_resume(args)
    # --preset NAME: guided multi-step run (cli/commands/preset.py).
    if getattr(args, "preset", None):
        from .preset import cmd_preset

        try:
            return cmd_preset(args)
        except KeyboardInterrupt:
            print_error(
                f"interrupted; rerun `llm-speed bench --preset {args.preset}` to resume"
            )
            return 130

    console = get_console()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    strict_anon = getattr(args, "strict_anon", False)
    dry_run = getattr(args, "dry_run", False)
    print_payload = getattr(args, "print_payload", False)
    json_path = getattr(args, "json", None)
    no_upload = (
        getattr(args, "no_upload", False)
        or json_path is not None
        or dry_run
        or offline()
    )
    api_base = getattr(args, "api_base", None) or DEFAULT_API_BASE
    api_key = getattr(args, "api_key", None)
    # --strict-anon implies --anon for the soft headers fallback path.
    anon = getattr(args, "anon", False) or strict_anon

    # Warn loudly (but don't block) if the bearer would travel over plaintext.
    # Skipped under any anon mode because those paths don't send Authorization.
    if not anon and not no_upload:
        warn_if_insecure_api_base(api_base, api_key)

    # 1. Fingerprint.
    try:
        fp = detect_fingerprint()
    except Exception as exc:
        print_error(f"fingerprint failed: {exc}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1

    # 2. Backend.
    explicit_backend = getattr(args, "backend", None)
    backend_name = _pick_backend(explicit_backend)
    if backend_name == "hosted-api" and not explicit_backend:
        # Auto-pick fell back to hosted-api: tell the user which provider URL
        # their prompts + API key will hit. They already gave us the API key
        # (so consent to send-to-provider is implied), but the auto-pick path
        # would otherwise be a silent surprise.
        try:
            from ..drivers.hosted_api import PROVIDERS, _enabled_providers

            providers = _enabled_providers()
            for p in providers:
                _, base_url, _ = PROVIDERS[p]
                sys.stderr.write(
                    f"Backend: hosted-api via {p} -- your benchmark prompts "
                    f"and API key will be sent to {base_url}. "
                    f"Pass --backend explicitly to choose a local backend.\n"
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("hosted-api auto-pick notice failed: %s", exc)
    if not backend_name:
        registered = all_driver_names()
        if not registered:
            print_warning("no backends registered yet - let's check what's needed.")
        else:
            print_warning("no backend is ready on this machine yet - let's set one up.")
        # Hand off to the doctor: interactive install on a TTY, structured
        # guidance + non-zero exit on a non-TTY. If it makes the host ready,
        # re-pick and continue the benchmark.
        from .doctor import run_doctor

        rc = run_doctor(api_base=api_base)
        if rc != 0:
            return rc
        backend_name = _pick_backend(explicit_backend)
        if not backend_name:
            print_error(
                "still no usable backend after setup. run `llm-speed detect` to inspect."
            )
            return 1

    try:
        driver = get_driver(backend_name)
    except KeyError as exc:
        print_error(str(exc))
        return 1

    # llama-server passthrough args (e.g. --n-cpu-moe 20 for MoE expert offload).
    server_args = _llama_args(args)
    if server_args:
        if hasattr(driver, "server_args"):
            driver.server_args = server_args
        else:
            print_warning(
                f"--llama-args ignored: backend {backend_name!r} does not take llama-server args"
            )

    # Try to capture backend version from detection.
    backend_version: str | None = None
    try:
        det = driver.detect()
        backend_version = det.version
    except Exception as exc:
        log.debug("detect failed for %s: %s", backend_name, exc)

    # 3. Model.
    model = _resolve_model(driver, getattr(args, "model", None))
    if model is None:
        # Backend present but no usable model. Route through the doctor to
        # offer a model pull (ollama) or print guidance, then retry once.
        from .doctor import run_doctor

        rc = run_doctor(api_base=api_base)
        if rc != 0:
            return rc
        backend_name = _pick_backend(explicit_backend) or backend_name
        try:
            driver = get_driver(backend_name)
        except KeyError as exc:
            print_error(str(exc))
            return 1
        model = _resolve_model(driver, getattr(args, "model", None))
        if model is None:
            return 1

    # 4. Workloads.
    workload_names = _select_workloads(
        getattr(args, "workload", None), getattr(args, "quick", False)
    )
    registered_workloads = set(all_workload_names())
    if not registered_workloads:
        print_error("no workloads registered. cannot run benchmarks.")
        return 1
    missing = [w for w in workload_names if w not in registered_workloads]
    if missing:
        print_warning(
            f"unknown workloads (skipped): {missing}; registered: {sorted(registered_workloads)}"
        )
        workload_names = [w for w in workload_names if w in registered_workloads]
    if not workload_names:
        print_error("no runnable workloads selected.")
        return 1

    # 5. Run each workload.
    results: list[WorkloadResult] = []
    console.print(
        f"[bold]Running[/bold] {len(workload_names)} workloads on [cyan]{backend_name}[/cyan] / [magenta]{model.name or model.identifier}[/magenta]"
    )
    load_failure: WorkloadResult | None = None
    for wname in workload_names:
        console.print(f"  - {wname} ...", end="")
        if load_failure is not None:
            # The model could not be loaded (OOM / unsupported arch / timeout):
            # every later workload would fail the same way after another long
            # load attempt. Record them as skipped instead.
            skipped = _failed_result(
                wname,
                model,
                backend_name,
                f"skipped: model failed to load ({(load_failure.extras or {}).get('failure_kind')})",
            )
            skipped.extras = {"failure_kind": FAILURE_SKIPPED}
            results.append(skipped)
            console.print(" [yellow]skipped[/yellow] (model failed to load)")
            continue
        reset_tel = getattr(driver, "reset_run_telemetry", None)
        if callable(reset_tel):
            reset_tel()
        sampler = _start_memory_probe(driver, fp)
        try:
            wl = get_workload(wname)
            try:
                res = wl.run(driver, model)
            finally:
                mem = _finish_memory_probe(driver, sampler)
                sampler = None
            if not isinstance(res, WorkloadResult):
                # Defensive: a workload returned something weird.
                res = _failed_result(
                    wname,
                    model,
                    backend_name,
                    f"workload returned {type(res).__name__}, expected WorkloadResult",
                )
            # Workloads may not have access to the detected backend version;
            # fill it in here so summary tables and uploaded JSON show it.
            if res.backend_version is None and backend_version is not None:
                res.backend_version = backend_version
            _attach_launch_telemetry(res, driver, model, mem)
            if is_load_failure((res.extras or {}).get("failure_kind")):
                load_failure = res
            results.append(res)
            if res.error:
                console.print(f" [red]error:[/red] {res.error[:80]}")
            else:
                console.print(
                    f" [green]ok[/green]  decode={res.decode_tps or 0:.1f} tps"
                )
        except Exception as exc:
            log.debug("workload %s raised: %s", wname, exc, exc_info=True)
            if getattr(args, "verbose", False):
                traceback.print_exc()
            if sampler is not None:
                sampler.stop()
            failed = _failed_result(
                wname, model, backend_name, f"{type(exc).__name__}: {exc}"
            )
            failed.extras = {"failure_kind": classify_failure(failed.error)}
            results.append(failed)
            console.print(f" [red]exception:[/red] {exc}")

    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()

    # 6. Build + sign report.
    report = RunReport(
        suite_version=SUITE_VERSION,
        cli_version=__version__,
        fingerprint=fp,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    # 6a. Sign-once sanity check: produce a JWS to validate the keypair + payload
    # invariants (privacy, etc) up-front so an invalid run fails before save/upload.
    # save_offline / upload_or_save will sign their own JWS from the same payload.
    try:
        _sign_validate(report, strict_anon=strict_anon)
    except Exception as exc:
        print_error(f"signing failed: {exc}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1

    # --print-payload: emit the upload body (a `{"jws": "..."}` envelope).
    if print_payload:
        payload = build_upload_payload(report, strict_anon=strict_anon)
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        sys.stdout.flush()

    # 7. Local save + optional upload.
    result_url: str | None = None
    offline_path: Path | None = None

    if dry_run:
        # Run only; no save, no upload.
        pass
    elif json_path is not None:
        # Explicit local save target. No upload.
        offline_path = save_offline(report, Path(json_path), strict_anon=strict_anon)
    else:
        # Default: always save locally.
        offline_path = save_offline(report, strict_anon=strict_anon)

        has_measurement = any(r.error is None for r in results)
        # --upload-failures (opt-in): publish a run whose only content is a
        # classified load failure (OOM / unsupported arch / load timeout), so
        # "does not fit" is a recorded result rather than a missing one.
        upload_failures = getattr(args, "upload_failures", False) and any(
            is_load_failure((r.extras or {}).get("failure_kind")) for r in results
        )
        if not no_upload and not has_measurement and not upload_failures:
            # Never publish a run where every workload errored (e.g. the model
            # failed to load): it carries no measurement and would only create an
            # aggregation-excluded orphan on the API. The local save above keeps
            # the failure inspectable.
            console.print(
                "[yellow]Skipping upload:[/yellow] every workload errored "
                "(no measurement to publish). Saved locally for inspection."
            )
        elif not no_upload and upload_block_reason(api_base):
            console.print(
                f"[yellow]Not uploading:[/yellow] {upload_block_reason(api_base)}. Result saved locally."
            )
        elif not no_upload:
            # Consent gate. Strict-anon implicitly consents (most private option).
            consent_ok = strict_anon or has_consented()
            if not consent_ok:
                fp_hash_for_record = (
                    report.fingerprint.fingerprint_hash if not strict_anon else ""
                )
                consent_ok = prompt_for_consent(
                    api_base=api_base,
                    cli_version=__version__,
                    fingerprint_hash=fp_hash_for_record,
                )

            if consent_ok:
                result_url, fallback_path = upload_or_save(
                    report,
                    api_base=api_base,
                    api_key=api_key,
                    anon=anon,
                    strict_anon=strict_anon,
                )
                # If upload failed and upload_or_save fell back to a save, prefer
                # the originally-saved path for the message (less confusing).
                if fallback_path is not None and offline_path is None:
                    offline_path = fallback_path
            else:
                console.print(
                    "[yellow]Skipping upload (no consent).[/yellow] Result saved locally."
                )

    # 8. Pretty-print.
    if not dry_run:
        print_run_summary(report, result_url=result_url, offline_path=offline_path)
    else:
        console.print(
            "[yellow]--dry-run:[/yellow] benchmark complete; nothing saved or uploaded."
        )

    # 9. Share affordances: only on a successful upload AND only when stdin
    #    is a TTY (so CI logs stay clean). Wrapped in a broad try/except
    #    because failure to print the share block must never affect the
    #    benchmark exit code.
    if result_url and not dry_run and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from ..ui.share import interactive_share_prompt

            interactive_share_prompt(result_url)
        except Exception as exc:  # noqa: BLE001
            log.debug("share prompt failed: %s", exc)

    # Exit code: 1 if every workload errored, else 0.
    if results and all(r.error for r in results):
        return 1
    return 0
