"""`llm-speed bench --preset <name>`: one pasteable command per machine.

    curl -fsSL https://llm-speed.com/install.sh | sh && llm-speed bench --preset flint-40b

Stages:
  1. hardware   detect NVIDIA (nvidia-smi) or Apple Silicon; pick the backend(s) and
                the quants that fit this machine's memory (cli/presets/<name>.json)
  2. backends   find llama-server (PATH, Homebrew, common build dirs) and require
                build >= the preset's minimum; offer the exact install command (y/N,
                no sudo, nothing installed silently); check mlx-lm on Macs
  3. benchmark  cli.fit_sweep: free-disk check before each download, resumable HF
                download with progress (reuses partial files and the HF cache), one
                llama-server per (file, context) on 4K -> 16K -> 32K, stop climbing at
                the first failure; every OOM / failed load / skip is a result record.
                Rerun = resume (completed steps are skipped). Ctrl-C stops servers and
                keeps finished steps. A model is deleted after its steps only if this
                preset downloaded it AND the next one needs the space.
  4. results    fit_attempts rows (cli/fit_attempts.py) saved locally, summary table,
                upload through the normal consent flow (--strict-anon honoured):
                successful runs to /v1/results. fit_attempts rows stay local: the only
                write path is an `llm-speed agent` job result (no POST /v1/fit-attempts).
"""

from __future__ import annotations

import datetime as dt
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import __version__
from .. import fit_sweep as fs
from .. import sampling as sampling_mod
from ..config import CACHE_DIR, DEFAULT_API_BASE
from ..fit_attempts import from_sweep_attempt

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"
MIB = 1024 * 1024


def say(msg: str = "") -> None:
    print(msg, flush=True)


def stage(n: int, total: int, title: str) -> None:
    say(f"\n== [{n}/{total}] {title}")


# ----------------------------------------------------------------- preset data


def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def canonical_bytes(obj: Any) -> bytes:
    """Canonical serialization used for review-then-publish: keys sorted at every level,
    UTF-8 (no ASCII escaping), no insignificant whitespace."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def preset_sha256(preset: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_bytes(preset)).hexdigest()


def cmd_preset_tool(args) -> int:
    """`llm-speed preset hash <name>` / `llm-speed preset list` / `llm-speed preset show <name>`."""
    action = getattr(args, "preset_action", None)
    if action == "list":
        for n in available_presets():
            p = load_preset(n)
            print(f"{n}\tversion {p.get('version')}\tsha256 {preset_sha256(p)}")
        return 0
    if not getattr(args, "name", None):
        print("usage: llm-speed preset {hash,show,list} [NAME]", file=sys.stderr)
        return 2
    p = load_preset(args.name)
    if action == "hash":
        print(preset_sha256(p))
        return 0
    if action == "show":
        sys.stdout.write(canonical_bytes(p).decode("utf-8") + "\n")
        return 0
    print("usage: llm-speed preset {hash,show,list} [NAME]", file=sys.stderr)
    return 2


def load_preset(name: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"unknown preset {name!r}; available: {', '.join(available_presets()) or 'none'}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def memory_budget_mib(preset: dict[str, Any], facts: dict[str, Any]) -> float | None:
    """Memory a model file may occupy on this machine (weights + KV need headroom)."""
    head = preset.get("memory_headroom_mib", {})
    if facts.get("kind") == "nvidia" and facts.get("gpu_memory_total_mib"):
        return float(facts["gpu_memory_total_mib"]) - float(head.get("nvidia", 1536))
    if facts.get("kind") == "apple" and facts.get("ram_gb"):
        wired = facts.get("iogpu_wired_limit_mb") or 0
        gpu_mib = (
            float(wired)
            if wired
            else facts["ram_gb"]
            * 1024
            * float(preset.get("apple_gpu_memory_fraction", 0.75))
        )
        return gpu_mib - float(head.get("apple", 4096))
    return None


def build_plan(
    preset: dict[str, Any],
    facts: dict[str, Any],
    *,
    with_control: bool = False,
    models_dir: Path,
    mlx_dir: Path,
    llama_server: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Preset + detected hardware -> a cli.fit_sweep plan. Returns (plan, notes)."""
    kind = facts.get("kind")
    profile = (preset.get("profiles") or {}).get(kind or "")
    if not profile:
        raise SystemExit(
            f"preset {preset['preset']} supports {', '.join(preset.get('profiles', {}))}; "
            f"this machine looks like '{kind}' (no NVIDIA GPU via nvidia-smi, not Apple Silicon)"
        )
    models = preset["models"]
    budget = memory_budget_mib(preset, facts)
    gpu = facts.get("gpu") or facts.get("chip") or ""
    notes: list[str] = []
    skip_status = "skipped_vram" if kind == "nvidia" else "skipped_ram"
    mem_label = "VRAM" if kind == "nvidia" else "GPU-usable unified memory"

    def make(eid: str) -> dict[str, Any] | None:
        m = dict(models[eid])
        m["id"] = eid
        rx = m.get("requires_gpu_regex")
        if rx and not re.search(rx, gpu):
            notes.append(
                f"{eid}: not on this GPU ({m.get('requires_gpu_why', 'unsupported')})"
            )
            return None
        override = (profile.get("contexts") or {}).get(eid)
        if override is not None:
            m["contexts"] = list(override)
        m.setdefault("contexts", list(preset["contexts"]))
        size_mib = (m.get("bytes") or 0) / MIB
        if budget is not None and size_mib > budget:
            m["skip_status"] = skip_status
            m["skip_note"] = (
                f"file {size_mib / 1024:.1f} GiB > {budget / 1024:.1f} GiB {mem_label} available for weights"
            )
        return m

    entries = [e for e in (make(i) for i in profile.get("entries", [])) if e]
    if not any(not e.get("skip_status") for e in entries):
        for fid in profile.get("fallback", []):
            e = make(fid)
            if e:
                entries.append(e)
                notes.append(f"{fid}: added because the main quants do not fit")
        probe = profile.get("offload_probe")
        if probe and not any(not e.get("skip_status") for e in entries):
            e = dict(models[probe["entry"]])
            e.update(
                id=f"{probe['entry']}-n-cpu-moe-{probe['n_cpu_moe']}",
                n_cpu_moe=probe["n_cpu_moe"],
                contexts=probe.get("contexts", [4096]),
            )
            entries.append(e)
            notes.append(
                f"nothing fits fully on the GPU: probing CPU expert offload "
                f"(--n-cpu-moe {probe['n_cpu_moe']}) with {probe['entry']}"
            )
    if with_control:
        entries += [e for e in (make(i) for i in profile.get("control", [])) if e]
    for e in entries:
        if e.get("skip_status"):
            notes.append(
                f"{e['id']}: {e['skip_note']} (recorded as {e['skip_status']})"
            )
    plan = {
        "schema": "fit-plan/1",
        "launch": preset.get("launch"),
        "preset": preset["preset"],
        "machine": "auto",
        "models_dir": str(models_dir),
        "mlx_models_dir": str(mlx_dir),
        "llama_server": llama_server or "llama-server",
        "min_llama_build": preset.get("min_llama_build"),
        "ngl": 999,
        "health_timeout_s": preset.get("health_timeout_s", 900),
        "bench_timeout_s": preset.get("bench_timeout_s", 3600),
        "disk_margin_gb": preset.get("disk_margin_gb", 3),
        "server_args": preset.get("server_args", fs.DEFAULT_SERVER_ARGS),
        # 0.0.10: sent with every request by each step's `llm-speed bench`
        # (per-model "sampling" and --temperature/... flags override key by key).
        "sampling": sampling_mod.merge(preset.get("sampling", fs.DEFAULT_SAMPLING)),
        "workloads_by_ctx": preset.get("workloads_by_ctx", fs.DEFAULT_WORKLOADS_BY_CTX),
        "entries": entries,
    }
    return plan, notes


# ----------------------------------------------------------------- llama-server


def llama_server_candidates(explicit: str | None = None) -> list[str]:
    home = Path.home()
    c: list[str] = []
    for v in (explicit, os.environ.get("LLAMA_SERVER"), shutil.which("llama-server")):
        if v:
            c.append(str(Path(v).expanduser()))
    c += [
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
        str(home / ".local/bin/llama-server"),
        str(home / "llama.cpp/build/bin/llama-server"),
    ]
    c += sorted(
        glob.glob(str(home / "llama.cpp*/build/bin/llama-server")), reverse=True
    )
    c += sorted(
        glob.glob(str(home / "src/llama.cpp*/build/bin/llama-server")), reverse=True
    )
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_llama_server(
    explicit: str | None, min_build: int | None
) -> tuple[str | None, dict | None, list[str]]:
    """First candidate that runs and meets the minimum build; else the best found."""
    tried: list[str] = []
    best: tuple[str, dict] | None = None
    for cand in llama_server_candidates(explicit):
        if not (Path(cand).is_file() and os.access(cand, os.X_OK)):
            continue
        info = fs.llama_version(cand)
        bn = info.get("build_number")
        tried.append(f"{cand}: build {info.get('build') or bn or 'unknown'}")
        if min_build and bn and bn >= int(min_build):
            return cand, info, tried
        if best is None:
            best = (cand, info)
    if best:
        return best[0], best[1], tried
    return None, None, tried


def llama_install_command(
    facts: dict[str, Any], tag: str
) -> tuple[list[str] | None, str]:
    """(command to run after y/N, human text). Never sudo."""
    if facts.get("kind") == "apple":
        if shutil.which("brew"):
            return [
                "brew",
                "install",
                "llama.cpp",
            ], "brew install llama.cpp   (or: brew upgrade llama.cpp)"
        return None, "install Homebrew (https://brew.sh), then: brew install llama.cpp"
    dest = Path.home() / f"llama.cpp-{tag}"
    script = (
        f"git clone https://github.com/ggml-org/llama.cpp {dest} && "
        f"git -C {dest} checkout {tag} && "
        f"cmake -S {dest} -B {dest}/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && "
        f"cmake --build {dest}/build --target llama-server -j"
    )
    if not (shutil.which("git") and shutil.which("cmake") and shutil.which("nvcc")):
        return None, (
            script
            + "\n  (needs git, cmake and the CUDA toolkit's nvcc; full clone, not --depth 1, "
            "so the build number is real)"
        )
    return ["bash", "-c", script], script


def ask_yes_no(question: str, *, assume_yes: bool = False) -> bool:
    if assume_yes:
        say(f"{question} [y/N] y (--yes)")
        return True
    if not sys.stdin.isatty():
        say(f"{question} [y/N] -> no (not an interactive terminal)")
        return False
    try:
        ans = input(f"{question} [y/N] ")
    except (EOFError, KeyboardInterrupt):
        say("")
        return False
    return ans.strip().lower() in ("y", "yes")


# ----------------------------------------------------------------- results


def results_dir(preset_name: str) -> Path:
    d = CACHE_DIR / "presets" / preset_name
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def fit_rows(
    sweep: fs.Sweep, preset_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = sweep.load_results()
    attempts = [
        a for a in data.get("attempts", []) if a.get("machine") == sweep.machine
    ]
    sweep_id = f"{preset_name}-{sweep.machine}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')}"
    rows = [
        from_sweep_attempt(
            a,
            sweep.facts,
            sweep_id=sweep_id,
            hardware_slug=sweep.machine,
            cli_version=__version__,
        )
        for a in attempts
    ]
    return attempts, rows


def latest_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        out[(r["model_file"], r["quant"], r["context_tokens"], r.get("n_cpu_moe"))] = r
    return list(out.values())


def summary_table(rows: list[dict[str, Any]]) -> str:
    hdr = [
        "model file",
        "backend",
        "ctx",
        "status",
        "decode tok/s",
        "answer after",
        "peak MiB",
    ]
    body = []
    for r in rows:
        status = r["status"] + (f" ({r['error_type']})" if r.get("error_type") else "")
        moe = f" +n-cpu-moe {r['n_cpu_moe']}" if r.get("n_cpu_moe") else ""
        body.append(
            [
                (r["model_file"] or "-") + moe,
                r.get("backend") or "-",
                f"{r['context_tokens'] // 1024}K" if r.get("context_tokens") else "-",
                status,
                f"{r['decode_tps']:.1f}" if r.get("decode_tps") else "-",
                f"{r['ttfa_ms'] / 1000:.1f}s" if r.get("ttfa_ms") is not None else "-",
                str(r.get("vram_peak_mib") or "-"),
            ]
        )
    w = [
        max(len(h), *(len(b[i]) for b in body)) if body else len(h)
        for i, h in enumerate(hdr)
    ]
    lines = [
        "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)),
        "  ".join("-" * x for x in w),
    ]
    lines += ["  ".join(c.ljust(w[i]) for i, c in enumerate(b)) for b in body]
    return "\n".join(lines)


def _uploads_path(rdir: Path) -> Path:
    return rdir / "uploads.json"


def _upload_api_base(args) -> str:
    # Read $LLM_SPEED_API now, not at import time, so a test/staging override always wins.
    return (
        getattr(args, "api_base", None)
        or os.environ.get("LLM_SPEED_API")
        or DEFAULT_API_BASE
    )


def _upload_api_base_source(args) -> str:
    from ..api_host import api_base_source

    return api_base_source(args)


# Security Desk fix 3 (R-9): a 400 "invalid agent_job" fails the agent job. The preset
# exits with this code (the agent reports `failed` / `agent_job_refused`) and the runs are
# marked in uploads.json so no later run, by hand or by the agent, sends them without it.
AGENT_JOB_REFUSED_RC = 86
AGENT_JOB_REFUSED_MARK = "not uploaded: the queue refused this agent job's upload"


def _redact_nonce(s: str) -> str:
    return re.sub(r"lsn_[0-9A-Za-z]+", "lsn_[redacted]", s)


def upload_stage(
    args,
    attempts,
    rows,
    rdir: Path,
    preset_name: str,
    sweep_id: str,
    preset: dict[str, Any] | None = None,
) -> int | None:
    from ..api_host import (
        InsecureApiBase,
        display_host,
        guard_api_base,
        token_allowed,
        token_opt_in_hint,
    )
    from ..consent import has_consented, prompt_for_consent
    from ..upload import upload_saved_payload

    if getattr(args, "no_upload", False) or getattr(args, "dry_run", False):
        say("Not uploading (--no-upload). Everything is saved locally.")
        return
    strict = bool(getattr(args, "strict_anon", False))
    anon = bool(getattr(args, "anon", False)) or strict
    api_base = _upload_api_base(args)
    from ..upload_guard import upload_block_reason

    synthetic = next((a.get("synthetic") for a in attempts if a.get("synthetic")), None)
    blocked = upload_block_reason(api_base, (getattr(args, "llama_server", None),))
    if synthetic and not blocked:
        blocked = f"these results came from a synthetic environment ({synthetic})"
    if blocked:
        say(f"Not uploading: {blocked}. Everything is saved locally.")
        return
    consent_ok = strict or has_consented()
    if not consent_ok:
        consent_ok = prompt_for_consent(api_base=api_base, cli_version=__version__)
    if not consent_ok:
        say("Skipping upload (no consent). Results are saved locally.")
        return
    from ..credit import effective_credit, maybe_ask

    maybe_ask(args)
    credit = None if anon else effective_credit(args)
    # L-1: plain http off loopback is refused (cmd_preset already refused it before
    # the run; this is the last check before any POST). Any non-canonical host gets
    # one stderr line naming it; in an agent child that is the enrolled queue.
    # N-5: the API key only goes to api.llm-speed.com or a host the user opted in to
    # (--allow-token-to-host / $LLM_SPEED_ALLOW_TOKEN_HOST); otherwise upload without it.
    api_key = getattr(args, "api_key", None)
    if api_key and not anon and not token_allowed(api_base, args):
        say(
            f"  not sending your API key to {display_host(api_base)} (not "
            f"api.llm-speed.com); uploading without it. To send it there, pass "
            f"{token_opt_in_hint(api_base)}."
        )
        api_key = None
    try:
        guard_api_base(
            api_base,
            source=_upload_api_base_source(args),
            bearer=bool(api_key) and not anon,
        )
    except InsecureApiBase as exc:
        say(f"Not uploading: {exc}. Everything is saved locally.")
        return
    # R-9: inside an `llm-speed agent` job, each upload carries the job's nonce in its signed
    # payload, only to the queue that issued it. Never for a hand-run preset.
    from ..agent_job import for_upload

    aj = for_upload(api_base, strict_anon=strict)
    up_path = _uploads_path(rdir)
    done = json.loads(up_path.read_text()) if up_path.exists() else {}
    from ..upload import UploadRejected

    refused = False
    for a in attempts:
        art = a.get("artifact")
        if a.get("status") not in ("ok", "partial") or not art or art in done:
            continue
        if refused:
            # Security Desk fix 3: this job's runs are never sent without its nonce, so the
            # rest are marked and kept local only (summarize() ignores non-URL values).
            done[art] = AGENT_JOB_REFUSED_MARK
            continue
        try:
            saved = json.loads(Path(art).read_text(encoding="utf-8"))
            url = upload_saved_payload(
                saved,
                api_base=api_base,
                api_key=api_key,
                anon=anon,
                strict_anon=strict,
                credit=credit,
                agent_job=aj,
            )
            done[art] = url
            say(f"  submitted {a['entry_id']} ctx {a.get('ctx')}: {url}")
        except UploadRejected as exc:
            if aj is not None and exc.agent_job_refused:
                refused = True
                done[art] = AGENT_JOB_REFUSED_MARK
                say(
                    f"  the queue refused this job's upload of {a['entry_id']} ctx "
                    f"{a.get('ctx')} (invalid agent_job); the job fails and its runs stay "
                    "local only (never uploaded without the job's nonce)"
                )
                continue
            say(
                f"  upload failed for {a['entry_id']} ctx {a.get('ctx')}: "
                f"{_redact_nonce(str(exc))} (kept locally; rerun to retry)"
            )
        except Exception as exc:  # noqa: BLE001
            say(
                f"  upload failed for {a['entry_id']} ctx {a.get('ctx')}: "
                f"{_redact_nonce(str(exc))} (kept locally; rerun to retry)"
            )
    fs.atomic_write_json(up_path, done)
    if refused:
        return AGENT_JOB_REFUSED_RC
    # There is no public fit_attempts write route (ops/AGENT_QUEUE_API.md section 5):
    # rows reach D1 only in an `llm-speed agent` job result. A manual run keeps them local.
    say(
        f"  fit attempts: {len(rows)} rows kept locally in fit_attempts.json "
        "(the site takes them only from `llm-speed agent` job results)"
    )


# ----------------------------------------------------------------- command


def cmd_preset(args) -> int:
    from ..agent_job import take_from_env

    # R-9: an agent child gets its job's upload nonce once through an inherited pipe; read
    # it now (memory only) so the per-cell children never inherit the fd or its number.
    take_from_env()
    name = args.preset
    preset = load_preset(name)
    rdir = results_dir(name)
    models_dir = Path(
        os.path.expanduser(
            getattr(args, "models_dir", None)
            or os.environ.get("LLM_SPEED_MODELS_DIR")
            or str(CACHE_DIR / "models")
        )
    )
    total = 4
    say(
        f"llm-speed {__version__}: preset {name} v{preset.get('version')} - {preset.get('title', '')}"
    )
    say(f"  preset sha256 {preset_sha256(preset)}")
    if not (getattr(args, "no_upload", False) or getattr(args, "dry_run", False)):
        # L-1: refuse a plain-http upload target up front, not after an hour of runs.
        from ..api_host import InsecureApiBase, check_api_base

        try:
            check_api_base(_upload_api_base(args))
        except InsecureApiBase as exc:
            print(
                f"error: {exc} (from {_upload_api_base_source(args)}). "
                "Fix the URL or pass --no-upload.",
                file=sys.stderr,
            )
            return 2

    stage(1, total, "Hardware")
    facts = fs.machine_facts(0)
    if facts.get("kind") == "nvidia":
        say(
            f"  NVIDIA {facts.get('gpu')} ({(facts.get('gpu_memory_total_mib') or 0) / 1024:.0f} GiB VRAM), "
            f"RAM {facts.get('ram_gb')} GB"
        )
    elif facts.get("kind") == "apple":
        say(
            f"  {facts.get('chip')}, {facts.get('ram_gb')} GB unified memory, macOS {facts.get('macos')}"
        )
    plan, notes = build_plan(
        preset,
        facts,
        with_control=getattr(args, "with_control", False),
        models_dir=models_dir,
        mlx_dir=models_dir / "mlx",
        llama_server=None,
    )
    plan["cli_sampling"] = sampling_mod.from_args(args)
    say(
        f"  sampling: {sampling_mod.describe(sampling_mod.merge(plan['sampling'], plan['cli_sampling']))}"
    )
    for n in notes:
        say(f"  note: {n}")
    live = [e for e in plan["entries"] if not e.get("skip_status")]
    for e in live:
        ctxs = ", ".join(
            f"{(c['ctx'] if isinstance(c, dict) else c) // 1024}K"
            for c in e["contexts"]
        )
        say(
            f"  will run: {e['id']} ({e['backend']}, {e['quant']}, {e['bytes'] / 1e9:.1f} GB) at {ctxs}"
        )

    stage(2, total, "Backends")
    min_build = preset.get("min_llama_build")
    need_llama = any(e.get("backend", "llama.cpp") == "llama.cpp" for e in live)
    if need_llama:
        path, info, tried = find_llama_server(
            getattr(args, "llama_server", None), min_build
        )
        bn = (info or {}).get("build_number")
        ok = bool(path and bn and min_build and bn >= int(min_build))
        if path and not ok and not getattr(args, "force", False):
            why = preset.get("min_llama_build_why", "")
            if bn and bn > 1:
                say(
                    f"  llama-server at {path} is build b{bn}; this preset needs b{min_build} or newer ({why})."
                )
            else:
                say(
                    f"  llama-server at {path} does not report a build number "
                    f"(commit {(info or {}).get('commit') or '?'}); cannot confirm b{min_build}+."
                )
            path = None
        if not path:
            cmd, text = llama_install_command(facts, f"b{min_build}")
            say(f"  llama-server b{min_build}+ not found. Install it with:\n    {text}")
            if cmd and ask_yes_no(
                "  Run this install command now?",
                assume_yes=getattr(args, "yes", False),
            ):
                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    say(f"  install command exited {rc}")
                    return 2
                path, info, tried = find_llama_server(
                    getattr(args, "llama_server", None), min_build
                )
                bn = (info or {}).get("build_number")
                if not (path and bn and bn >= int(min_build)):
                    say(
                        "  still no llama-server at the required build; see the command above"
                    )
                    return 2
            elif not getattr(args, "force", False):
                say("  Then rerun this command (or pass --llama-server PATH).")
                return 2
        if path:
            plan["llama_server"] = path
            say(
                f"  llama-server: {Path(path).name} build {(info or {}).get('build') or bn} "
                f"commit {(info or {}).get('commit')} ({', '.join((info or {}).get('flags') or []) or 'cpu'})"
            )
    if any(e.get("backend") == "mlx" for e in live):
        if importlib.util.find_spec("mlx_lm") is None:
            hint = "pipx inject llm-speed mlx-lm   (or: pip install -U mlx-lm in this Python)"
            say(
                f"  mlx-lm is not installed in this Python; MLX steps will be skipped. Install with:\n    {hint}"
            )
            for e in live:
                if e.get("backend") == "mlx":
                    e["skip_status"], e["skip_note"] = (
                        "skipped_backend",
                        f"mlx-lm not installed ({hint})",
                    )
        else:
            say(
                "  mlx-lm: installed (Flint needs a version with the 'flint' architecture; checked at load)"
            )

    stage(
        3,
        total,
        "Benchmark (rerun the same command to resume; Ctrl-C keeps finished steps)",
    )
    sargs = SimpleNamespace(
        only="",
        include_optional=True,
        machine=None,
        results=str(rdir / "results.json"),
        download=not getattr(args, "no_download", False),
        cleanup_mode="never" if getattr(args, "keep_models", False) else "auto",
        delete_preexisting=False,
        no_verify=False,
        upload=False,
        force=getattr(args, "force", False),
        strict_anon=getattr(args, "strict_anon", False),
        cli_prefix=[sys.executable, "-m", "cli"],
    )
    sweep = fs.Sweep(None, sargs, plan=plan, launch_dir=rdir)
    try:
        rc = sweep.run()
    except KeyboardInterrupt:
        rc = 130
    if getattr(sweep, "facts", None) is None or sweep.machine == "auto":
        return rc
    if rc == 2:
        say("Preflight failed; nothing was run.")
        return rc

    stage(4, total, "Results")
    attempts, rows = fit_rows(sweep, name)
    sweep_id = rows[0]["sweep_id"] if rows else f"{name}-{sweep.machine}"
    fs.atomic_write_json(
        rdir / "fit_attempts.json",
        {
            "schema": "fit-attempts/1",
            "cli_version": __version__,
            "preset": name,
            "preset_version": preset.get("version"),
            "preset_sha256": preset_sha256(preset),
            "attempts": rows,
        },
    )
    shown = latest_per_cell(rows)
    say(summary_table(shown))
    say(f"\nSaved: {rdir / 'results.json'} and {rdir / 'fit_attempts.json'}")
    if rc == 130:
        say(
            f"Interrupted. Rerun to resume: llm-speed bench --preset {name}"
            + (" --with-control" if getattr(args, "with_control", False) else "")
        )
        return rc
    if upload_stage(args, attempts, rows, rdir, name, sweep_id, preset) == (
        AGENT_JOB_REFUSED_RC
    ):
        return AGENT_JOB_REFUSED_RC
    return 0 if rc == 0 else rc
