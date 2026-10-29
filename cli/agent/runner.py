"""The agent loop: gates -> claim -> validate -> (approve) -> run preset child -> report.

Wire format: ops/AGENT_QUEUE_API.md v1 (through transport.py only).
The benchmark itself is the child process `python -m cli bench --preset NAME ...`
(the 0.0.8 preset path in cli/commands/preset.py). The agent only ever signals that
child's own process group; it never touches any other process on the machine.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import re
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import __version__, preset_schema
from .. import agent_job as _agent_job
from ..api_host import ALLOW_TOKEN_HOST_ENV, display_host, is_canonical_api_base
from ..config import CACHE_DIR
from . import gates
from .allowlist import JobRejected, advertised, build_argv, job_context, validate_job
from .settings import (
    Settings,
    atomic_write_json,
    ensure_dirs,
    logs_dir,
    outbox_dir,
    pause_path,
    state_dir,
)
from .tokens import redact
from .transport import QueueClient, QueueError

try:
    import fcntl
except ImportError:  # pragma: no cover (Windows later)
    fcntl = None  # type: ignore[assignment]

# Exit code of `bench --preset` when the queue refused the job's signed agent_job (400
# "invalid agent_job"); same value as cli/commands/preset.py AGENT_JOB_REFUSED_RC.
AGENT_JOB_REFUSED_RC = 86
# Logged once per uploading job whose claim had no upload_nonce (spec rev 3, section 1.1).
# Neutral wording only: how the server labels or classifies runs is decided server-side and
# never described by the CLI (guarded by tests/cli/test_agent_nonce.py). Never the value.
NO_NONCE_NOTE = (
    "this job had no upload nonce from the queue; its results upload without one"
)
BAD_NONCE_NOTE = "the queue's upload nonce for this job could not be used; its results upload without one"

log = logging.getLogger("cli.agent")

IDLE_POLL_S = 60
# After a claimed job is handed back because this machine lacks the free VRAM / memory /
# disk it needs, wait longer before claiming again (the queue may offer the same job).
RESOURCE_RETRY_S = 300
MAX_BACKOFF_S = 900
CANCEL_GRACE_S = 30
KEEP_JOB_LOGS = 20
MAX_FIT_ROWS = 500
ID_RE = re.compile(r"[A-Za-z0-9_\-]{1,80}")
REASON_RE = re.compile(r"[^a-z0-9_:.\-]")
# Result statuses the route accepts (queue API spec section 5).
RESULT_STATUSES = ("succeeded", "partial", "failed", "cancelled", "rejected")
# fit_attempts columns the result route accepts (queue API spec section 5). Unknown fields -> 400
# (the whole result is refused). The 0.0.8 Flint extension columns (ttfa_ms, n_cpu_moe,
# context_requested, ...; cli.fit_attempts.FLINT_COLUMNS) are accepted since migration
# worker/migrations/0010_fit_attempts_flint.sql and FIT_OPT in worker/src/agent-queue.ts
# (live 2026-10-22). Set False to strip them again.
SEND_FLINT_FIT_COLUMNS = True
FIT_REQUIRED = (
    "sweep_id",
    "model_slug",
    "model_file",
    "hardware_slug",
    "quant",
    "context_tokens",
    "status",
    "measured_at",
)
FIT_OPTIONAL = (
    "gpu_name",
    "n_gpu_layers",
    "n_layers",
    "cpu_offload",
    "error_type",
    "error_detail",
    "decode_tps",
    "prefill_tps",
    "vram_total_mib",
    "vram_peak_mib",
    "file_bytes",
    "backend",
    "backend_version",
)


def utcnow() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def reason_code(s: str | None) -> str | None:
    if s is None:
        return None
    return REASON_RE.sub("_", s.lower())[:80] or "unknown"


# ------------------------------------------------------------------ capabilities


def _short(s: Any, n: int = 80) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s or "/" in s or "\\" in s:
        return None
    return s[:n]


def capabilities(settings: Settings, *, facts: dict | None = None) -> dict[str, Any]:
    """Coarse capabilities (queue API spec section 2). No hostnames, usernames, paths, serials.
    memory_gb = system/unified RAM; vram_gb = GPU memory (Apple: the GPU-usable share).
    hardware_slug is the same slug the preset writes into fit_attempts rows."""
    from .. import fit_sweep

    f = facts if facts is not None else fit_sweep.machine_facts(0)
    kind = f.get("kind")
    caps: dict[str, Any] = {
        "os": {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
            f.get("os"), "linux"
        ),
        "arch": _short(f.get("arch")),
        "gpu_vendor": {"nvidia": "nvidia", "apple": "apple"}.get(kind, "none"),
        "gpu_count": 1 if kind in ("nvidia", "apple") else 0,
        "memory_gb": round(f["ram_gb"]) if f.get("ram_gb") else None,
        "cpu_cores": os.cpu_count(),
        "backends": [],
        "hardware_slug": _short(fit_sweep.machine_slug(f)),
    }
    if kind == "nvidia":
        caps["gpu_name"] = _short(f.get("gpu"))
        mib = f.get("gpu_memory_total_mib")
        caps["vram_gb"] = round(mib / 1024) if mib else None
        drv = str(f.get("driver_version") or "").split(".")[0]
        caps["driver"] = drv if drv.isdigit() else None  # major only, like the uploads
        caps["backends"] = ["llama.cpp"]
    elif kind == "apple":
        caps["chip"] = caps["gpu_name"] = _short(f.get("chip"))
        wl = f.get("iogpu_wired_limit_mb")
        ram = f.get("ram_gb") or 0
        caps["vram_gb"] = (
            round(wl / 1024) if wl else (round(ram * 0.75) if ram else None)
        )
        caps["backends"] = ["llama.cpp", "mlx"]
    try:
        caps["disk_free_gb"] = round(gates.free_disk_gb(settings))
    except OSError:
        caps["disk_free_gb"] = None
    caps["presets"] = advertised(settings)
    return {k: v for k, v in caps.items() if v is not None}


# ------------------------------------------------------------------ local state files


def _status_path() -> Path:
    return state_dir() / "status.json"


def _state_path() -> Path:
    return state_dir() / "state.json"


def approvals_dir() -> Path:
    return state_dir() / "approvals"


def write_status(**kw: Any) -> None:
    p = _status_path()
    cur = _load_json(p) or {}
    cur.update(kw, updated_at=utcnow())
    atomic_write_json(p, cur)


def read_status() -> dict:
    return _load_json(_status_path()) or {}


def pause_now_requested() -> bool:
    p = pause_path()
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text() or "{}").get("now"))
    except ValueError:
        return False


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ child process


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal only OUR child's process group (it was started with start_new_session)."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_child(proc: subprocess.Popen, grace: float = CANCEL_GRACE_S) -> None:
    if proc.poll() is not None:
        return
    _signal_group(
        proc, signal.SIGINT
    )  # the preset treats Ctrl-C as "keep finished steps"
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()


def child_env(api_base: str | None = None) -> dict[str, str]:
    """Environment of the preset child. `api_base` is the queue this agent is enrolled
    with: the child's result uploads (`llm-speed bench --preset` reads $LLM_SPEED_API)
    go to that same server, never to a different default.

    N-5: enrolling was the explicit choice of that server, so a self-hosted queue gets
    the token opt-in ($LLM_SPEED_ALLOW_TOKEN_HOST) for exactly its host. Any opt-in
    inherited from the agent's own environment is dropped, so the environment can't
    point the child at a different host. (The child runs --strict-anon and sends no
    bearer today; this keeps it that way if that ever changes.)"""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("LLM_SPEED_AGENT_")
        and k not in (ALLOW_TOKEN_HOST_ENV, "LLM_SPEED_ENROLMENT_TOKEN")
    }
    env["PYTHONUNBUFFERED"] = "1"
    if api_base:
        base = api_base.rstrip("/")
        env["LLM_SPEED_API"] = base
        if not is_canonical_api_base(base):
            env[ALLOW_TOKEN_HOST_ENV] = display_host(base)
    return env


def preset_results_dir(name: str) -> Path:
    return CACHE_DIR / "presets" / name


Launcher = Callable[..., subprocess.Popen]  # (argv, logfile, *, env) -> Popen


def default_launcher(
    argv: list[str],
    logfile: Path,
    *,
    env: dict[str, str] | None = None,
    agent_job: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Start the preset child. `agent_job` (R-9 upload nonce context) reaches the child
    only through an inherited pipe whose fd number is in $LLM_SPEED_AGENT_JOB_FD: never
    argv, never the environment, never a file, never the job log."""
    env = dict(env if env is not None else child_env())
    env.pop(_agent_job.FD_ENV, None)
    rfd = None
    pass_fds: tuple[int, ...] = ()
    if agent_job is not None and os.name == "posix":
        rfd, wfd = os.pipe()
        try:
            os.write(wfd, _agent_job.encode(agent_job))  # < 1 KiB: fits any pipe buffer
        finally:
            os.close(wfd)
        env[_agent_job.FD_ENV] = str(rfd)
        pass_fds = (rfd,)
    try:
        with open(logfile, "ab") as f:
            return subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
                shell=False,
                pass_fds=pass_fds,
            )
    finally:
        if rfd is not None:
            os.close(rfd)


def fit_optional_columns() -> tuple[str, ...]:
    if not SEND_FLINT_FIT_COLUMNS:
        return FIT_OPTIONAL
    from ..fit_attempts import FLINT_COLUMNS

    return FIT_OPTIONAL + tuple(FLINT_COLUMNS)


def _clean_row(r: dict) -> dict | None:
    if not all(r.get(k) is not None for k in FIT_REQUIRED):
        return None
    out = {k: r[k] for k in FIT_REQUIRED}
    out.update({k: r[k] for k in fit_optional_columns() if r.get(k) is not None})
    if "/" in str(out["model_file"]) or "\\" in str(out["model_file"]):
        return None
    if isinstance(out.get("error_detail"), str):
        out["error_detail"] = out["error_detail"][:500]
    return out


def summarize(
    name: str, uploads_before: dict, started_at: str | None = None
) -> dict[str, Any]:
    """Result fields from the preset's own files: counts, run ids, fit rows (the route's columns only)."""
    rdir = preset_results_dir(name)
    fa = _load_json(rdir / "fit_attempts.json") or {}
    rows = fa.get("attempts", []) if isinstance(fa, dict) else []
    latest: dict[tuple, dict] = {}
    for r in rows:
        if isinstance(r, dict):
            latest[
                (
                    r.get("model_file"),
                    r.get("quant"),
                    r.get("context_tokens"),
                    r.get("n_gpu_layers"),
                    r.get("n_cpu_moe"),
                )
            ] = r
    counts = {"ok": 0, "oom": 0, "failed": 0, "skipped": 0}
    for r in latest.values():
        if r.get("status") in counts:
            counts[r["status"]] += 1
    counts["rows"] = len(latest)
    uploads = _load_json(rdir / "uploads.json") or {}
    run_ids = []
    for k, url in uploads.items() if isinstance(uploads, dict) else []:
        if k in uploads_before or not isinstance(url, str):
            continue
        m = re.search(r"/r/([A-Za-z0-9_\-]+)", url)
        if m:
            run_ids.append(m.group(1))
    clean = [c for c in (_clean_row(r) for r in latest.values()) if c][:MAX_FIT_ROWS]
    sweep_id = next(
        (r.get("sweep_id") for r in rows if isinstance(r, dict) and r.get("sweep_id")),
        None,
    )
    return {
        "summary": counts,
        "run_ids": run_ids,
        "fit_sweep_id": sweep_id,
        "fit_attempts": clean,
    }


def status_for_exit(rc: int) -> tuple[str, str | None]:
    if rc == 0:
        return (
            "succeeded",
            None,
        )  # the preset exits 0 only when every planned cell has a row
    if rc in (130, -signal.SIGINT):
        return "cancelled", "interrupted"
    if rc == 2:
        return "failed", "preflight_failed"
    if rc == AGENT_JOB_REFUSED_RC:
        return (
            "failed",
            "agent_job_refused",
        )  # 400 invalid agent_job: never re-sent bare
    return "failed", f"exit_{rc}"


# ------------------------------------------------------------------ runner


class Runner:
    def __init__(
        self,
        client: QueueClient,
        settings: Settings,
        *,
        launcher: Launcher = default_launcher,
        sleep: Callable[[float], None] = time.sleep,
        gate_runner=gates._run,
        preset_keys: dict[str, str] | list[str] | None = None,
        preset_loader=None,
        flags: frozenset[str] | None = None,
        facts: dict | None = None,
    ):
        self.client, self.settings = client, settings
        self.launcher, self.sleep, self.gate_runner = launcher, sleep, gate_runner
        self.preset_keys, self.flags, self.facts = preset_keys, flags, facts
        self.preset_loader = preset_loader or preset_schema.packaged_preset
        self.backoff = 5.0
        self.next_wait: float | None = None  # set by handle() when a job is handed back
        self.stopped_reason: str | None = None
        self.server_requires_approval = False
        ensure_dirs()

    # ------------------------------------------------------------- control
    def obey(self, control: dict | None) -> None:
        c = control or {}
        if c.get("disabled"):
            self.stopped_reason = "disabled by server (kill switch)"
        mv = c.get("min_cli_version")
        if mv and _vtuple(__version__) < _vtuple(str(mv)):
            self.stopped_reason = f"update required: server needs llm-speed >= {mv}"
        if "paused" in c:
            write_status(server_paused=bool(c.get("paused")))

    def poll_after(self, control: dict | None) -> float:
        v = (control or {}).get("poll_after_s") or IDLE_POLL_S
        return min(max(float(v), 15.0), 3600.0) * random.uniform(0.8, 1.2)

    def refresh_status(self) -> None:
        try:
            st = self.client.status()
        except QueueError as e:
            if e.status in (401, 403, 426):
                self._on_error(e)
            return
        self.stopped_reason = None
        self.server_requires_approval = bool(st.get("require_approval"))
        self.obey(st.get("control"))
        if st.get("status") == "paused":
            write_status(server_paused=True)

    # ------------------------------------------------------------- outbox
    def flush_outbox(self) -> None:
        for p in sorted(outbox_dir().glob("*.json")):
            item = _load_json(p)
            if not item:
                p.unlink(missing_ok=True)
                continue
            try:
                self.client.result(item["job_id"], item["report"], item["key"])
            except QueueError as e:
                if e.status in (400, 404, 409, 413, 422):
                    log.warning(
                        "result for %s refused by server (%s); dropped",
                        item["job_id"],
                        e.code,
                    )
                else:
                    log.info(
                        "result for %s not delivered yet (%s); will retry",
                        item["job_id"],
                        e.code,
                    )
                    return
            p.unlink(missing_ok=True)

    def report(
        self,
        job: dict,
        lease: dict,
        status: str,
        reason: str | None,
        extra: dict | None = None,
        rc: int | None = None,
        started_at: str | None = None,
    ) -> None:
        if status not in RESULT_STATUSES:
            raise ValueError(f"not a result status: {status}")
        rep: dict[str, Any] = {
            "lease_id": lease.get("lease_id"),
            "status": status,
            "exit_code": rc,
            "reason": reason_code(reason),
            "started_at": started_at,
            "finished_at": utcnow(),
            "cli_version": __version__,
            "summary": None,
            "run_ids": [],
            "fit_sweep_id": None,
        }
        rep.update(extra or {})
        if not rep.get("fit_attempts"):
            rep.pop("fit_attempts", None)
        job_id = str(job.get("job_id"))
        key = f"{job_id}:{job.get('attempt', 1)}"
        atomic_write_json(
            outbox_dir() / f"{job_id}.json",
            {"job_id": job_id, "key": key, "report": rep},
        )
        write_status(
            last_result={
                "job_id": job_id,
                "preset": (job.get("preset") or {}).get("name"),
                "status": status,
                "reason": rep["reason"],
                "at": utcnow(),
            },
            current_job=None,
        )
        self.flush_outbox()

    # ------------------------------------------------------------- one cycle
    def cycle(self) -> float:
        """One claim cycle. Returns seconds to sleep before the next."""
        self.flush_outbox()
        if self.stopped_reason:
            write_status(state="stopped", reason=self.stopped_reason)
            self.refresh_status()
            if self.stopped_reason:
                return 3600.0
        g = gates.check(self.settings, runner=self.gate_runner, sleep=self.sleep)
        write_status(gates={"open": g.open, "reasons": g.reasons, "facts": g.facts})
        if not g.open:
            write_status(state="waiting", reason="; ".join(g.reasons))
            return IDLE_POLL_S
        try:
            claim = self.client.claim(capabilities(self.settings, facts=self.facts))
        except QueueError as e:
            return self._on_error(e)
        self.backoff = 5.0
        self.obey(claim.control)
        if self.stopped_reason:
            return 3600.0
        if not claim.job:
            write_status(
                state="paused by server" if claim.control.get("paused") else "idle",
                reason=None,
            )
            return self.poll_after(claim.control)
        self.next_wait = None
        self.handle(claim.job, claim.lease or {})
        wait, self.next_wait = self.next_wait, None
        return wait if wait is not None else 1.0

    def _on_error(self, e: QueueError) -> float:
        if e.status == 401:
            self.stopped_reason = (
                "token rejected (401): re-enroll with `llm-speed agent enroll`"
            )
        elif e.status == 403:
            self.stopped_reason = f"machine disabled by server (403 {e.code})"
        elif e.status == 426:
            self.stopped_reason = "update required (426): pipx upgrade llm-speed && llm-speed agent restart"
        elif e.code == "insecure_api_base":
            self.stopped_reason = str(e)
        if self.stopped_reason:
            write_status(state="stopped", reason=self.stopped_reason)
            return 3600.0
        wait = e.retry_after or self.backoff
        self.backoff = min(self.backoff * 2, MAX_BACKOFF_S)
        write_status(state="backoff", reason=f"queue error {e.code}")
        return wait * random.uniform(0.8, 1.2)

    # ------------------------------------------------------------- one job
    def handle(self, job: dict, lease: dict) -> None:
        job_id = str(job.get("job_id") or "")
        if not ID_RE.fullmatch(job_id):
            log.warning("server sent a job with a malformed id; ignoring it")
            return
        name = (
            (job.get("preset") or {}).get("name")
            if isinstance(job.get("preset"), dict)
            else None
        )
        write_status(
            state="claimed",
            current_job={"job_id": job_id, "preset": name, "at": utcnow()},
        )
        try:
            preset, overrides = validate_job(
                job, self.settings, preset_loader=self.preset_loader, flags=self.flags
            )
            if self.settings.remote_presets:
                self._check_revocations(preset)
        except JobRejected as r:
            if r.reason == "not_before":
                self._release(job, lease, "gate_closed")
                return
            log.warning("job %s rejected: %s", job_id, r.reason)
            self.report(job, lease, "rejected", r.reason)
            return
        needs_ok = (
            self.settings.manual_approve
            or bool(job.get("require_approval"))
            or self.server_requires_approval
        )
        if needs_ok and not self._wait_approval(job, lease):
            return
        needs = gates.preset_needs(preset, overrides, self.settings, self._facts())
        g = gates.check(
            self.settings,
            runner=self.gate_runner,
            sleep=self.sleep,
            needs=needs,
            sample_busy=needs_ok,  # sampled just before the claim unless we waited
        )
        if not g.open:
            reason = "; ".join(g.reasons)
            log.info("job %s handed back: %s", job_id, reason)
            write_status(gates={"open": False, "reasons": g.reasons, "facts": g.facts})
            self._release(job, lease, "gate_closed", detail=reason)
            if any(r.startswith("not enough free") for r in g.reasons):
                self.next_wait = RESOURCE_RETRY_S * random.uniform(0.8, 1.2)
            return
        if preset.get("preset") == "selftest":
            self._selftest(job, lease, overrides)
            return
        self._run(
            job,
            lease,
            preset,
            overrides,
            upload=bool(job.get("upload", True)),
            max_runtime_min=job.get("max_runtime_min"),
        )

    def _check_revocations(self, preset: dict) -> None:
        """Phase 2 (remote_presets on): the signed revocation list is checked before every
        job, packaged presets included. Any failure = reject (fail closed)."""
        from .signed_presets import PresetVerificationError, check_revocations

        if preset.get("preset") == "selftest":
            return
        try:
            check_revocations(
                self.client, preset["preset"], int(preset["version"]), self.preset_keys
            )
        except PresetVerificationError as e:
            raise JobRejected(
                f"preset_verification_failed:{reason_code(str(e))}"[:80]
            ) from None

    def _facts(self) -> dict:
        if self.facts is not None:
            return self.facts
        from .. import fit_sweep

        return fit_sweep.machine_facts(0)

    def _release(
        self, job: dict, lease: dict, reason: str, detail: str | None = None
    ) -> None:
        """`reason` is the queue's release code; `detail` (local status only) says why."""
        try:
            self.client.release(str(job.get("job_id")), lease.get("lease_id"), reason)
        except QueueError as e:
            log.info("release failed (%s); the lease will expire", e.code)
        write_status(
            state="released",
            current_job=None,
            reason=f"{reason}: {detail}" if detail else reason,
        )

    def _heartbeat(
        self, job: dict, lease: dict, phase: str, progress: dict | None = None
    ) -> str | None:
        """None = carry on; 'cancel' = server cancelled; 'lost' = lease lost (409)."""
        try:
            r = self.client.heartbeat(
                str(job.get("job_id")), lease.get("lease_id"), phase, progress
            )
        except QueueError as e:
            if e.status == 409:
                return "lost"
            log.info("heartbeat failed (%s); carrying on", e.code)
            return None
        self.obey(r.get("control"))
        return "cancel" if r.get("cancel") else None

    def _wait_approval(self, job: dict, lease: dict) -> bool:
        job_id = str(job.get("job_id"))
        ad = approvals_dir()
        ad.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_write_json(
            ad / f"{job_id}.pending",
            {
                "job_id": job_id,
                "preset": job.get("preset"),
                "overrides": job.get("overrides"),
                "launch_id": job.get("launch_id"),
                "at": utcnow(),
            },
        )
        write_status(
            state="waiting_approval", reason=f"run `llm-speed agent approve {job_id}`"
        )
        deadline = time.monotonic() + self.settings.approve_timeout_min * 60
        hb = float(lease.get("heartbeat_s") or 60)
        last_hb = 0.0
        try:
            while time.monotonic() < deadline:
                if (ad / f"{job_id}.approve").exists():
                    return True
                if (ad / f"{job_id}.decline").exists():
                    self._release(job, lease, "declined")
                    return False
                if time.monotonic() - last_hb >= hb:
                    last_hb = time.monotonic()
                    hbr = self._heartbeat(job, lease, "waiting_approval")
                    if hbr == "cancel":
                        self.report(job, lease, "cancelled", "cancelled_by_server")
                        return False
                    if hbr == "lost":
                        write_status(
                            state="idle", current_job=None, reason="lease lost"
                        )
                        return False
                self.sleep(min(hb, 2.0))
            self._release(job, lease, "approval_timeout")
            return False
        finally:
            for suffix in ("pending", "approve", "decline"):
                (ad / f"{job_id}.{suffix}").unlink(missing_ok=True)

    def _selftest(self, job: dict, lease: dict, ov: dict) -> None:
        started = utcnow()
        write_status(
            state="running",
            current_job={
                "job_id": job.get("job_id"),
                "preset": "selftest",
                "started_at": started,
            },
        )
        end = time.monotonic() + int(ov.get("sleep_s", 0))
        while time.monotonic() < end:
            hbr = self._heartbeat(job, lease, "running", {"stage": "selftest"})
            if hbr == "lost":
                write_status(
                    state="idle", current_job=None, reason="lease lost; job dropped"
                )
                return
            if hbr == "cancel":
                self.report(
                    job, lease, "cancelled", "cancelled_by_server", started_at=started
                )
                return
            self.sleep(
                min(
                    float(lease.get("heartbeat_s") or 60),
                    max(0.0, end - time.monotonic()),
                )
            )
        outcome = ov.get("outcome", "succeeded")
        self.report(
            job,
            lease,
            outcome,
            None if outcome == "succeeded" else "selftest_failed",
            {"summary": {"ok": 0, "oom": 0, "failed": 0, "skipped": 0, "rows": 0}},
            rc=0 if outcome == "succeeded" else 1,
            started_at=started,
        )

    def _run(
        self,
        job: dict,
        lease: dict,
        preset: dict,
        overrides: dict,
        *,
        upload: bool,
        max_runtime_min: int | None,
    ) -> None:
        job_id, name = str(job.get("job_id")), str(preset["preset"])
        argv = build_argv(name, overrides, self.settings, upload=upload)
        logfile = logs_dir() / f"job-{job_id}.log"
        uploads_before = _load_json(preset_results_dir(name) / "uploads.json") or {}
        started = utcnow()
        # R-9: the claim's upload nonce goes to the child in memory only (a pipe), and only
        # when this job uploads. No nonce from the server = run exactly as before.
        ctx = (
            job_context(job, self.client.api_base)
            if upload and self.settings.upload
            else None
        )
        launch_kw: dict[str, Any] = {"env": child_env(self.client.api_base)}
        if ctx is not None:
            launch_kw["agent_job"] = ctx
        elif upload and self.settings.upload:
            # No nonce (older server, or a queue that didn't see this build's feature), or one
            # this build can't use. Once per job, no values.
            log.info(
                NO_NONCE_NOTE if job.get("upload_nonce") is None else BAD_NONCE_NOTE
            )
        proc = self.launcher(argv, logfile, **launch_kw)
        atomic_write_json(
            _state_path(),
            {
                "job_id": job_id,
                "preset": name,
                "child_pid": proc.pid,
                "started_at": started,
            },
        )
        write_status(
            state="running",
            current_job={"job_id": job_id, "preset": name, "started_at": started},
        )
        log.info("job %s: running preset %s (pid %s)", job_id, name, proc.pid)
        hb = max(1.0, float(lease.get("heartbeat_s") or 60))
        deadline = time.monotonic() + 60 * max_runtime_min if max_runtime_min else None
        outcome: tuple[str, str | None] | None = None
        while True:
            try:
                rc = proc.wait(timeout=hb)
                break
            except subprocess.TimeoutExpired:
                pass
            if pause_now_requested():
                stop_child(proc)
                outcome, rc = ("released", "paused"), proc.returncode
                break
            if deadline and time.monotonic() > deadline:
                stop_child(proc)
                outcome, rc = ("failed", "max_runtime_exceeded"), proc.returncode
                break
            hbr = self._heartbeat(job, lease, "running")
            if hbr in ("cancel", "lost"):
                stop_child(proc)
                outcome = (
                    ("cancelled", "cancelled_by_server")
                    if hbr == "cancel"
                    else ("dropped", "lease_lost")
                )
                rc = proc.returncode
                break
        _state_path().unlink(missing_ok=True)
        self._prune_logs()
        if outcome and outcome[0] == "dropped":
            write_status(
                state="idle", current_job=None, reason="lease lost; job dropped"
            )
            return
        if outcome and outcome[0] == "released":
            self._release(job, lease, "paused")
            return
        extra = summarize(name, uploads_before, started)
        status, reason = outcome or status_for_exit(rc)
        if (
            status == "failed"
            and reason != "preflight_failed"  # preflight runs no cell; rows are older
            and reason != "agent_job_refused"  # the queue refused the job's uploads
            and (extra.get("summary") or {}).get("ok")
        ):
            status = "partial"  # some cells measured before the crash / runtime cap
        log.info("job %s finished: %s %s", job_id, status, redact(reason or ""))
        self.report(job, lease, status, reason, extra, rc=rc, started_at=started)

    @staticmethod
    def _prune_logs() -> None:
        logs = sorted(logs_dir().glob("job-*.log"), key=lambda p: p.stat().st_mtime)
        for p in logs[:-KEEP_JOB_LOGS]:
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------- forever
    def run_forever(self, once: bool = False) -> int:
        lock_f = open(state_dir() / "run.lock", "a")  # noqa: SIM115 (held for the process lifetime)
        if fcntl is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                print(
                    "another `llm-speed agent run` is already running on this machine"
                )
                return 2
        self._recover()
        self.refresh_status()
        while True:
            wait = self.cycle()
            if once:
                return 0 if not self.stopped_reason else 1
            self.sleep(wait)

    def _recover(self) -> None:
        """After a crash/reboot: stop our own orphaned preset child (never anything else);
        the idempotent claim hands the job back and the preset resumes from its files."""
        st = _load_json(_state_path())
        if not st:
            return
        pid, name = st.get("child_pid"), str(st.get("preset") or "")
        log.info("recovering after an unclean stop (job %s)", st.get("job_id"))
        if pid and name and _is_our_preset_child(int(pid), name):
            try:
                os.killpg(int(pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            for _ in range(int(CANCEL_GRACE_S * 4)):
                if not _is_our_preset_child(int(pid), name):
                    break
                self.sleep(1.0)
        _state_path().unlink(missing_ok=True)


def _is_our_preset_child(pid: int, name: str) -> bool:
    """True only if pid is alive, leads its own process group, and its command line is
    our preset child. A reused pid belonging to anything else never matches."""
    try:
        if os.getpgid(pid) != pid:
            return False
    except (ProcessLookupError, PermissionError, OSError):
        return False
    out = gates._run(["ps", "-o", "command=", "-p", str(pid)]) or ""
    return f"-m cli bench --preset {name}" in out


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)
