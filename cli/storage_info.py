"""Opt-in storage-drive info (0.0.13, off by default).

When the user opts in, `llm-speed bench` adds one small block describing the drive
that holds the model file being benchmarked, ONLY on runs that stream weights from
disk (an `ssd_stream: true` preset entry, extras.ssd_stream = true in the run record),
where speed mostly depends on the drive. Any other run is never probed and never
carries the block, whatever the setting says.

The block (signed run payload, ``fingerprint.storage``) has exactly these keys::

    {"bus": "nvme"|"sata"|"usb"|"virtual"|"other" | None,
     "pcie_gen": int | None,       # NVMe only
     "pcie_width": int | None,     # NVMe only
     "fs": "ext4"|"xfs"|"btrfs"|"apfs"|"ntfs"|"9p"|"other" | None,
     "wsl_9p": bool}               # model sits on a WSL /mnt/<drive> share

0.0.13 has NO drive model name (GM, after Security Desk R-128): the model string is
never read, probed, uploaded or accepted by the Worker. Never collected either:
capacity, serial numbers, WWN/EUI, device paths, mount points, partition or volume
ids, labels. Probes read only the bus type, the PCIe link of the controller and the
filesystem type. The Worker (worker/src/storage-info.ts) rejects any other shape with
a 400; ``is_canonical`` is the same rule here.

Consent: the setting ``share_storage_info`` in ``~/.config/llm-speed/config.json`` counts
as on only when it is JSON ``true``. ``bench --share-storage-info`` /
``--no-share-storage-info`` override it for one run. ``--strict-anon`` always forces it
off, and so does an ``llm-speed agent`` job. The one-time question is asked only at an
interactive terminal, right after upload consent; Enter means no.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("cli.storage_info")

# --------------------------------------------------------------------------- wording

# Approved by Boss 2026-10-31. The ONLY copy of the one-time question; docs/CLI.md
# and docs/PRIVACY.md quote it. Changing it needs Boss's approval again.
STORAGE_PROMPT_TEXT = (
    "Also share which drive holds the model, for runs that stream weights from disk?\n"
    "  When a model is too big for your GPU and RAM, its speed depends mostly on the\n"
    "  drive it's read from. For those runs only, this adds the drive's connection\n"
    "  (NVMe, SATA or USB), its PCIe link and the filesystem type to the upload.\n"
    "  Never model names, serial numbers, capacity, paths or labels.\n"
    "  Change it later: llm-speed config storage-info on|off\n"
    "Share drive info? [y/N] "
)

# --------------------------------------------------------------------------- schema

BUS_VALUES = ("nvme", "sata", "usb", "virtual", "other")
FS_VALUES = ("ext4", "xfs", "btrfs", "apfs", "ntfs", "9p", "other")
UPLOAD_KEYS = ("bus", "pcie_gen", "pcie_width", "fs", "wsl_9p")
PCIE_GEN_MAX = 7
PCIE_WIDTH_MAX = 32

PROBE_TIMEOUT_S = 10.0  # each external command; any timeout leaves the field null

# R-131: the GGUF a `bench --preset` ssd_stream step's server reads (local only).
MODEL_PATH_ENV = "LLM_SPEED_MODEL_PATH_LOCAL"

SETTING_KEY = "share_storage_info"
PROMPTED_KEY = "share_storage_info_prompted"


def empty_block() -> dict[str, Any]:
    return {
        "bus": None,
        "pcie_gen": None,
        "pcie_width": None,
        "fs": None,
        "wsl_9p": False,
    }


def _norm_bus(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if v in BUS_VALUES else None


def _norm_fs(fstype: Any) -> str | None:
    if not isinstance(fstype, str) or not fstype.strip():
        return None
    f = fstype.strip().lower()
    if f in ("ext4", "xfs", "btrfs", "apfs", "9p"):
        return f
    if f in ("ntfs", "ntfs3"):
        return "ntfs"
    return "other"


def _int_or_none(v: Any, lo: int, hi: int) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v if lo <= v <= hi else None


def validate_block(block: Any) -> dict[str, Any] | None:
    """The exact upload shape, or None. Unknown keys (including any ``model``) are
    dropped, enums and ranges enforced. Its output always passes ``is_canonical``."""
    if not isinstance(block, dict):
        return None
    out = empty_block()
    out["bus"] = _norm_bus(block.get("bus"))
    fs = block.get("fs")
    out["fs"] = fs if isinstance(fs, str) and fs in FS_VALUES else None
    out["wsl_9p"] = block.get("wsl_9p") is True
    if out["bus"] == "nvme":
        out["pcie_gen"] = _int_or_none(block.get("pcie_gen"), 1, PCIE_GEN_MAX)
        out["pcie_width"] = _int_or_none(block.get("pcie_width"), 1, PCIE_WIDTH_MAX)
    return out


def _pcie_ok(v: Any, bus: Any, hi: int) -> bool:
    if v is None:
        return True
    return (
        bus == "nvme"
        and isinstance(v, int)
        and not isinstance(v, bool)
        and 1 <= v <= hi
    )


def is_canonical(block: Any) -> bool:
    """The Worker's acceptance rule (worker/src/storage-info.ts storageBlockCanonical):
    exactly the five keys, bus/fs in their lists or null, pcie_gen 1..7 / pcie_width
    1..32 ints (NVMe only) or null, wsl_9p a boolean. Anything else gets a 400."""
    if not isinstance(block, dict) or set(block) != set(UPLOAD_KEYS):
        return False
    bus, fs = block["bus"], block["fs"]
    if bus is not None and not (isinstance(bus, str) and bus in BUS_VALUES):
        return False
    if fs is not None and not (isinstance(fs, str) and fs in FS_VALUES):
        return False
    if not isinstance(block["wsl_9p"], bool):
        return False
    return _pcie_ok(block["pcie_gen"], bus, PCIE_GEN_MAX) and _pcie_ok(
        block["pcie_width"], bus, PCIE_WIDTH_MAX
    )


def _gen_from_speed(speed: Any) -> int | None:
    """'16.0 GT/s PCIe' / '8.0 GT/s' / 16 -> PCIe generation."""
    if isinstance(speed, (int, float)) and not isinstance(speed, bool):
        val = float(speed)
    elif isinstance(speed, str):
        m = re.search(r"(\d+(?:\.\d+)?)\s*GT/s", speed, re.I)
        if not m:
            return None
        val = float(m.group(1))
    else:
        return None
    table = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
    for k, g in table.items():
        if abs(val - k) < 0.01:
            return g
    return None


def _width_from(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return _int_or_none(v, 1, 32)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*x?(\d{1,2})\s*", v, re.I)
        if m:
            return _int_or_none(int(m.group(1)), 1, 32)
    return None


# --------------------------------------------------------------------------- Linux


def _read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _parse_mountinfo(text: str) -> list[tuple[str, str, str, str]]:
    """[(major:minor, mount_point, fstype, source)] from /proc/self/mountinfo text."""
    out = []
    for line in text.splitlines():
        parts = line.split(" - ", 1)
        if len(parts) != 2:
            continue
        left, right = parts[0].split(), parts[1].split()
        if len(left) < 5 or len(right) < 2:
            continue
        mp = left[4].replace("\\040", " ")
        out.append((left[2], mp, right[0], right[1]))
    return out


def _mount_for(path: str, mounts: list[tuple[str, str, str, str]]):
    best = None
    for m in mounts:
        mp = m[1].rstrip("/") or "/"
        if (
            path == mp
            or path.startswith(mp if mp.endswith("/") else mp + "/")
            or mp == "/"
        ):
            if best is None or len(mp) > len(best[1].rstrip("/") or "/"):
                best = m
    return best


_WSL_DRIVE_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])$")


def _sys_disk_for_devnum(devnum: str, sys_root: Path, _depth: int = 0) -> Path | None:
    """/sys/block/<disk> (resolved) for a 'major:minor', following one partition level
    and single-slave device-mapper / md stacks. None when ambiguous or unknown."""
    if _depth > 4 or not re.fullmatch(r"\d+:\d+", devnum or ""):
        return None
    node = sys_root / "dev" / "block" / devnum
    try:
        real = node.resolve(strict=True)
    except OSError:
        return None
    if (real / "partition").exists():
        real = real.parent
    slaves = real / "slaves"
    if slaves.is_dir():
        try:
            entries = sorted(slaves.iterdir())
        except OSError:
            entries = []
        if len(entries) == 1:
            d = _read(entries[0] / "dev")
            if d:
                return _sys_disk_for_devnum(d, sys_root, _depth + 1)
        if entries:
            return None  # striped / mirrored: no single drive
    return real


def _pci_link(dev: Path) -> tuple[int | None, int | None]:
    """Walk up from a block device's controller to the first dir with a PCIe link."""
    try:
        cur = dev.resolve(strict=True)
    except OSError:
        return None, None
    for _ in range(6):
        speed = _read(cur / "current_link_speed")
        width = _read(cur / "current_link_width")
        if speed is not None or width is not None:
            return _gen_from_speed(speed), _width_from(width)
        if cur.parent == cur:
            break
        cur = cur.parent
    return None, None


def probe_linux_disk(disk: Path) -> dict[str, Any]:
    """bus / PCIe link for a resolved /sys/devices/.../block/<disk> dir. The drive's
    model string is never read."""
    out: dict[str, Any] = {"bus": None, "pcie_gen": None, "pcie_width": None}
    name = disk.name
    real = str(disk)
    dev = disk / "device"
    vendor = (_read(dev / "vendor") or "").strip()
    if name.startswith("nvme"):
        out["bus"] = "nvme"
        # namespace -> controller (device) -> PCI function (device/device)
        gen, width = _pci_link(dev / "device")
        out["pcie_gen"], out["pcie_width"] = gen, width
    elif "/usb" in real:
        out["bus"] = "usb"
    elif (
        name.startswith(("vd", "xvd"))
        or "/virtio" in real
        or "VMBUS" in real.upper()
        or vendor.lower() in ("msft", "qemu", "vmware", "vbox")
    ):
        out["bus"] = "virtual"
    elif "/ata" in real or vendor.upper() == "ATA":
        out["bus"] = "sata"
    else:
        out["bus"] = "other"
    return out


def probe_linux(
    path: str,
    *,
    sys_root: Path = Path("/sys"),
    mountinfo: str | None = None,
    wsl: bool | None = None,
    windows_probe=None,
    vhdx_letter=None,
    file_devnum: str | None = None,
) -> dict[str, Any]:
    block = empty_block()
    real = os.path.realpath(path)
    if mountinfo is None:
        mountinfo = _read(Path("/proc/self/mountinfo")) or ""
    mounts = _parse_mountinfo(mountinfo)
    m = _mount_for(real, mounts)
    if wsl is None:
        from .agent.wsl import is_wsl

        wsl = is_wsl()
    if m is not None:
        devnum, mp, fstype, _src = m
        block["fs"] = _norm_fs(fstype)
        drive = _WSL_DRIVE_MOUNT_RE.match(mp)
        if wsl and drive and fstype.lower() in ("9p", "drvfs", "virtiofs"):
            block["wsl_9p"] = True
            block["fs"] = "9p" if fstype.lower() in ("9p", "drvfs") else "other"
            win = (windows_probe or probe_windows)(drive_letter=drive.group(1).upper())
            block.update({k: win.get(k) for k in ("bus", "pcie_gen", "pcie_width")})
            return block
    else:
        devnum = None
    # Block device: st_dev of the file, else (btrfs / overlay anonymous dev) the mount's.
    if file_devnum is not None:
        cand = [file_devnum]
    else:
        try:
            st = os.stat(real)
            cand = [f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"]
        except OSError:
            cand = []
    if devnum:
        cand.append(devnum)
    disk = None
    for c in cand:
        if c.startswith("0:"):
            continue
        disk = _sys_disk_for_devnum(c, sys_root)
        if disk is not None:
            break
    if disk is not None:
        info = probe_linux_disk(disk)
        block.update(info)
    if wsl and block["bus"] in (None, "virtual", "other"):
        # The distro's ext4 lives in a Hyper-V VHDX: report the physical drive that holds
        # it when Windows can tell us (only for the distro's own root disk), else
        # bus=virtual with no model.
        root = _mount_for("/", mounts)
        on_root_disk = bool(root and devnum and root[0] == devnum)
        letter = (vhdx_letter or wsl_vhdx_drive_letter)() if on_root_disk else None
        win = (windows_probe or probe_windows)(drive_letter=letter) if letter else {}
        if win.get("bus"):
            block.update({k: win.get(k) for k in ("bus", "pcie_gen", "pcie_width")})
        else:
            block.update({"bus": "virtual", "pcie_gen": None, "pcie_width": None})
    return block


# --------------------------------------------------------------------------- Windows / WSL

# Outputs ONLY BusType, the filesystem type and the PCIe link of the disk behind one
# drive letter. No model / FriendlyName (user-renamable, R-128), no serial, no size, no
# path leaves PowerShell. {L} is a validated A-Z letter.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$o = [ordered]@{ bus = $null; link_speed = $null; link_width = $null; fs = $null }
try {
  $v = Get-Volume -DriveLetter {L}
  $o.fs = [string]$v.FileSystem
  $d = Get-Partition -DriveLetter {L} | Get-Disk
  $p = Get-PhysicalDisk | Where-Object { [string]$_.DeviceId -eq [string]$d.Number } | Select-Object -First 1
  if ($p) { $o.bus = [string]$p.BusType }
  try {
    $iid = (([string]$d.Path) -replace '^\\\\\?\\', '' -replace '#\{[^}]*\}$', '') -replace '#', '\'
    $parent = (Get-PnpDeviceProperty -InstanceId $iid -KeyName 'DEVPKEY_Device_Parent').Data
    if ($parent -like 'PCI\*') {
      $o.link_speed = (Get-PnpDeviceProperty -InstanceId $parent -KeyName 'DEVPKEY_PciDevice_CurrentLinkSpeed').Data
      $o.link_width = (Get-PnpDeviceProperty -InstanceId $parent -KeyName 'DEVPKEY_PciDevice_CurrentLinkWidth').Data
    }
  } catch {}
} catch {}
$o | ConvertTo-Json -Compress
"""

# Get-PhysicalDisk BusType: numeric (MSFT_PhysicalDisk) or its display string.
_WIN_BUS = {
    "17": "nvme",
    "nvme": "nvme",
    "11": "sata",
    "sata": "sata",
    "7": "usb",
    "usb": "usb",
    "14": "virtual",
    "virtual": "virtual",
    "15": "virtual",
    "file backed virtual": "virtual",
}


def parse_windows_output(text: str) -> dict[str, Any]:
    """bus/pcie/fs from the PowerShell JSON. Any other key is ignored; any problem ->
    nulls."""
    out: dict[str, Any] = {
        "bus": None,
        "pcie_gen": None,
        "pcie_width": None,
        "fs": None,
    }
    try:
        data = json.loads((text or "").strip() or "null")
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    bus_raw = data.get("bus")
    if bus_raw is not None:
        out["bus"] = _WIN_BUS.get(str(bus_raw).strip().lower(), "other")
    if out["bus"] == "nvme":
        # DEVPKEY_PciDevice_CurrentLinkSpeed is the PCIe generation (1..6).
        sp = data.get("link_speed")
        out["pcie_gen"] = (
            _int_or_none(sp, 1, PCIE_GEN_MAX) if isinstance(sp, int) else None
        )
        out["pcie_width"] = _width_from(data.get("link_width"))
    out["fs"] = _norm_fs(data.get("fs"))
    return out


# R-129: only Windows' own PowerShell, by fixed path. Never PATH, never the current
# directory, never pwsh.
WSL_POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
_SYSTEMROOT_RE = re.compile(r'^[A-Za-z]:\\[^"]+$')


def _powershell(
    *, native_windows: bool | None = None, env=None, isfile=None
) -> str | None:
    is_file = isfile or os.path.isfile
    if native_windows is None:
        native_windows = sys.platform.startswith("win")
    if native_windows:
        root = (env if env is not None else os.environ).get("SystemRoot", "")
        if not _SYSTEMROOT_RE.match(root or ""):
            return None
        p = root.rstrip("\\") + r"\System32\WindowsPowerShell\v1.0\powershell.exe"
        return p if is_file(p) else None
    return WSL_POWERSHELL if is_file(WSL_POWERSHELL) else None


def _run(argv: list[str], timeout: float = PROBE_TIMEOUT_S) -> str | None:
    try:
        cp = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if cp.returncode != 0:
        return None
    return cp.stdout.decode("utf-8", errors="replace")


def probe_windows(*, drive_letter: str | None, runner=_run) -> dict[str, Any]:
    if not drive_letter or not re.fullmatch(r"[A-Z]", drive_letter):
        return {}
    ps = _powershell()
    if not ps:
        return {}
    script = _PS_SCRIPT.replace("{L}", drive_letter)
    text = runner([ps, "-NoProfile", "-NonInteractive", "-Command", script])
    return parse_windows_output(text or "")


_VHDX_PS = r"""
$ErrorActionPreference = 'Stop'
try {
  $k = Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | ForEach-Object { Get-ItemProperty $_.PSPath } | Where-Object { $_.DistributionName -eq '{D}' } | Select-Object -First 1
  if ($k -and $k.BasePath) { ([string]$k.BasePath).TrimStart('\\?\').Substring(0,1) }
} catch {}
"""


def wsl_vhdx_drive_letter(runner=_run) -> str | None:
    """Drive letter of the Windows volume holding this distro's ext4.vhdx, or None.
    Only the letter is kept; the base path never leaves this function."""
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", distro):
        return None
    ps = _powershell()
    if not ps:
        return None
    text = runner(
        [
            ps,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _VHDX_PS.replace("{D}", distro),
        ]
    )
    letter = (text or "").strip()[:1].upper()
    return letter if re.fullmatch(r"[A-Z]", letter) else None


def probe_native_windows(path: str, runner=_run) -> dict[str, Any]:
    block = empty_block()
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    m = re.fullmatch(r"([A-Za-z]):", drive or "")
    if not m:
        return block
    win = probe_windows(drive_letter=m.group(1).upper(), runner=runner)
    for k in ("bus", "pcie_gen", "pcie_width", "fs"):
        if win.get(k) is not None:
            block[k] = win[k]
    return block


# --------------------------------------------------------------------------- macOS


def parse_macos_profiler(text: str, whole_disk: str | None) -> dict[str, Any]:
    """bus / PCIe from `system_profiler SPNVMeDataType SPSerialATADataType -json`.
    Only ``bsd_name`` (to match the disk) and the link fields are read; model and
    serial are never read."""
    out: dict[str, Any] = {"bus": None, "pcie_gen": None, "pcie_width": None}
    try:
        data = json.loads(text or "{}")
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    drives: list[tuple[str, dict]] = []
    for key, bus in (("SPNVMeDataType", "nvme"), ("SPSerialATADataType", "sata")):
        for ctrl in data.get(key) or []:
            if not isinstance(ctrl, dict):
                continue
            for item in ctrl.get("_items") or []:
                if isinstance(item, dict):
                    drives.append((bus, item))
    pick = None
    if whole_disk:
        pick = next((d for d in drives if d[1].get("bsd_name") == whole_disk), None)
    if pick is None:
        return out
    bus, item = pick
    out["bus"] = bus
    if bus == "nvme":
        out["pcie_gen"] = _gen_from_speed(item.get("spnvme_linkspeed"))
        out["pcie_width"] = _width_from(item.get("spnvme_linkwidth"))
    return out


def parse_diskutil_plist(raw: bytes) -> tuple[str | None, str | None]:
    """(whole physical disk e.g. 'disk0', fs enum) from `diskutil info -plist PATH`."""
    try:
        info = plistlib.loads(raw)
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(info, dict):
        return None, None
    fs = _norm_fs(info.get("FilesystemType") or info.get("FilesystemName"))
    whole = None
    stores = info.get("APFSPhysicalStores")
    if isinstance(stores, list) and len(stores) == 1 and isinstance(stores[0], dict):
        whole = stores[0].get("APFSPhysicalStore")
    if not whole:
        whole = info.get("ParentWholeDisk")
    if isinstance(whole, str):
        m = re.match(r"(disk\d+)", whole)
        whole = m.group(1) if m else None
    else:
        whole = None
    return whole, fs


def probe_macos(path: str, runner=None) -> dict[str, Any]:
    block = empty_block()

    def _bytes(argv):
        try:
            cp = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=PROBE_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return cp.stdout if cp.returncode == 0 else None

    run_b = runner or _bytes
    mp = os.path.realpath(path)
    while not os.path.ismount(mp) and os.path.dirname(mp) != mp:
        mp = os.path.dirname(mp)
    raw = run_b(["diskutil", "info", "-plist", mp])
    whole, fs = parse_diskutil_plist(raw) if raw else (None, None)
    block["fs"] = fs
    prof = run_b(["system_profiler", "SPNVMeDataType", "SPSerialATADataType", "-json"])
    info = parse_macos_profiler(prof.decode("utf-8", "replace") if prof else "", whole)
    if info.get("bus"):
        block.update(info)
    else:
        block["bus"] = "other" if whole else None
    return block


# --------------------------------------------------------------------------- disk-streamed runs only


def streams_from_disk() -> bool:
    """True only for a run of an `ssd_stream: true` preset entry: the same signal that
    puts extras.ssd_stream = true in its run record (cli/ssd_stream.py upload_extras,
    set by `bench --preset` for that entry only). Anything else, or a CLI without
    ssd_stream support, is False."""
    try:
        from . import ssd_stream
    except ImportError:
        return False
    try:
        return ssd_stream.upload_extras().get("ssd_stream") is True
    except Exception:  # noqa: BLE001
        return False


def run_streams_from_disk(results) -> bool:
    """The run record check: some result has extras.ssd_stream exactly True."""
    for r in results or []:
        extras = r.get("extras") if isinstance(r, dict) else getattr(r, "extras", None)
        if isinstance(extras, dict) and extras.get("ssd_stream") is True:
            return True
    return False


# --------------------------------------------------------------------------- entry points


def model_path(driver, model) -> str | None:
    """A local path for the model being benchmarked, or None (hosted / remote server)."""
    if model is None or getattr(model, "backend", "") == "hosted-api":
        return None
    # R-131: `bench --preset` (cli/fit_sweep.py) runs its bench child against the
    # llama-server it started (LLAMA_CPP_SERVER_URL), and for an ssd_stream entry passes
    # that server's GGUF here. Local only: used for this probe, never uploaded or logged.
    local = os.environ.get(MODEL_PATH_ENV, "")
    if local:
        try:
            if Path(local).is_file():
                return local
        except (OSError, ValueError):
            pass
        return None
    if os.environ.get("LLAMA_CPP_SERVER_URL"):
        return None  # an external server: we don't know where it reads weights from
    cands: list[str] = []
    ident = getattr(model, "identifier", None)
    if isinstance(ident, str):
        cands.append(ident)
    extras = getattr(model, "extras", None) or {}
    for k in ("path", "snapshot_path"):
        if isinstance(extras.get(k), str):
            cands.append(extras[k])
    if getattr(model, "backend", "") == "ollama":
        cands.append(
            os.environ.get("OLLAMA_MODELS") or str(Path.home() / ".ollama" / "models")
        )
    for c in cands:
        try:
            p = Path(c).expanduser()
            if p.exists():
                return str(p)
        except (OSError, ValueError):
            continue
    return None


def probe(path: str | None) -> dict[str, Any] | None:
    """The validated storage block for the drive holding `path`, or None if there is
    no local path. Never raises; anything that fails stays null."""
    if not path:
        return None
    try:
        if sys.platform == "darwin":
            raw = probe_macos(path)
        elif sys.platform.startswith("win"):
            raw = probe_native_windows(path)
        elif sys.platform.startswith("linux"):
            raw = probe_linux(path)
        else:
            raw = empty_block()
    except Exception as exc:  # noqa: BLE001
        log.debug("storage probe failed: %s", exc)
        raw = empty_block()
    return validate_block(raw)


# --------------------------------------------------------------------------- consent


def _cfg():
    from . import credit

    return credit


def configured() -> bool:
    """The saved setting. Only JSON true counts (R-103 style)."""
    return _cfg().load_config().get(SETTING_KEY) is True


def was_prompted() -> bool:
    return _cfg().load_config().get(PROMPTED_KEY) is True


def set_configured(on: bool) -> None:
    c = _cfg()
    data = c.load_config()
    data[SETTING_KEY] = bool(on) is True
    data[PROMPTED_KEY] = True
    data.setdefault(PROMPTED_KEY + "_at", dt.datetime.now(dt.timezone.utc).isoformat())
    c.save_config(data)


def _in_agent_job() -> bool:
    from .agent_job import FD_ENV

    return bool(os.environ.get(FD_ENV))


def effective(args) -> bool:
    """Share storage info for this run? --strict-anon and agent jobs always say no;
    then the per-run flag; then the saved setting (unset = off)."""
    if getattr(args, "strict_anon", False) or _in_agent_job():
        return False
    if getattr(args, "no_share_storage_info", False):
        return False
    if getattr(args, "share_storage_info", False):
        return True
    return configured()


def check_flags(args) -> str | None:
    """Error text for a forbidden combination, else None. A note is printed (not an
    error) when --strict-anon overrides --share-storage-info."""
    if getattr(args, "share_storage_info", False) and getattr(
        args, "no_share_storage_info", False
    ):
        return "--share-storage-info and --no-share-storage-info can't be used together"
    if getattr(args, "share_storage_info", False) and getattr(
        args, "strict_anon", False
    ):
        sys.stderr.write(
            "note: --strict-anon never sends drive info; --share-storage-info is ignored.\n"
        )
    return None


def should_ask(args, *, stdin=None, stdout=None) -> bool:
    """Only a human at a terminal, never asked before, no flag given, not strict-anon,
    not --yes / CI / an agent job."""
    c = _cfg()
    if getattr(args, "strict_anon", False) or _in_agent_job():
        return False
    if getattr(args, "share_storage_info", False) or getattr(
        args, "no_share_storage_info", False
    ):
        return False
    if getattr(args, "yes", False) or getattr(args, "no_prompt", False):
        return False
    if c.ci_detected() or not c.interactive(stdin, stdout):
        return False
    data = c.load_config()
    if data.get(PROMPTED_KEY) is True or SETTING_KEY in data:
        return False
    return True


def maybe_ask(args, *, stdin=None, stdout=None) -> bool:
    """Ask the one-time question (default No). Returns the answer; records it."""
    if not should_ask(args, stdin=stdin, stdout=stdout):
        return False
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    out.write("\n" + STORAGE_PROMPT_TEXT)
    out.flush()
    try:
        line = inp.readline()
    except KeyboardInterrupt:
        out.write("\n")
        line = ""
    yes = (line or "").strip().lower() in ("y", "yes")
    set_configured(yes)
    out.write(
        "  Drive info will be added to uploads.\n"
        if yes
        else "  Not sharing drive info.\n"
    )
    out.flush()
    return yes


def cmd_config_storage(args) -> int:
    value = getattr(args, "value", None)
    if getattr(args, "clear", False):
        print("error: storage-info takes on or off, not --clear", file=sys.stderr)
        return 2
    if value is None:
        state = "on" if configured() else "off"
        print(f"Storage info: {state}. Change: llm-speed config storage-info on|off")
        return 0
    v = value.strip().lower()
    if v not in ("on", "off"):
        print("error: use on or off", file=sys.stderr)
        return 2
    set_configured(v == "on")
    print(
        "Storage info on: runs that stream weights from disk include the drive's "
        "connection, PCIe link and filesystem (never with --strict-anon)."
        if v == "on"
        else "Storage info off."
    )
    return 0
