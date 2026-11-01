"""Pinned, verified install of the official llama.cpp Linux x86_64 CUDA prebuilt for
`llm-speed agent` auto-install (0.0.13, Security Desk R-123).

Only the build in PINNED_ASSETS is ever installed this way; a preset's min_llama_build
never picks the version (newer than the pin: backend_missing). Each archive is fetched
over https only (redirects too) into a staging dir under prebuilt_root(), its size and
sha256 are checked before anything is unpacked, it is unpacked with tarfile's "data"
filter (no absolute paths, no "..", no links leaving the tree, no devices, no setuid)
into the staging dir, and the result is moved into place with os.replace. Any failure
removes the staging dir and leaves nothing half-installed."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

PINNED_BUILD = 12020
PINNED_TAG = f"b{PINNED_BUILD}"
RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"

# (asset name, size in bytes, sha256). GitHub's published digests for release b12020,
# re-confirmed by CLI Lab on 2026-10-31 by downloading each asset and hashing it.
# Changing a value or adding a build is a new trusted download: Security review first.
PINNED_ASSETS: dict[str, tuple[tuple[str, int, str], tuple[str, int, str]]] = {
    "12.8": (
        (
            "llama-b12020-bin-ubuntu-cuda-12.8-x64.tar.gz",
            182781376,
            "0170c61c0ddf6208215a68b36f78f45b1641631317736c093f3faa40f05f8e85",
        ),
        (
            "cudart-llama-b12020-bin-ubuntu-cuda-12.8-x64.tar.gz",
            594372521,
            "a17d59ed20873a82cbf2ff6c651d5bb432689d8569b4af0b6b1c7e35aeb001ba",
        ),
    ),
    "13.4": (
        (
            "llama-b12020-bin-ubuntu-cuda-13.4-x64.tar.gz",
            164195754,
            "a9323978129be5948665e8cff614548b602d3c05cc2ea0ad2c7b905eb065ce77",
        ),
        (
            "cudart-llama-b12020-bin-ubuntu-cuda-13.4-x64.tar.gz",
            440232630,
            "22ebd98c96c5156f00731d1327f1fe40f0d1cce4ff6d51ac4c82845b1424b0e1",
        ),
    ),
}


class InstallError(Exception):
    pass


def install_dir(root: Path, cuda: str) -> Path:
    return root / f"{PINNED_TAG}-cuda-{cuda}"


def server_path(root: Path, cuda: str) -> Path:
    return install_dir(root, cuda) / f"llama-{PINNED_TAG}" / "llama-server"


def newer_than_pin(min_build) -> bool:
    try:
        return bool(min_build) and int(min_build) > PINNED_BUILD
    except (TypeError, ValueError):
        return True  # unreadable minimum: never guess a version


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not str(newurl).startswith("https://"):
            raise InstallError(f"refused a non-https redirect ({code})")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_https(url: str):
    if not url.startswith("https://"):
        raise InstallError("refused a non-https URL")
    opener = urllib.request.build_opener(_HttpsOnly())
    return opener.open(
        urllib.request.Request(url, headers={"User-Agent": "llm-speed"}), timeout=60
    )


def fetch_verified(
    url: str, dest: Path, size: int, sha256: str, *, opener: Callable = _open_https
) -> None:
    """Stream url to dest, then check size + sha256. On any mismatch or error the file
    is deleted and InstallError raised; nothing is unpacked from it."""
    h, n = hashlib.sha256(), 0
    try:
        with opener(url) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > size:
                    raise InstallError(
                        f"{dest.name}: larger than the pinned {size} bytes"
                    )
                h.update(chunk)
                f.write(chunk)
        if n != size:
            raise InstallError(f"{dest.name}: {n} bytes, pinned {size}")
        if h.hexdigest() != sha256:
            raise InstallError(f"{dest.name}: sha256 mismatch (not unpacked)")
    except InstallError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 (network, TLS, disk)
        dest.unlink(missing_ok=True)
        raise InstallError(
            f"{dest.name}: download failed ({type(exc).__name__})"
        ) from None


def _strip_first(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    parts = member.name.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None  # the top-level directory itself
    return tarfile.data_filter(member.replace(name=parts[1], deep=False), path)


def safe_extract(archive: Path, dest: Path, *, strip_first: bool = False) -> None:
    """Unpack with the stdlib "data" filter (refuses absolute paths, "..", links that
    leave dest, device files; clears setuid). Pythons without it: refused."""
    if not hasattr(tarfile, "data_filter"):
        raise InstallError("this Python's tarfile has no safe extraction filter")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest, filter=_strip_first if strip_first else "data")
    except tarfile.TarError as exc:
        raise InstallError(f"{archive.name}: refused ({exc})") from None


def install(
    cuda: str,
    root: Path,
    *,
    opener: Callable = _open_https,
    run: Callable = subprocess.run,
    say: Callable[[str], None] = print,
) -> Path:
    """Install the pinned build for `cuda` under root; returns the full llama-server
    path (to run by that path, never via PATH). Raises InstallError."""
    if cuda not in PINNED_ASSETS:
        raise InstallError(f"no pinned llama.cpp {PINNED_TAG} build for CUDA {cuda}")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    try:
        files = []
        for name, size, sha in PINNED_ASSETS[cuda]:
            say(f"  downloading {name} ({size / 1e6:.0f} MB, sha256 pinned)")
            f = staging / name
            fetch_verified(
                f"{RELEASES}/{PINNED_TAG}/{name}", f, size, sha, opener=opener
            )
            files.append(f)
        out = staging / "out"
        safe_extract(files[0], out)
        sub = out / f"llama-{PINNED_TAG}"
        if not sub.is_dir():
            raise InstallError(f"{files[0].name}: no llama-{PINNED_TAG}/ inside")
        # CUDA runtime next to llama-server (its RUNPATH is $ORIGIN)
        safe_extract(files[1], sub, strip_first=True)
        exe = sub / "llama-server"
        if not exe.is_file() or exe.is_symlink():
            raise InstallError("the archive has no llama-server")
        for f in files:
            f.unlink()
        final = install_dir(root, cuda)
        if final.exists():
            os.replace(final, staging / "previous")
        os.replace(out, final)
        server = server_path(root, cuda)
        rc = run([str(server), "--version"], check=False, timeout=60).returncode
        if rc != 0:
            os.replace(final, staging / "failed")  # roll back
            if (staging / "previous").exists():
                os.replace(staging / "previous", final)
            raise InstallError(f"{server} --version exited {rc}")
        return server
    finally:
        shutil.rmtree(staging, ignore_errors=True)
