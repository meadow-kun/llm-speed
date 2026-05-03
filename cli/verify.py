"""``llm-speed verify`` - prove the wheel matches the published artifact.

Concretely: compute the SHA-256 of a local wheel and compare it against the
sidecar checksum file the orchestrator publishes next to the wheel on
llm-speed.com. If they match, the user can show a friend that the binary they
are about to run is byte-identical to a publicly-served artifact whose source
is auditable on GitHub.

What this does NOT prove:

* That the published wheel was built from the commit it claims to be from -
  reproducible-build attestation lives elsewhere. We *do* surface the
  ``llm-speed`` package version + the linked git tag / commit so the user can
  follow the chain manually.
* That the wheel hasn't been swapped on the CDN since this run - the
  comparison is point-in-time. Re-running ``verify`` is the way to recheck.

The verdict is printed in plain English so the user can paste it as evidence
of provenance.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from .ui.theme import (
    C_BRAND,
    C_ERR,
    C_HINT,
    C_OK,
    C_WARN,
    REPO_URL,
    SITE_URL,
    get_console,
)

log = logging.getLogger("cli.verify")

DEFAULT_DIST_BASE = f"{SITE_URL}/dist"
WHEEL_NAME_DEFAULT = "llm_speed-0.0.1-py3-none-any.whl"

# Independent trust domain. The sidecar at this URL must match the one served
# from llm-speed.com — disagreement means one of the two distribution chains
# was compromised.
GH_RELEASES_BASE = "https://github.com/meadow-kun/llm-speed/releases/download"

# The release tag the cross-check URL is anchored to. Bumped on each release
# alongside the wheel. Kept separate from the package's __version__ string
# (which is "0.0.1-dev" while we're pre-1.0) because the GH release tags
# follow a v<major>.<minor>.<patch> convention and we don't want to retag
# every dev push.
RELEASE_TAG = "v1.0.2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Stream the file through sha256 and return the hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            buf = fp.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_checksum(text: str) -> str | None:
    """Accept either ``<hex>`` alone or ``<hex>  <filename>`` (sha256sum format).

    Returns the lowercased hex digest, or None if no plausible 64-char hex run
    is present.
    """
    if not text:
        return None
    for line in text.splitlines():
        toks = line.strip().split()
        if not toks:
            continue
        cand = toks[0].lower()
        if len(cand) == 64 and all(c in "0123456789abcdef" for c in cand):
            return cand
    return None


def fetch_checksum(url: str, *, timeout: float = 15.0) -> str:
    """Fetch the sidecar ``.sha256`` file. Raises on HTTP / network failure."""
    # GitHub Releases serves assets via a 302 redirect to objects.githubusercontent.com,
    # so we must follow redirects or the cross-check URL aborts before reading the body.
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    return resp.text


def installed_wheel_path() -> Path | None:
    """Best effort: locate a local wheel for this install. Looks in the
    common dev (``./dist``) and pipx (``~/.local/pipx/.cache``,
    ``~/.local/share/pipx/.cache``) layouts. Returns None when no local
    wheel can be located — callers can then fall back to downloading the
    published wheel and checking it against the sidecar."""
    home = Path.home()
    candidates = [
        Path.cwd() / "dist" / WHEEL_NAME_DEFAULT,
        Path(__file__).parent.parent / "dist" / WHEEL_NAME_DEFAULT,
        # pipx wheel cache (Linux / macOS default layouts)
        home / ".local" / "pipx" / ".cache" / WHEEL_NAME_DEFAULT,
        home / ".local" / "share" / "pipx" / ".cache" / WHEEL_NAME_DEFAULT,
        # pip download cache (less common but cheap to check)
        home / ".cache" / "pip" / "wheels" / WHEEL_NAME_DEFAULT,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def fetch_published_wheel_to_temp(
    dist_base: str = DEFAULT_DIST_BASE, wheel_name: str = WHEEL_NAME_DEFAULT,
) -> Path:
    """Download the published wheel to a temp file so we can hash it.

    This is the fallback when no local wheel is on disk (e.g. pipx
    installed and discarded its build cache). Comparing the freshly-
    downloaded wheel against its sidecar proves "what the CDN serves
    right now matches its declared digest" — it doesn't tell you
    anything about what's already installed in your venv, but it does
    establish trust in the publishing pipeline.
    """
    import tempfile

    # Cap the download size so an attacker-controlled --dist-base can't make
    # us write a 10 GB wheel to /tmp. The real wheel is ~140 KB; 50 MB is a
    # generous ceiling that still aborts long before disk-fill is plausible.
    MAX_WHEEL_BYTES = 50 * 1024 * 1024  # 50 MB

    url = f"{dist_base.rstrip('/')}/{wheel_name}"
    fd, tmp = tempfile.mkstemp(prefix="llm-speed-verify-", suffix=".whl")
    import os as _os

    _os.close(fd)
    out = Path(tmp)
    written = 0
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    written += len(chunk)
                    if written > MAX_WHEEL_BYTES:
                        # Clean up the partial file before raising.
                        try:
                            f.close()
                            out.unlink(missing_ok=True)
                        except OSError:
                            pass
                        raise RuntimeError(
                            f"wheel download exceeded {MAX_WHEEL_BYTES} bytes "
                            f"from {url} — refusing to continue"
                        )
                    f.write(chunk)
    return out


def installed_version() -> str | None:
    """Return the installed ``llm-speed`` version string via the package itself."""
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return None


def installed_pip_show() -> str | None:
    """Return ``pip show llm-speed`` output. Used purely for display."""
    pip = shutil.which("pip") or shutil.which("pip3")
    if not pip:
        return None
    try:
        proc = subprocess.run(
            [pip, "show", "llm-speed"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("pip show failed: %s", exc)
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    matched: bool
    local_path: Path
    local_hash: str
    expected_hash: str
    sidecar_url: str
    cross_check_url: str | None = None
    note: str = ""


def gh_releases_sidecar_url(version_tag: str, wheel_name: str) -> str:
    """The sidecar's GitHub-Releases URL for a given tag + wheel name. Tag
    is e.g. ``v1.0.1`` (the convention used by ``release_publish.sh``)."""
    return f"{GH_RELEASES_BASE}/{version_tag}/{wheel_name}.sha256"


def verify_wheel(
    wheel_path: Path,
    *,
    dist_base: str = DEFAULT_DIST_BASE,
    sidecar_name: str | None = None,
    cross_check_url: str | None = None,
) -> Verdict:
    """Compute local sha256 and compare against the sidecar at dist_base.

    Raises ``RuntimeError`` when the network lookup fails so the caller can
    surface that distinct from a hash-mismatch verdict.

    ``sidecar_name`` lets the caller override the sidecar's URL filename when
    ``wheel_path.name`` isn't the publishable name (e.g. when we downloaded
    the wheel to a tempfile with a generated name and need to query the
    canonical ``llm_speed-0.0.1-py3-none-any.whl.sha256``).

    ``cross_check_url`` is an optional second sidecar URL (typically the
    GitHub-Releases mirror) — when provided, we fetch BOTH and require they
    agree before declaring a match. Disagreement means one of the two
    distribution chains was tampered with; we surface that as a hard fail.
    """
    wheel_path = wheel_path.resolve()
    if not wheel_path.exists():
        raise RuntimeError(f"wheel not found: {wheel_path}")

    local = sha256_file(wheel_path)
    sidecar_filename = sidecar_name or wheel_path.name
    sidecar = f"{dist_base.rstrip('/')}/{sidecar_filename}.sha256"
    raw = fetch_checksum(sidecar)
    expected = parse_checksum(raw)
    if expected is None:
        raise RuntimeError(f"sidecar at {sidecar} does not contain a sha256 digest")

    # Cross-check: an attacker who compromises only one distribution chain
    # can swap the wheel + sidecar there. Independently fetching the same
    # sidecar from a separate domain (typically GitHub Releases) and
    # requiring both to match closes that single-point-of-failure.
    #
    # Fail CLOSED on cross-check failures (per pentest_2026-05-01 P-1):
    # falling back to "trust llm-speed.com only" defeats the entire point of
    # the dual-domain check. If the user genuinely can't reach GitHub
    # Releases (offline / corp firewall), they can pass cross_check_url=None
    # explicitly, but they should know they're degrading the trust chain.
    if cross_check_url:
        try:
            cross_raw = fetch_checksum(cross_check_url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cross-check fetch failed at {cross_check_url}: {exc}. "
                "Refusing to verify against a single distribution chain — "
                "an attacker who controlled only the primary CDN could swap "
                "the wheel + sidecar atomically. If you genuinely cannot "
                "reach the cross-check URL (offline / corp firewall), "
                "pass --no-cross-check to acknowledge the degraded trust."
            ) from exc
        cross_expected = parse_checksum(cross_raw)
        if cross_expected is None:
            raise RuntimeError(
                f"cross-check sidecar at {cross_check_url} did not contain a "
                "valid sha256 digest"
            )
        if cross_expected != expected:
            raise RuntimeError(
                f"sidecar disagreement: {sidecar} says {expected}, "
                f"{cross_check_url} says {cross_expected}. "
                "One of the distribution chains has been tampered with — "
                "do not trust this binary."
            )

    return Verdict(
        matched=(local == expected),
        local_path=wheel_path,
        local_hash=local,
        expected_hash=expected,
        sidecar_url=sidecar,
        cross_check_url=cross_check_url,
    )


def cmd_verify(args) -> int:
    """``llm-speed verify [<path>]`` entry point."""
    cons = get_console()
    cons.print(
        f"[{C_BRAND}]llm-speed verify[/] - prove this binary matches the public artifact"
    )
    cons.print()

    target = getattr(args, "wheel", None)
    dist_base = getattr(args, "dist_base", None) or DEFAULT_DIST_BASE
    fetched_temp: Path | None = None
    if target:
        wheel = Path(target).expanduser()
    else:
        wheel = installed_wheel_path()
        if wheel is None:
            # No local wheel — fetch the published one and check it against
            # its own sidecar. This establishes trust in the publishing
            # pipeline; it does not prove what's already in the user's venv
            # matches, but it's a useful default for pipx users.
            cons.print(
                f"  [dim]no local wheel found — fetching from {dist_base.rstrip('/')}/[/]"
            )
            try:
                wheel = fetch_published_wheel_to_temp(dist_base)
                fetched_temp = wheel
            except Exception as exc:  # noqa: BLE001
                cons.print(f"[{C_ERR}]error:[/] download failed: {exc}")
                cons.print(
                    f"  [dim]pass --wheel <path>, e.g.[/] llm-speed verify --wheel ./dist/{WHEEL_NAME_DEFAULT}"
                )
                return 1

    cons.print(f"  local wheel    : {wheel}")

    # When we fetched the wheel ourselves, the on-disk filename is a tempname
    # (e.g. /tmp/llm-speed-verify-xxxxx.whl) — but the published sidecar lives
    # at <dist>/<canonical-wheel-name>.sha256, NOT under the tempname.
    sidecar_name = WHEEL_NAME_DEFAULT if fetched_temp is not None else None
    try:
        # Cross-check against the GitHub-Releases mirror by default. We anchor
        # to the hardcoded RELEASE_TAG rather than the package's __version__
        # because dev builds carry a "-dev" suffix that doesn't correspond to
        # any real GH release. --no-cross-check disables the second domain
        # — degrades to single-source trust, only valid for offline / corp
        # firewall scenarios.
        if getattr(args, "no_cross_check", False):
            cross_check = None
            cons.print(
                f"  [{C_WARN}]cross-check disabled[/] — single-domain trust only"
            )
        else:
            cross_check = gh_releases_sidecar_url(
                RELEASE_TAG, sidecar_name or wheel.name,
            )
        verdict = verify_wheel(
            wheel,
            dist_base=dist_base,
            sidecar_name=sidecar_name,
            cross_check_url=cross_check,
        )
    except RuntimeError as exc:
        cons.print(f"[{C_ERR}]error:[/] {exc}")
        cons.print(f"  [dim]could not reach sidecar at {dist_base.rstrip('/')}[/]")
        return 1

    cons.print(f"  local sha256   : {verdict.local_hash}")
    cons.print(f"  expected sha256: {verdict.expected_hash}")
    cons.print(f"  sidecar URL    : {verdict.sidecar_url}")
    if verdict.cross_check_url:
        cons.print(f"  cross-check OK : {verdict.cross_check_url}")
    cons.print()

    if verdict.matched:
        cons.print(
            f"[{C_OK}]MATCH[/]. The bytes you are about to run are byte-identical "
            f"to the wheel served at\n      {dist_base.rstrip('/')}/{wheel.name}"
        )
        cons.print(f"      Source code (audit it yourself): {REPO_URL}")
    else:
        cons.print(
            f"[{C_ERR}]MISMATCH[/]. Local wheel does NOT match the published artifact.\n"
            f"      Do not trust this binary; redownload from {SITE_URL} or build from source."
        )

    # Bonus: surface the version + the show output so the user has the full
    # provenance picture in one block.
    ver = installed_version()
    if ver:
        cons.print()
        cons.print(f"  installed version: [{C_HINT}]llm-speed {ver}[/]")
    show = installed_pip_show()
    if show:
        cons.print("  [dim]pip show llm-speed:[/]")
        for line in show.splitlines():
            cons.print(f"    [dim]{line}[/]")

    # Clean up the downloaded wheel if we fetched it ourselves.
    if fetched_temp is not None:
        try:
            fetched_temp.unlink(missing_ok=True)
        except OSError:
            pass

    return 0 if verdict.matched else 1
