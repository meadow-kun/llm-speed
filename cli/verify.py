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

# Independent trust domain. The sidecar at this URL is a best-effort
# cross-check against PyPI — disagreement means one of the two
# distribution chains was compromised, but 404 is acceptable (it just
# means the CDN+GH-Releases legs aren't published for this version).
GH_RELEASES_BASE = "https://github.com/meadow-kun/llm-speed/releases/download"


def wheel_name_for(version: str) -> str:
    """The publishable wheel filename for a given version string.

    Matches what `python -m build` produces and what PyPI / the CDN
    serves. Centralised so a version bump doesn't drift filenames in
    five places.
    """
    return f"llm_speed-{version}-py3-none-any.whl"


def release_tag_for(version: str) -> str:
    """The GitHub Releases tag name for a given version string.

    Convention: `v<version>`. The 2026-05-07 audit (F-13) flagged that
    a separate `RELEASE_TAG = "v1.0.2"` hardcode let the GH-Releases
    leg drift independently of `__version__`. The fix is to derive it.
    If a future release uses a different tag scheme (e.g. a date-based
    "release/2026-05" prefix), the *single* place to change is here.
    """
    return f"v{version}"


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


# F-30: PyPI publishes a per-file `.sha256` digest in its JSON API at
# `https://pypi.org/pypi/<pkg>/<ver>/json`. We fold this into the
# trust-chain check so verify is 3-of-3 (CDN + GH Releases + PyPI).
PYPI_PROJECT_NAME = "llm-speed"


def fetch_sigstore_bundle(
    dist_base: str, wheel_name: str, *, timeout: float = 20.0
) -> bytes | None:
    """Fetch the Sigstore bundle for `wheel_name` from `<dist_base>/<wheel_name>.sigstore`.

    Returns the raw bundle bytes or None if the bundle isn't published.
    Raises on network-level failures (so the caller can distinguish
    "no bundle yet" from "couldn't reach the server").
    """
    url = f"{dist_base.rstrip('/')}/{wheel_name}.sigstore"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    if len(resp.content) > 256 * 1024:
        raise RuntimeError(f"sigstore bundle at {url} unexpectedly large")
    return resp.content


# V-4: identity claims that the Sigstore bundle's signing certificate
# MUST match. These are the OIDC subject + issuer that GitHub mints
# for our `release-sign.yml` workflow. A bundle signed by any other
# identity is a forgery — even if the Rekor log has it.
SIGSTORE_EXPECTED_ISSUER = "https://token.actions.githubusercontent.com"
SIGSTORE_EXPECTED_IDENTITY = "https://github.com/meadow-kun/llm-speed-web/.github/workflows/release-pypi.yml@refs/heads/main"


def verify_sigstore_bundle(wheel_path: Path, bundle_bytes: bytes) -> tuple[bool, str]:
    """Verify a Sigstore bundle against the wheel.

    Returns (ok, message). When the `sigstore` package isn't installed,
    returns (False, "<package missing>") — the caller treats this as
    informational, not a hard fail, so users without the optional dep
    aren't worse off than before V-4.
    """
    try:
        from sigstore.models import Bundle  # type: ignore
        from sigstore.verify import Verifier  # type: ignore
        from sigstore.verify import policy as _policy
    except ImportError:
        return False, (
            "sigstore-python not installed; install with `pip install sigstore` "
            "to enable transparency-log verification of wheels"
        )
    try:
        bundle = Bundle.from_json(bundle_bytes)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not parse sigstore bundle: {exc}"
    verifier = Verifier.production()
    try:
        # Match identity by prefix — every release-sign.yml run produces a
        # subject of the form `https://github.com/.../release-sign.yml@refs/tags/v0.0.1`.
        identity_policy = _policy.Identity(
            identity=SIGSTORE_EXPECTED_IDENTITY,
            issuer=SIGSTORE_EXPECTED_ISSUER,
        )
        with open(wheel_path, "rb") as f:
            blob = f.read()
        verifier.verify_artifact(blob, bundle, identity_policy)
        return True, "sigstore verification OK"
    except Exception as exc:  # noqa: BLE001
        return False, f"sigstore verification failed: {exc}"


def fetch_pypi_sha256(
    project: str, version: str, wheel_name: str, *, timeout: float = 15.0
) -> str | None:
    """Return the sha256 PyPI advertises for `<wheel_name>` of `<project>@<version>`.

    Returns None when:
      - the project / version is not on PyPI yet (404),
      - PyPI returns malformed JSON,
      - the requested wheel filename isn't in the version's `urls[]`.

    Raises RuntimeError on transport-level failures so the caller can
    distinguish "PyPI doesn't know about this version" (None) from
    "we couldn't reach PyPI" (raise).
    """
    import json as _json

    url = f"https://pypi.org/pypi/{project}/{version}/json"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    try:
        data = _json.loads(resp.text)
    except _json.JSONDecodeError as exc:
        raise RuntimeError(f"PyPI returned non-JSON from {url}: {exc}") from exc
    for entry in data.get("urls") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("filename") != wheel_name:
            continue
        digests = entry.get("digests") or {}
        sha = digests.get("sha256")
        if isinstance(sha, str) and len(sha) == 64:
            return sha.lower()
    return None


def installed_wheel_path() -> Path | None:
    """Best effort: locate a local wheel for this install. Looks in the
    common dev (``./dist``) and pipx (``~/.local/pipx/.cache``,
    ``~/.local/share/pipx/.cache``) layouts. Returns None when no local
    wheel can be located — callers can then fall back to downloading the
    published wheel and checking it against the sidecar."""
    home = Path.home()
    ver = installed_version() or "0.0.0"
    name = wheel_name_for(ver)
    candidates = [
        Path.cwd() / "dist" / name,
        Path(__file__).parent.parent / "dist" / name,
        # pipx wheel cache (Linux / macOS default layouts)
        home / ".local" / "pipx" / ".cache" / name,
        home / ".local" / "share" / "pipx" / ".cache" / name,
        # pip download cache (less common but cheap to check)
        home / ".cache" / "pip" / "wheels" / name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def fetch_published_wheel_to_temp(
    dist_base: str = DEFAULT_DIST_BASE,
    wheel_name: str | None = None,
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

    if wheel_name is None:
        ver = installed_version() or "0.0.0"
        wheel_name = wheel_name_for(ver)
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
    pypi_url: str | None = None  # F-30: third-leg PyPI digest URL when consulted
    pypi_sha256: str | None = None  # F-30: digest PyPI advertised
    pypi_status: str = (
        "skipped"  # one of {"skipped", "agreed", "not_published", "disagreed"}
    )
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
    pypi_check: bool = True,
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

    # Post-2026-05-12 (F-13 fix): PyPI is the primary trust anchor; the
    # CDN sidecar at `<dist_base>/<wheel>.sha256` is a best-effort
    # cross-check. A 404 here is non-fatal (the CDN doesn't have to host
    # a separate sidecar when PyPI's per-file digest covers the same
    # ground). A 200 with a disagreeing digest IS fatal — that's tampering.
    expected: str | None = None
    try:
        raw = fetch_checksum(sidecar)
    except RuntimeError:
        # CDN leg unavailable — defer to PyPI as primary, below.
        raw = ""
    if raw:
        expected = parse_checksum(raw)
        if expected is None:
            raise RuntimeError(f"sidecar at {sidecar} does not contain a sha256 digest")

    # Cross-check: an attacker who compromises only one distribution chain
    # can swap the wheel + sidecar there. Independently fetching the same
    # sidecar from a separate domain (typically GitHub Releases) closes
    # that single-point-of-failure when the leg is published.
    #
    # As of 2026-05-12 this leg is best-effort, not load-bearing. A 404 means
    # the GH-Releases mirror hasn't been published for this version — fall
    # through to PyPI (which IS published). A 200 with disagreement is still
    # a hard fail.
    if cross_check_url:
        try:
            cross_raw = fetch_checksum(cross_check_url)
        except RuntimeError:
            cross_raw = ""
        if cross_raw:
            cross_expected = parse_checksum(cross_raw)
            if cross_expected is None:
                raise RuntimeError(
                    f"cross-check sidecar at {cross_check_url} did not contain a "
                    "valid sha256 digest"
                )
            if expected is None:
                # CDN was 404 but GH-Releases agrees; promote it to expected.
                expected = cross_expected
            elif cross_expected != expected:
                raise RuntimeError(
                    f"sidecar disagreement: {sidecar} says {expected}, "
                    f"{cross_check_url} says {cross_expected}. "
                    "One of the distribution chains has been tampered with — "
                    "do not trust this binary."
                )

    # F-30: optional third-leg cross-check against PyPI's published
    # per-file digest. PyPI is an independent root of trust from CDN +
    # GitHub Releases — its digests are part of the index payload that
    # `pip` consumes during normal `pip install llm-speed`. Folding it
    # in here makes the verify chain 3-of-3 once PyPI publish lands.
    pypi_url: str | None = None
    pypi_sha: str | None = None
    pypi_status = "skipped"
    if pypi_check:
        ver = installed_version()
        wheel_basename = sidecar_filename
        if ver:
            pypi_url = f"https://pypi.org/pypi/{PYPI_PROJECT_NAME}/{ver}/json"
            try:
                pypi_sha = fetch_pypi_sha256(PYPI_PROJECT_NAME, ver, wheel_basename)
            except RuntimeError:
                # Network failure to PyPI is non-fatal when the CDN/GH legs
                # established `expected`; if BOTH the CDN+GH and PyPI legs
                # are unreachable, the caller surfaces "matched=False with
                # expected=None" — caller decides whether to fail closed.
                pypi_status = "skipped"
                pypi_sha = None
            else:
                if pypi_sha is None:
                    pypi_status = "not_published"
                elif expected is None:
                    # CDN+GH 404'd; PyPI is the only published digest.
                    # Promote it to the canonical expected — this is the
                    # post-F-13 normal path.
                    expected = pypi_sha
                    pypi_status = "agreed"
                elif pypi_sha != expected:
                    raise RuntimeError(
                        f"PyPI sidecar disagreement: PyPI advertises "
                        f"sha256={pypi_sha} for {wheel_basename} at version "
                        f"{ver}, but {sidecar} says {expected}. One of the "
                        "distribution chains has been tampered with — do not "
                        "trust this binary."
                    )
                else:
                    pypi_status = "agreed"

    if expected is None:
        # Every leg (CDN sidecar, GH-Releases mirror, PyPI per-file digest)
        # was unreachable or unpublished. Cannot verify — fail closed.
        raise RuntimeError(
            f"verify: no published sha256 digest found for this wheel.\n"
            f"  CDN sidecar  : {sidecar} (404 or unreachable)\n"
            f"  GH Releases  : {cross_check_url or '(not consulted)'}\n"
            f"  PyPI         : {pypi_url or '(no installed version)'} ({pypi_status})\n"
            "  Either the published artifact has been withdrawn, or all "
            "three trust roots are unreachable. Refusing to declare 'match' "
            "against nothing."
        )

    return Verdict(
        matched=(local == expected),
        local_path=wheel_path,
        local_hash=local,
        expected_hash=expected,
        sidecar_url=sidecar,
        cross_check_url=cross_check_url,
        pypi_url=pypi_url,
        pypi_sha256=pypi_sha,
        pypi_status=pypi_status,
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
    no_cross_check = bool(getattr(args, "no_cross_check", False))
    # V-13: a hostile install script could pass both `--no-cross-check`
    # AND `--dist-base <attacker-controlled-url>` to degrade to
    # single-source trust — verify would then only consult the attacker's
    # own sidecar. Refuse the combination outright. `--no-cross-check`
    # is intended for offline / corporate-firewall users on the DEFAULT
    # dist-base, not as a way to escape the dual-domain anchor.
    if no_cross_check and dist_base != DEFAULT_DIST_BASE:
        cons.print(
            f"[{C_ERR}]error:[/] --no-cross-check is only valid with the "
            "default --dist-base. Pass either flag, not both."
        )
        cons.print(f"  [dim]default dist-base: {DEFAULT_DIST_BASE}[/]")
        return 2
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
                _ver_hint = installed_version() or "0.0.0"
                cons.print(
                    f"  [dim]pass --wheel <path>, e.g.[/] llm-speed verify --wheel ./dist/{wheel_name_for(_ver_hint)}"
                )
                return 1

    cons.print(f"  local wheel    : {wheel}")

    # When we fetched the wheel ourselves, the on-disk filename is a tempname
    # (e.g. /tmp/llm-speed-verify-xxxxx.whl) — but the published sidecar lives
    # at <dist>/<canonical-wheel-name>.sha256, NOT under the tempname.
    _ver = installed_version() or "0.0.0"
    sidecar_name = wheel_name_for(_ver) if fetched_temp is not None else None
    try:
        # Cross-check against the GitHub-Releases mirror by default. Tag is
        # derived from __version__ (post-2026-05-12 F-13 fix). A 404 on the
        # cross-check leg is non-fatal because PyPI now serves the canonical
        # per-file digest (F-30); the leg just adds defence-in-depth when it
        # exists. --no-cross-check disables the second domain entirely.
        if no_cross_check:
            cross_check = None
            cons.print(f"  [{C_WARN}]cross-check disabled[/] — PyPI-only trust")
        else:
            cross_check = gh_releases_sidecar_url(
                release_tag_for(_ver),
                sidecar_name or wheel.name,
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
    # F-30: third-leg PyPI status. Print one of:
    #   "PyPI digest OK : <url>"           (3-of-3 agreement)
    #   "PyPI not yet published"           (graceful pre-publish state)
    #   "PyPI digest skipped (no version)" (dev install / source tree)
    if verdict.pypi_status == "agreed":
        cons.print(f"  PyPI digest OK : {verdict.pypi_url}")
    elif verdict.pypi_status == "not_published":
        cons.print(
            f"  [{C_HINT}]PyPI digest    : not yet published at {verdict.pypi_url}[/]"
        )
    elif verdict.pypi_status == "skipped":
        cons.print(
            f"  [{C_HINT}]PyPI digest    : skipped (couldn't determine "
            "installed version)[/]"
        )
    cons.print()

    if verdict.matched:
        cons.print(
            f"[{C_OK}]MATCH[/]. The bytes you are about to run are byte-identical "
            f"to the wheel served at\n      {dist_base.rstrip('/')}/{wheel.name}"
        )
        cons.print(f"      Source code (audit it yourself): {REPO_URL}")
        # V-4: optional Sigstore verification. We only run this when the
        # match holds — otherwise the bundle is moot. The default is
        # opt-in (--sigstore) because it adds a ~3s round-trip + needs
        # the sigstore-python dep; once that's in the base install we'll
        # flip the default.
        if getattr(args, "sigstore", False):
            _ver_canonical = installed_version() or "0.0.0"
            sidecar_canonical = (
                wheel_name_for(_ver_canonical)
                if fetched_temp is not None
                else wheel.name
            )
            try:
                bundle = fetch_sigstore_bundle(dist_base, sidecar_canonical)
            except RuntimeError as exc:
                cons.print(f"  [{C_WARN}]sigstore fetch failed:[/] {exc}")
                bundle = None
            if bundle is None:
                cons.print(
                    f"  [{C_HINT}]sigstore: bundle not yet published at "
                    f"{dist_base.rstrip('/')}/{sidecar_canonical}.sigstore[/]"
                )
            else:
                ok, msg = verify_sigstore_bundle(wheel, bundle)
                if ok:
                    cons.print(f"  [{C_OK}]sigstore: {msg}[/]")
                else:
                    cons.print(f"  [{C_ERR}]sigstore: {msg}[/]")
                    return 1
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
