"""``llm-speed verify`` - prove the wheel matches the published artifact.

Compute the SHA-256 of a wheel (a local one, or the published one downloaded from the
public GitHub release, else PyPI) and compare it with the two independent published
digests:

* the GitHub release sidecar
  ``https://github.com/meadow-kun/llm-speed/releases/download/v<V>/<wheel>.sha256``
  (written by ``release-sign.yml``, copied to the public release in docs/RELEASE.md
  step 7), and
* PyPI's per-file digest from ``https://pypi.org/pypi/llm-speed/<V>/json``.

Two published digests that disagree are a hard failure (one distribution chain was
tampered with). Without ``--sigstore`` a MATCH needs at least one published digest equal
to the wheel's, and every source that is missing or unreachable is named.

``--sigstore`` is strict: both digests must be published, reachable and equal to the
wheel's, and the Sigstore bundle ``<wheel>.sigstore.json`` from the same GitHub release
must verify against the exact signer identity of ``release-sign.yml`` on tag ``v<V>``.
Anything missing, unreachable or failing prints NOT VERIFIED with the URLs tried and
exits 1.

llm-speed.com/dist/ is never consulted (it was the 0.0.1 pre-PyPI wheel host; nothing
publishes it any more).

What this does NOT prove: that the published wheel was built from the commit it claims
(reproducible builds live elsewhere), or that nothing changes after this point-in-time
check.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .ui.theme import (
    C_BRAND,
    C_ERR,
    C_HINT,
    C_OK,
    C_WARN,
    REPO_URL,
    get_console,
)

log = logging.getLogger("cli.verify")

# The public source/releases mirror. Every artifact verify reads from GitHub lives under
# <GH_RELEASES_BASE>/v<version>/<file>. (Module constant so tests can point it at a
# fake; there is deliberately no CLI flag to swap the trust root.)
GH_RELEASES_BASE = "https://github.com/meadow-kun/llm-speed/releases/download"
PYPI_PROJECT_NAME = "llm-speed"
SIGSTORE_BUNDLE_SUFFIX = ".sigstore.json"
MAX_BUNDLE_BYTES = 256 * 1024
MAX_WHEEL_BYTES = 50 * 1024 * 1024  # the real wheel is ~200 KB

# V-4: the signing certificate MUST carry this issuer and the identity of
# `release-sign.yml` dispatched on the exact release tag (docs/RELEASE.md, "Signing
# identity"). Any other identity, workflow or tag is a forgery, even if Rekor has it.
# (0.0.9 and older pinned `release-pypi.yml@refs/heads/main`, which never signs.)
SIGSTORE_EXPECTED_ISSUER = "https://token.actions.githubusercontent.com"
SIGSTORE_IDENTITY_PREFIX = "https://github.com/meadow-kun/llm-speed-web/.github/workflows/release-sign.yml@refs/tags/"


def wheel_name_for(version: str) -> str:
    """The publishable wheel filename for a given version string."""
    return f"llm_speed-{version}-py3-none-any.whl"


def release_tag_for(version: str) -> str:
    """The GitHub Releases tag for a version: `v<version>` (F-13: derived, never a
    separate hard-coded tag)."""
    return f"v{version}"


def version_from_wheel_name(name: str) -> str | None:
    """``llm_speed-0.0.10-py3-none-any.whl`` -> ``0.0.10``; None for other names."""
    prefix, suffix = "llm_speed-", "-py3-none-any.whl"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)] or None
    return None


def gh_release_url(version: str, filename: str) -> str:
    return f"{GH_RELEASES_BASE}/{release_tag_for(version)}/{filename}"


def gh_releases_sidecar_url(version_tag: str, wheel_name: str) -> str:
    """The GitHub-release `.sha256` URL for a tag (e.g. ``v0.0.9``) + wheel name."""
    return f"{GH_RELEASES_BASE}/{version_tag}/{wheel_name}.sha256"


def sigstore_bundle_url(version: str, wheel_name: str) -> str:
    return gh_release_url(version, f"{wheel_name}{SIGSTORE_BUNDLE_SUFFIX}")


def pypi_json_url(version: str) -> str:
    return f"https://pypi.org/pypi/{PYPI_PROJECT_NAME}/{version}/json"


def sigstore_identity_for(version: str) -> str:
    return f"{SIGSTORE_IDENTITY_PREFIX}{release_tag_for(version)}"


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


class ArtifactMissing(RuntimeError):
    """The URL answered 404: the artifact is not published there."""

    def __init__(self, url: str):
        super().__init__(f"GET {url} returned HTTP 404 (not published)")
        self.url = url


def _get(url: str, *, timeout: float) -> httpx.Response:
    """GET with redirects (GitHub Releases 302s to its object store). Network-level
    failures become RuntimeError naming the URL, never an unhandled httpx error."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            return client.get(url)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"could not reach {url} ({type(exc).__name__}: {exc})"
        ) from None


def _get_ok(url: str, *, timeout: float) -> httpx.Response:
    resp = _get(url, timeout=timeout)
    if resp.status_code == 404:
        raise ArtifactMissing(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    return resp


def fetch_checksum(url: str, *, timeout: float = 15.0) -> str:
    """Fetch a ``.sha256`` sidecar. `ArtifactMissing` on 404, RuntimeError otherwise."""
    return _get_ok(url, timeout=timeout).text


def fetch_sigstore_bundle(url: str, *, timeout: float = 20.0) -> bytes:
    """Fetch one Sigstore bundle. `ArtifactMissing` on 404; RuntimeError on any other
    HTTP / network failure or an implausibly large body."""
    resp = _get_ok(url, timeout=timeout)
    if len(resp.content) > MAX_BUNDLE_BYTES:
        raise RuntimeError(f"sigstore bundle at {url} unexpectedly large")
    return resp.content


def fetch_pypi_file(
    project: str, version: str, wheel_name: str, *, timeout: float = 15.0
) -> tuple[str, str | None] | None:
    """(sha256, download url) PyPI lists for `wheel_name` of `project==version`, or
    None when the version (404) or that file isn't on PyPI. RuntimeError when PyPI
    can't be reached or returns garbage."""
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    resp = _get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    try:
        data = _json.loads(resp.text)
    except _json.JSONDecodeError as exc:
        raise RuntimeError(f"PyPI returned non-JSON from {url}: {exc}") from exc
    for entry in data.get("urls") or []:
        if not isinstance(entry, dict) or entry.get("filename") != wheel_name:
            continue
        sha = (entry.get("digests") or {}).get("sha256")
        if isinstance(sha, str) and len(sha) == 64:
            dl = entry.get("url")
            return sha.lower(), dl if isinstance(dl, str) else None
    return None


def fetch_pypi_sha256(
    project: str, version: str, wheel_name: str, *, timeout: float = 15.0
) -> str | None:
    """The sha256 PyPI advertises for `wheel_name`, or None if not published."""
    found = fetch_pypi_file(project, version, wheel_name, timeout=timeout)
    return found[0] if found else None


def verify_sigstore_bundle(
    wheel_path: Path, bundle_bytes: bytes, version: str
) -> tuple[bool, str]:
    """Verify a Sigstore bundle against the wheel and the exact signer identity for
    `version`. Returns (ok, message). Without the `sigstore` package this returns
    (False, ...): `--sigstore` fails closed rather than pretending the check ran."""
    try:
        from sigstore.models import Bundle  # type: ignore
        from sigstore.verify import Verifier  # type: ignore
        from sigstore.verify import policy as _policy
    except ImportError:
        return False, (
            "sigstore-python is not installed, so the signature was not checked; "
            "install it with `pipx inject llm-speed sigstore` (or `pip install sigstore`)"
        )
    try:
        bundle = Bundle.from_json(bundle_bytes)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not parse sigstore bundle: {exc}"
    identity = sigstore_identity_for(version)
    try:
        verifier = Verifier.production()
        identity_policy = _policy.Identity(
            identity=identity, issuer=SIGSTORE_EXPECTED_ISSUER
        )
        with open(wheel_path, "rb") as f:
            blob = f.read()
        verifier.verify_artifact(blob, bundle, identity_policy)
        return True, f"sigstore verification OK (signed by {identity})"
    except Exception as exc:  # noqa: BLE001
        return False, f"sigstore verification failed for identity {identity}: {exc}"


def installed_wheel_path() -> Path | None:
    """Best effort: locate a local wheel for this install. Looks in the
    common dev (``./dist``) and pipx (``~/.local/pipx/.cache``,
    ``~/.local/share/pipx/.cache``) layouts. Returns None when no local
    wheel can be located — callers then download the published wheel."""
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


def published_wheel_urls(
    wheel_name: str, version: str, *, github: bool = True
) -> list[str]:
    """Where the published wheel is downloaded from when there is no local one: the
    public GitHub release, then PyPI (its file URL comes from the PyPI JSON, so only
    the JSON URL is known up front)."""
    urls = [gh_release_url(version, wheel_name)] if github else []
    return urls + [pypi_json_url(version)]


def _download(url: str, out: Path) -> None:
    written = 0
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            if r.status_code == 404:
                raise ArtifactMissing(url)
            if r.status_code != 200:
                raise RuntimeError(f"GET {url} returned HTTP {r.status_code}")
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    written += len(chunk)
                    if written > MAX_WHEEL_BYTES:
                        raise RuntimeError(
                            f"wheel download exceeded {MAX_WHEEL_BYTES} bytes "
                            f"from {url} — refusing to continue"
                        )
                    f.write(chunk)


def fetch_published_wheel_to_temp(
    wheel_name: str | None = None,
    version: str | None = None,
    *,
    github: bool = True,
) -> tuple[Path, str]:
    """Download the published wheel to a temp file so it can be hashed: the GitHub
    release first, then the file PyPI lists. Returns (path, url it came from). The
    caller checks it against both published digests; where it came from is not a
    trust decision. RuntimeError naming every URL tried when none serves it."""
    import os as _os
    import tempfile

    version = version or installed_version() or "0.0.0"
    wheel_name = wheel_name or wheel_name_for(version)
    candidates: list[str] = []
    if github:
        candidates.append(gh_release_url(version, wheel_name))
    tried: list[str] = []
    pypi_listed = False
    while True:
        if not candidates and not pypi_listed:
            pypi_listed = True
            jurl = pypi_json_url(version)
            try:
                found = fetch_pypi_file(PYPI_PROJECT_NAME, version, wheel_name)
            except RuntimeError as exc:
                tried.append(f"{jurl} ({exc})")
                found = None
            else:
                if not found or not found[1]:
                    tried.append(f"{jurl} ({wheel_name} not listed)")
            if found and found[1]:
                candidates.append(found[1])
        if not candidates:
            break
        url = candidates.pop(0)
        fd, tmp = tempfile.mkstemp(prefix="llm-speed-verify-", suffix=".whl")
        _os.close(fd)
        out = Path(tmp)
        try:
            _download(url, out)
        except ArtifactMissing:
            out.unlink(missing_ok=True)
            tried.append(f"{url} (HTTP 404)")
            continue
        except httpx.HTTPError as exc:
            out.unlink(missing_ok=True)
            tried.append(f"{url} ({type(exc).__name__})")
            continue
        except RuntimeError as exc:
            out.unlink(missing_ok=True)
            if "exceeded" in str(exc):
                raise
            tried.append(f"{url} ({exc})")
            continue
        except BaseException:
            out.unlink(missing_ok=True)
            raise
        return out, url
    raise RuntimeError(
        "could not download the published wheel; tried: " + "; ".join(tried)
    )


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


# Leg statuses: "agreed" (published and equal to the expected digest), "missing"
# (HTTP 404 / not listed), "unreachable" (network or other HTTP error).
@dataclass
class Leg:
    name: str
    url: str | None
    status: str = "pending"
    detail: str = ""


@dataclass
class Verdict:
    matched: bool
    local_path: Path
    local_hash: str
    expected_hash: str
    version: str
    legs: list[Leg] = field(default_factory=list)

    def agreed_urls(self) -> list[str]:
        urls = [leg.url for leg in self.legs if leg.status == "agreed" and leg.url]
        return list(dict.fromkeys(urls))

    def unavailable(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.status in ("missing", "unreachable")]


def _legs_text(legs: list[Leg]) -> str:
    return "".join(
        f"  {leg.name:<22}: {leg.url} ({leg.detail or leg.status})\n" for leg in legs
    )


def verify_wheel(
    wheel_path: Path,
    *,
    published_name: str | None = None,
    version: str | None = None,
    github: bool = True,
    pypi_check: bool = True,
    strict: bool = False,
) -> Verdict:
    """Hash the wheel and compare it with the GitHub release ``.sha256`` and PyPI's
    digest for release `version` (default: from the wheel name, else this install).

    Two published digests that disagree raise (tampering). No published digest at all
    raises (never "match against nothing"). ``strict`` (``--sigstore``): both legs must
    be published and reachable, else raise naming the URLs.

    ``published_name`` is the canonical wheel file name when ``wheel_path`` is a temp
    file with a generated name.
    """
    wheel_path = wheel_path.resolve()
    if not wheel_path.exists():
        raise RuntimeError(f"wheel not found: {wheel_path}")
    local = sha256_file(wheel_path)
    name = published_name or wheel_path.name
    ver = version or version_from_wheel_name(name) or installed_version()
    if not ver:
        raise RuntimeError(f"cannot tell which release {name} belongs to")

    found: list[tuple[Leg, str]] = []
    legs: list[Leg] = []

    if github:
        gh = Leg(
            "GitHub release sha256", gh_releases_sidecar_url(release_tag_for(ver), name)
        )
        legs.append(gh)
        try:
            raw = fetch_checksum(gh.url or "")
        except ArtifactMissing:
            gh.status, gh.detail = "missing", "HTTP 404"
        except RuntimeError as exc:
            gh.status, gh.detail = "unreachable", str(exc)
        else:
            digest = parse_checksum(raw)
            if digest is None:
                raise RuntimeError(f"{gh.url} does not contain a sha256 digest")
            found.append((gh, digest))

    if pypi_check:
        pl = Leg("PyPI sha256", pypi_json_url(ver))
        legs.append(pl)
        try:
            sha = fetch_pypi_sha256(PYPI_PROJECT_NAME, ver, name)
        except RuntimeError as exc:
            pl.status, pl.detail = "unreachable", str(exc)
        else:
            if sha is None:
                pl.status, pl.detail = "missing", f"{name} not listed for {ver}"
            else:
                found.append((pl, sha))

    digests = {d for _, d in found}
    if len(digests) > 1:
        raise RuntimeError(
            "published digests disagree: "
            + "; ".join(f"{leg.url} says {d}" for leg, d in found)
            + ". One of the distribution chains has been tampered with — do not "
            "trust this binary."
        )
    if not digests:
        raise RuntimeError(
            "no published sha256 digest found for this wheel:\n"
            + _legs_text(legs)
            + "  Refusing to declare 'match' against nothing."
        )
    expected = digests.pop()
    for leg, _ in found:
        leg.status = "agreed"

    unavailable = [leg for leg in legs if leg.status in ("missing", "unreachable")]
    if strict and (unavailable or not (github and pypi_check)):
        raise RuntimeError(
            "--sigstore requires both the GitHub release sha256 and PyPI's sha256; "
            "missing or unreachable:\n" + _legs_text(unavailable)
        )

    return Verdict(
        matched=(local == expected),
        local_path=wheel_path,
        local_hash=local,
        expected_hash=expected,
        version=ver,
        legs=legs,
    )


def check_sigstore(
    wheel: Path, wheel_name: str, version: str
) -> tuple[bool, list[str]]:
    """The `--sigstore` leg: the bundle from the public GitHub release, verified
    against `sigstore_identity_for(version)`. Fails closed; names the URL tried."""
    url = sigstore_bundle_url(version, wheel_name)
    try:
        bundle = fetch_sigstore_bundle(url)
    except ArtifactMissing:
        return False, [f"no sigstore bundle published; tried {url} (HTTP 404)"]
    except RuntimeError as exc:
        return False, [f"could not fetch the sigstore bundle; tried {url} ({exc})"]
    ok, msg = verify_sigstore_bundle(wheel, bundle, version)
    return ok, [msg, f"bundle: {url}"]


def _leg_line(leg: Leg) -> str:
    if leg.status == "agreed":
        return f"  {leg.name:<22}: OK  {leg.url}"
    if leg.status == "missing":
        return f"  [{C_WARN}]{leg.name:<22}: not published[/] {leg.url} ({leg.detail})"
    return f"  [{C_WARN}]{leg.name:<22}: unreachable[/] {leg.url} ({leg.detail})"


class _NoWrapConsole:
    """URLs in verify's output must stay copy-pasteable: rich never wraps a line."""

    def __init__(self, console):
        self._console = console

    def print(self, *a, **kw) -> None:
        kw.setdefault("soft_wrap", True)
        self._console.print(*a, **kw)


def cmd_verify(args) -> int:
    """``llm-speed verify [<path>]`` entry point."""
    cons = _NoWrapConsole(get_console())
    cons.print(
        f"[{C_BRAND}]llm-speed verify[/] - prove this binary matches the public artifact"
    )
    cons.print()

    target = getattr(args, "wheel", None)
    no_cross_check = bool(getattr(args, "no_cross_check", False))
    want_sigstore = bool(getattr(args, "sigstore", False))
    if no_cross_check and want_sigstore:
        cons.print(
            f"[{C_ERR}]error:[/] --sigstore needs the GitHub release; it can't be "
            "combined with --no-cross-check."
        )
        return 2
    fetched_temp: Path | None = None
    try:
        installed = installed_version() or "0.0.0"
        if target:
            wheel = Path(target).expanduser()
            published_name = wheel.name
        else:
            local = installed_wheel_path()
            published_name = wheel_name_for(installed)
            if local is not None:
                wheel = local
            else:
                where = "PyPI" if no_cross_check else "the GitHub release, then PyPI"
                cons.print(
                    f"  [dim]no local wheel found — downloading {published_name} from {where}[/]"
                )
                try:
                    wheel, src = fetch_published_wheel_to_temp(
                        published_name, installed, github=not no_cross_check
                    )
                    fetched_temp = wheel
                except Exception as exc:  # noqa: BLE001
                    cons.print(f"[{C_ERR}]error:[/] download failed: {exc}")
                    cons.print(
                        f"  [dim]pass a wheel path, e.g.[/] llm-speed verify ./dist/{published_name}"
                    )
                    if want_sigstore:
                        cons.print(f"[{C_ERR}]NOT VERIFIED[/].")
                    return 1
                cons.print(f"  downloaded from: {src}")

        cons.print(f"  local wheel    : {wheel}")
        if no_cross_check:
            cons.print(
                f"  [{C_WARN}]cross-check disabled[/] — PyPI only, no GitHub leg"
            )
        try:
            verdict = verify_wheel(
                wheel,
                published_name=published_name,
                github=not no_cross_check,
                strict=want_sigstore,
            )
        except RuntimeError as exc:
            cons.print(f"[{C_ERR}]error:[/] {exc}")
            cons.print(f"[{C_ERR}]NOT VERIFIED[/].")
            return 1

        cons.print(f"  local sha256   : {verdict.local_hash}")
        cons.print(f"  expected sha256: {verdict.expected_hash}")
        for leg in verdict.legs:
            cons.print(_leg_line(leg))
        cons.print()

        if not verdict.matched:
            cons.print(
                f"[{C_ERR}]MISMATCH[/]. The wheel does NOT match the published artifact.\n"
                "      Do not trust this binary; reinstall from PyPI or build from source."
            )
            return 1

        # L-2: with --sigstore the signature is checked BEFORE any verdict, and
        # anything short of a verified bundle is a failure.
        if want_sigstore:
            ok, lines = check_sigstore(wheel, published_name, verdict.version)
            colour = C_OK if ok else C_ERR
            cons.print(f"  [{colour}]sigstore: {lines[0]}[/]")
            for line in lines[1:]:
                cons.print(f"    {line}")
            if not ok:
                cons.print()
                cons.print(
                    f"[{C_ERR}]NOT VERIFIED[/]. The checksums matched, but --sigstore "
                    "was requested and the signature could not be verified."
                )
                return 1

        cons.print(
            f"[{C_OK}]MATCH[/]. The bytes you are about to run are byte-identical "
            f"to the published {published_name}; its digest is published at:"
        )
        for url in verdict.agreed_urls():
            cons.print(f"      {url}")
        if verdict.unavailable():
            cons.print(
                f"      [{C_HINT}]({len(verdict.unavailable())} source(s) not "
                "published or unreachable; listed above. `--sigstore` requires all.)[/]"
            )
        cons.print(f"      Source code (audit it yourself): {REPO_URL}")

        if installed_version():
            cons.print()
            cons.print(f"  installed version: [{C_HINT}]llm-speed {installed}[/]")
        show = installed_pip_show()
        if show:
            cons.print("  [dim]pip show llm-speed:[/]")
            for line in show.splitlines():
                cons.print(f"    [dim]{line}[/]")
        return 0
    finally:
        if fetched_temp is not None:
            try:
                fetched_temp.unlink(missing_ok=True)
            except OSError:
                pass
