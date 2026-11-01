#!/usr/bin/env python3
"""Resolve and wait for the versions the installer gate checks (read-only).

    python3 scripts/installer-gate-versions.py [--new V] [--prev V]
        [--wait-minutes N] [--require-latest] [--github-output FILE]
        [--summary FILE]

NEW defaults to the latest llm-speed release on PyPI; a leading "v" is
stripped, so a release tag can be passed as is. PREV defaults to the highest
non-yanked final PyPI release below NEW. With --wait-minutes, polls once a
minute until PyPI serves NEW (and, with --require-latest, lists it as the
latest) and the meadow-kun/homebrew-tap formula is at NEW. PyPI not ready in
time is an error (exit 1); a lagging tap only warns, and the brew rows of
scripts/check-installers.sh then FAIL with the tap's version.

Versions are N(.N)* with an optional aN/bN/rcN suffix, checked with
re.fullmatch on input and again on everything resolved from PyPI or the tap
before it is written (fail closed). Prints new=, prev=, pypi_latest= and tap= lines (also appended to
--github-output). Standard library only. Nothing is installed or uploaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

PKG = "llm-speed"
TAP_FORMULA = "https://raw.githubusercontent.com/meadow-kun/homebrew-tap/main/Formula/llm-speed.rb"
UA = {"User-Agent": "llm-speed-installer-gate/1"}
# Strict, whole-string version checks (Security R-114): always re.fullmatch,
# ASCII only, so no newline, CR, space or other character can get through.
FINAL = re.compile(r"[0-9]+(?:\.[0-9]+)*", re.ASCII)
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?", re.ASCII)


def valid(v: object) -> bool:
    return isinstance(v, str) and VERSION.fullmatch(v) is not None


def shown(v: object) -> str:
    """A value from PyPI or the tap, safe to print (no workflow commands)."""
    return v if valid(v) else ("?" if not v else "<invalid>")


def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (fixed https URLs)
            return r.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def pypi() -> dict | None:
    raw = fetch(f"https://pypi.org/pypi/{PKG}/json")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def previous(doc: dict, new: str) -> str:
    """Highest non-yanked final release strictly below NEW."""
    finals = []
    for v, files in doc.get("releases", {}).items():
        if isinstance(v, str) and FINAL.fullmatch(v) and files and not all(f.get("yanked") for f in files):
            finals.append(v)
    below = [v for v in finals if FINAL.fullmatch(new) and key(v) < key(new)]
    return max(below, key=key) if below else ""


def tap_version() -> str:
    raw = fetch(TAP_FORMULA)
    m = re.search(rb"llm[-_]speed-([0-9][0-9a-z.]*)\.tar\.gz", raw or b"")
    v = m.group(1).decode("ascii", "replace") if m else ""
    return v if valid(v) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="")
    ap.add_argument("--prev", default="")
    ap.add_argument("--wait-minutes", type=int, default=0)
    ap.add_argument("--require-latest", action="store_true")
    ap.add_argument("--github-output", default="")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()

    raw_new, raw_prev = a.new, a.prev  # no strip: whitespace is rejected, not fixed
    new = raw_new[1:] if raw_new.startswith("v") else raw_new
    prev = raw_prev[1:] if raw_prev.startswith("v") else raw_prev
    for label, raw, v in (("version", raw_new, new), ("prev", raw_prev, prev)):
        if raw and not valid(v):
            print(f"::error::bad {label} {raw!r} (want e.g. 0.0.12 or 0.0.13rc1)", file=sys.stderr)
            return 2
    if not 0 <= a.wait_minutes <= 240:
        print("::error::--wait-minutes must be 0-240", file=sys.stderr)
        return 2

    deadline = time.monotonic() + a.wait_minutes * 60
    doc, latest, tap, pypi_ok = None, "", "", False
    while True:
        doc = pypi()
        latest = (doc or {}).get("info", {}).get("version", "")
        if not valid(latest):
            latest = ""  # unreadable or malformed: treated as not ready
        target = new or latest
        has = bool(doc and doc.get("releases", {}).get(target))
        pypi_ok = has and (not a.require_latest or latest == target)
        tap = tap_version()
        if pypi_ok and tap == target:
            break
        if time.monotonic() >= deadline:
            break
        print(f"waiting: PyPI latest={shown(latest)} has {shown(target)}={has}, tap={shown(tap)}", flush=True)
        time.sleep(60)

    new = new or latest
    if not doc or not new:
        print("::error::could not read llm-speed from PyPI", file=sys.stderr)
        return 1
    if not pypi_ok:
        why = "is not on PyPI" if not doc.get("releases", {}).get(new) else f"is on PyPI but the latest is {latest}"
        print(f"::error::llm-speed {new} {why} (waited {a.wait_minutes} min)", file=sys.stderr)
        return 1
    prev = prev or previous(doc, new)
    if not prev or prev == new:
        print(f"::error::no previous release below {new} on PyPI; pass prev", file=sys.stderr)
        return 1
    if new == latest and tap != new:
        print(
            f"::warning::meadow-kun/homebrew-tap formula is at {tap or 'unreadable'}, not {new} "
            f"(the tap bump lags the release). The brew rows will FAIL; re-run with workflow_dispatch after the bump."
        )
    # Fail closed: re-check everything PyPI or the tap resolved, right before
    # it reaches GITHUB_OUTPUT (Security R-114). tap may be empty (unreadable).
    for name, v, may_be_empty in (("version", new, False), ("prev", prev, False),
                                  ("pypi_latest", latest, False), ("tap", tap, True)):
        if not (valid(v) or (may_be_empty and v == "")):
            print(f"::error::resolved {name} {v!r} is not a valid version; refusing to write outputs", file=sys.stderr)
            return 1
    lines = [f"new={new}", f"prev={prev}", f"pypi_latest={latest}", f"tap={tap}"]
    if any(not ln.isascii() or not ln.isprintable() for ln in lines):
        print("::error::refusing to write a non-printable output line", file=sys.stderr)
        return 1
    print("\n".join(lines))
    if a.github_output:
        with open(a.github_output, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(
                f"### llm-speed installer gate: {prev} -> {new}\n\n"
                f"PyPI latest: `{latest}` - Homebrew tap formula: `{tap or 'unreadable'}`"
                f"{'' if tap == new or new != latest else ' (lagging: brew rows will FAIL)'}\n\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
