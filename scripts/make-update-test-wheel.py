#!/usr/bin/env python3
"""Installer gate helper (auto-update scenario): copy a built llm-speed wheel to a
higher test version, so an install of the candidate can auto-update to it.

    make-update-test-wheel.py WHEEL OUT_DIR [SUFFIX]   (default SUFFIX: .post1)

Writes OUT_DIR/llm_speed-<ver><SUFFIX>-py3-none-any.whl with the dist-info directory,
METADATA Version, cli/__init__.py __version__ and RECORD rewritten, and prints its
path. Test use only: the result is never uploaded anywhere. Standard library only.
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
import zipfile
from pathlib import Path


def _digest(data: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    suffix = argv[2] if len(argv) == 3 else ".post1"
    m = re.match(r"^llm_speed-([0-9][0-9a-z.]*)-py3-none-any\.whl$", src.name)
    if not m or not re.match(r"^\.post[0-9]+$", suffix):
        print(
            "make-update-test-wheel: need llm_speed-<ver>-py3-none-any.whl and .postN",
            file=sys.stderr,
        )
        return 2
    old = m.group(1)
    new = old + suffix
    old_di, new_di = f"llm_speed-{old}.dist-info/", f"llm_speed-{new}.dist-info/"
    files: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            name, data = info.filename, z.read(info)
            if name.startswith(old_di):
                name = new_di + name[len(old_di) :]
            if name == new_di + "METADATA":
                data = re.sub(
                    rb"(?m)^Version: .*$", f"Version: {new}".encode(), data, count=1
                )
            elif name == "cli/__init__.py":
                data = re.sub(
                    rb'(?m)^__version__ = "[^"]*"',
                    f'__version__ = "{new}"'.encode(),
                    data,
                    count=1,
                )
            if name != new_di + "RECORD":
                files.append((name, data))
    record = (
        "".join(f"{n},{_digest(d)},{len(d)}\n" for n, d in files)
        + f"{new_di}RECORD,,\n"
    )
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"llm_speed-{new}-py3-none-any.whl"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files:
            z.writestr(n, d)
        z.writestr(new_di + "RECORD", record)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
