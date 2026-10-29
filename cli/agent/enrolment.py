"""Single-use enrolment token for `llm-speed agent enroll|install` (security retest R-8).

A queue admin can issue a one-time `lse_` token that lets a pairing skip the public
pairing limits (it still needs the admin's approval of the code). The CLI reads it from

- `--enrolment-token-file PATH`: a regular file owned by you that only you can read
  (mode 0600 or 0400; anything group/other-readable is refused), or
- `$LLM_SPEED_ENROLMENT_TOKEN`, removed from this process's environment as soon as it is
  read, so no service or benchmark child process ever inherits it.

Never from the command line (argv is visible to every user via `ps`), never logged or
printed, sent once in `POST /v1/agent/pair`, forgotten after the 201. A 401
`invalid_enrolment` stops enrolling; it is not retried without the token.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

ENV = "LLM_SPEED_ENROLMENT_TOKEN"
TOKEN_RE = re.compile(r"^lse_[A-Za-z0-9_-]{43}$")
_MAX = 256
_WINDOWS = os.name == "nt"
WINDOWS_NOTE = (
    "note: on Windows the enrolment token file's owner and permissions are not checked; "
    "keep it somewhere only you can read and delete it after enrolling."
)
DELETE_FILE_NOTE = (
    "The enrolment token is used up and can't be used again; you can delete the token "
    "file now."
)
UNSET_ENV_NOTE = (
    "The enrolment token is used up; you can remove LLM_SPEED_ENROLMENT_TOKEN from your "
    "shell now."
)


class EnrolmentTokenError(ValueError):
    """User-facing; never contains the token."""


def _from_file(path: str) -> str:
    p = Path(os.path.expanduser(path))
    try:
        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as e:
        raise EnrolmentTokenError(
            f"can't read the enrolment token file ({e.strerror or 'error'}; "
            "symlinks are not followed)"
        ) from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise EnrolmentTokenError("the enrolment token file is not a regular file")
        if _WINDOWS:
            # No POSIX owner/mode bits to check on Windows; say so instead of guessing.
            print(WINDOWS_NOTE, file=sys.stderr)
        elif hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise EnrolmentTokenError("the enrolment token file is not owned by you")
        elif st.st_mode & 0o077:
            raise EnrolmentTokenError(
                "the enrolment token file can be read by other users (mode "
                f"{stat.S_IMODE(st.st_mode):04o}); run `chmod 600` on it first"
            )
        data = os.read(fd, _MAX + 1)
    finally:
        os.close(fd)
    if len(data) > _MAX:
        raise EnrolmentTokenError("the enrolment token file is too large to be a token")
    return data.decode("ascii", "replace").strip()


def take(path: str | None, environ: dict | None = None) -> str | None:
    """The enrolment token, or None when none was given. The env var is always removed."""
    env = os.environ if environ is None else environ
    from_env = env.pop(ENV, None)
    if path:
        tok = _from_file(path)
        where = "the enrolment token file"
    elif from_env is not None and from_env.strip():
        tok = from_env.strip()
        where = f"${ENV}"
    else:
        return None
    if not TOKEN_RE.fullmatch(tok):
        raise EnrolmentTokenError(
            f"{where} does not hold an enrolment token (expected lse_ followed by 43 "
            "letters, digits, _ or -)"
        )
    return tok
