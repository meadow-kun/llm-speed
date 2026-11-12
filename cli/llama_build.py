"""Which llama.cpp build a preset accepts.

`min_llama_build` is a floor. `max_llama_build` is a ceiling.
`exact_llama_build` is one build. A known mismatch is not relaxed: the run
does not switch to another build. Auto-install only ever has the pinned
prebuilt, so a window that build does not satisfy is a failure, not a
download of a different build.
"""

from __future__ import annotations

from typing import Any


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def bound_errors(
    *, exact: Any = None, minimum: Any = None, maximum: Any = None
) -> list[str]:
    """Schema cross-checks JSON Schema cannot express. Empty when the fields agree."""
    exact_n, min_n, max_n = _int(exact), _int(minimum), _int(maximum)
    errs: list[str] = []
    if exact_n is not None and min_n is not None and exact_n < min_n:
        errs.append("exact_llama_build is below min_llama_build")
    if exact_n is not None and max_n is not None and exact_n > max_n:
        errs.append("exact_llama_build is above max_llama_build")
    if min_n is not None and max_n is not None and min_n > max_n:
        errs.append("min_llama_build is above max_llama_build")
    return errs


def llama_build_error(
    build_number: Any,
    *,
    exact: Any = None,
    minimum: Any = None,
    maximum: Any = None,
) -> str | None:
    """None when this build may run. Otherwise one sentence, no fallback offered.

    No bound at all: any build, including one that does not report a number.
    A bound: a missing number cannot be confirmed. A number outside the window
    is a mismatch.
    """
    exact_n, min_n, max_n = _int(exact), _int(minimum), _int(maximum)
    if exact_n is None and min_n is None and max_n is None:
        return None
    need: list[str] = []
    if exact_n is not None:
        need.append(f"exactly b{exact_n}")
    else:
        if min_n is not None:
            need.append(f"b{min_n} or newer")
        if max_n is not None:
            need.append(f"b{max_n} or older")
    wanted = " and ".join(need)
    bn = _int(build_number)
    if bn is None or bn <= 1:
        return (
            "llama-server does not report a build number; "
            f"cannot confirm it is {wanted}"
        )
    if exact_n is not None and bn != exact_n:
        return (
            f"llama.cpp build b{bn} is not the required b{exact_n} "
            "(this run does not use a different build)"
        )
    if min_n is not None and bn < min_n:
        return f"llama.cpp build b{bn} is older than the required b{min_n} or newer"
    if max_n is not None and bn > max_n:
        return f"llama.cpp build b{bn} is newer than the maximum b{max_n}"
    return None
