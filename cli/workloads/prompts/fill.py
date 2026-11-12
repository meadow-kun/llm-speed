"""Size a prompt to a token target.

The counter is the model's tokenizer or the server's tokenize endpoint.
It returns an int, or None when it cannot count. None keeps the old
4-characters-per-token text and reports no measurement; the run still
records whatever prompt-token count the backend reports afterwards.

The counter is called on the exact string that will be sent. A small
tolerance (2% of the target, at least 32 tokens) is enough for one
token boundary.
"""

from __future__ import annotations

from collections.abc import Callable

from .fixtures import CHARS_PER_TOKEN, build_prompt_of_chars

CountTokens = Callable[[str], int | None]


def tolerance(target: int) -> int:
    """How far from `target` still counts as filled."""
    return max(32, int(target) // 50)


def _text(n_chars: int, suffix: str) -> str:
    n_chars = max(int(n_chars), 0)
    if n_chars <= len(suffix):
        return suffix[:n_chars]
    body_len = n_chars - len(suffix)
    body = build_prompt_of_chars(body_len)
    if len(body) > body_len:
        body = body[:body_len]
    return body + suffix


def _usable(n: object) -> int | None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return None
    return n


def _better(
    current: tuple[str, int] | None, candidate: tuple[str, int], target: int
) -> tuple[str, int]:
    if current is None:
        return candidate

    def rank(n: int) -> tuple[int, int]:
        return (abs(n - target), 0 if n <= target else 1)

    return candidate if rank(candidate[1]) < rank(current[1]) else current


def fill_prompt(
    target_tokens: int,
    count_tokens: CountTokens,
    *,
    suffix: str = "",
) -> tuple[str, int | None]:
    """Return `(text, measured_tokens)`.

    `measured_tokens` is None only when `count_tokens` never returned a count.
    When it did, the text is the closest count to `target_tokens` found,
    within `tolerance` when the counter can express that length.
    """
    target = int(target_tokens)
    if target < 1:
        return suffix, None
    cap = max(len(suffix) + 1, target * 32)
    guess = min(cap, max(len(suffix), target * CHARS_PER_TOKEN))
    text = _text(guess, suffix)
    measured = _usable(count_tokens(text))
    if measured is None:
        return text, None

    best: tuple[str, int] | None = (text, measured)
    tol = tolerance(target)
    if abs(measured - target) <= tol:
        return text, measured

    lo = len(suffix)
    hi = guess
    if measured < target:
        lo = guess
        hi = guess
        chars = guess
        n = measured
        for _ in range(6):
            if chars >= cap or n >= target:
                break
            grown = min(cap, max(chars + 1, int(chars * target / max(n, 1) * 1.2)))
            if grown <= chars:
                break
            chars = grown
            trial = _text(chars, suffix)
            n_trial = _usable(count_tokens(trial))
            if n_trial is None:
                break
            best = _better(best, (trial, n_trial), target)
            n = n_trial
            hi = chars
            if abs(n - target) <= tol:
                return trial, n
        if best is not None and best[1] < target and hi >= cap:
            return best
    else:
        hi = guess
        lo = len(suffix)

    for _ in range(10):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        trial = _text(mid, suffix)
        n_trial = _usable(count_tokens(trial))
        if n_trial is None:
            break
        best = _better(best, (trial, n_trial), target)
        if abs(n_trial - target) <= tol:
            return trial, n_trial
        if n_trial < target:
            lo = mid + 1
        else:
            hi = mid
    assert best is not None
    return best
