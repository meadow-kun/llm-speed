"""Tests for the privacy invariant (`assert_no_long_strings`).

Specifically guards against the over-permissive suffix-match regression:
allowlist entries like `error`, `workload`, `backend` must apply only to
their fully-qualified paths (top-level or `results.*`-nested), NOT to any
field anywhere in the tree that happens to share the leaf name.
"""

from __future__ import annotations

import pytest

from cli.signing import assert_no_long_strings

LONG = "X" * 500


# --------------------------------------------------------------------------
# Allowlisted paths must pass.
# --------------------------------------------------------------------------


def test_top_level_cli_version_allowed():
    assert_no_long_strings({"cli_version": LONG})


def test_top_level_suite_version_allowed():
    assert_no_long_strings({"suite_version": LONG})


def test_fingerprint_accelerator_summary_allowed():
    assert_no_long_strings({"fingerprint": {"accelerator_summary": LONG}})


def test_fingerprint_hash_allowed():
    assert_no_long_strings({"fingerprint": {"fingerprint_hash": LONG}})


def test_results_error_allowed():
    assert_no_long_strings({"results": [{"error": LONG}]})


def test_results_workload_allowed():
    assert_no_long_strings({"results": [{"workload": LONG}]})


def test_results_backend_allowed():
    assert_no_long_strings({"results": [{"backend": LONG, "backend_version": LONG}]})


def test_results_model_fields_allowed():
    assert_no_long_strings(
        {
            "results": [
                {
                    "model": {
                        "identifier": LONG,
                        "name": LONG,
                        "size": LONG,
                        "quant": LONG,
                        "digest": LONG,
                    },
                }
            ],
        }
    )


# --------------------------------------------------------------------------
# Suffix-match regression cases: same leaf name in a non-allowlisted path
# must be REJECTED (the bug the F-5 fix closes).
# --------------------------------------------------------------------------


def test_nested_extras_error_rejected():
    """`results[i].extras.error` must NOT inherit `results.error`'s exemption."""
    payload = {"results": [{"extras": {"error": LONG}}]}
    with pytest.raises(ValueError, match="results.extras.error"):
        assert_no_long_strings(payload)


def test_nested_extras_workload_rejected():
    payload = {"results": [{"extras": {"workload": LONG}}]}
    with pytest.raises(ValueError, match="results.extras.workload"):
        assert_no_long_strings(payload)


def test_nested_extras_backend_extras_error_rejected():
    """The exact pattern called out in the audit (driver injects `error` key)."""
    payload = {"results": [{"extras": {"backend_extras": {"error": LONG}}}]}
    with pytest.raises(ValueError, match="error"):
        assert_no_long_strings(payload)


def test_top_level_error_rejected_when_not_in_results():
    """Bare `error` at the root is no longer auto-exempt."""
    payload = {"error": LONG}
    with pytest.raises(ValueError):
        assert_no_long_strings(payload)


def test_model_extras_raw_description_rejected():
    """`results[i].model.extras.raw.description` is not allowlisted."""
    payload = {"results": [{"model": {"extras": {"raw": {"description": LONG}}}}]}
    with pytest.raises(ValueError):
        assert_no_long_strings(payload)


def test_fingerprint_extras_backends_error_rejected():
    """`fingerprint.extras.backends.error` is NOT allowlisted (bypass closed)."""
    payload = {"fingerprint": {"extras": {"backends": {"error": LONG}}}}
    with pytest.raises(ValueError):
        assert_no_long_strings(payload)


# --------------------------------------------------------------------------
# Sanity: short strings everywhere are fine.
# --------------------------------------------------------------------------


def test_short_strings_pass_anywhere():
    payload = {
        "results": [
            {
                "extras": {"backend_extras": {"error": "short", "anything": "fine"}},
                "model": {"extras": {"raw": {"description": "short text"}}},
            }
        ],
        "fingerprint": {"extras": {"backends": {"error": "ok"}}},
    }
    assert_no_long_strings(payload)
