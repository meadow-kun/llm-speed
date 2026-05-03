"""Tests for `cli.ui.auto`.

Auto mode is a thin wrapper that builds the argparse-style namespace
``cmd_bench`` consumes, so the focus is on the namespace shape.
"""

from __future__ import annotations

from cli.ui import auto


def test_build_auto_args_sane_defaults():
    ns = auto.build_auto_args()
    assert ns.command == "bench"
    assert ns.quick is True
    assert ns.dry_run is False
    assert ns.strict_anon is False
    assert ns.anon is False
    assert ns.resume is None
    # bench's `--no-upload` defaults to False, but consent will gate the upload.
    assert ns.no_upload is False


def test_build_auto_args_no_upload_override():
    ns = auto.build_auto_args(no_upload=True, api_base="https://example.test")
    assert ns.no_upload is True
    assert ns.api_base == "https://example.test"


def test_build_auto_args_api_key_passthrough():
    ns = auto.build_auto_args(api_key="sk-test")
    assert ns.api_key == "sk-test"
