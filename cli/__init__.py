"""llm-speed CLI package.

Public surface (importable, agent-implemented): types in `cli.types`,
registries in `cli.registry`, fingerprint in `cli.fingerprint`,
backend drivers under `cli.drivers`, workloads under `cli.workloads`,
top-level commands under `cli.commands`.
"""

# Single source of truth for the CLI banner version. Must match the
# `version` in pyproject.toml (which is what PyPI / the wheel filename
# / `llm-speed verify --pypi-cross-check` all use). Keeping them in
# sync is a release-checklist item; if you change one, change both.
__version__ = "0.0.7"
SUITE_VERSION = "suite-v1"
