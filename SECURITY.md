# Security Policy

## Reporting a vulnerability

Email **privacy@llm-speed.com** with the details. We respond within 72 hours
and ship a fix or a mitigation plan within 14 days for high-severity issues.

For non-sensitive issues, GitHub issues at
[meadow-kun/llm-speed](https://github.com/meadow-kun/llm-speed/issues)
are also fine — but please use the email channel for anything that might
expose user data or compromise integrity until the fix is live.

If you'd like to report under a pseudonym or via Signal/PGP, mention that in
your initial email and we'll set up the channel.

## What we treat as in-scope

- The CLI at `https://llm-speed.com/dist/llm_speed-*.whl` (and the source
  tree under `cli/` in this repo)
- The API at `https://api.llm-speed.com/`
- The website at `https://llm-speed.com/`
- Distribution channels: PyPI (when published), GitHub Releases,
  Homebrew tap (when published), the `meadow-kun/llm-speed-action`
  repository

In-scope vulnerability classes (non-exhaustive):

- Anything that lets a remote actor exfiltrate uploaded run data
- Anything that lets a malicious upload poison the leaderboard
- Path traversal or command injection in the CLI
- Privacy contract violations beyond what is documented in
  [`docs/PRIVACY.md`](./docs/PRIVACY.md) (PII leak, identity bleed,
  unauthorized telemetry)
- Authentication / authorization bypass on the API
- Cryptographic issues with the JWS signing, the wheel sha256 chain, or
  the Ed25519 keypair persistence

## Out of scope

- Self-XSS that requires the user to paste attacker-controlled JavaScript
  into their own DevTools console
- Reports of "you don't have HSTS" / "you don't set X-Frame-Options" — we
  do, see `web/public/_headers`. Diff against that file before reporting.
- Issues in transitive dependencies that have an existing CVE and a known
  fix in a newer version — those are dependabot's job; we'll bump after
  a release.
- Brute-force / rate-limit-only issues that don't escalate to a different
  vulnerability class.

## Vulnerability disclosure timeline

Default: **90 days** from initial report to public disclosure. Earlier if
the fix shipped sooner. Later if you ask for more time and we agree.

If the issue affects user data already in the wild, we may publish a
shorter coordinated disclosure with you to alert affected users.

## What we will commit to

- Acknowledge your report within 72 hours
- Keep you informed at least weekly during fix development
- Credit you in the release notes if you'd like (or anonymously if you
  prefer)
- NOT pursue legal action against good-faith security research

## Out-of-band signals to watch

If you are evaluating whether to trust the project before the launch and want
to verify the chain yourself:

1. **Source matches binary.** Run `llm-speed verify` after install. It
   computes the sha256 of the installed wheel and compares against the
   published sidecar. If they disagree, do not run the binary.
2. **Audit trail.** The PII/privacy audit on 2026-05-01 is at
   [`docs/security/cli_pii_audit_2026-05-01.md`](./docs/security/cli_pii_audit_2026-05-01.md).
   The OSS-readiness audit is at
   [`docs/security/cli_oss_ready_2026-05-01.md`](./docs/security/cli_oss_ready_2026-05-01.md).
3. **Privacy contract.** Every field that leaves your machine is
   enumerated in [`docs/PRIVACY.md`](./docs/PRIVACY.md). Anything not on
   that list is a bug — file an issue.
4. **Signed runs.** Every uploaded benchmark is JWS-signed with an
   Ed25519 keypair on your machine. The public key rides in the JWS
   header so anyone can verify it without contacting us.
5. **Hash-pinned dependency install.** Maximum-paranoia install uses the
   committed lockfile so every transitive sha256 is verified at install
   time:
   ```sh
   curl -fsSLO https://raw.githubusercontent.com/meadow-kun/llm-speed/v1.0.2/requirements.lock
   pip install --require-hashes -r requirements.lock
   pip install https://llm-speed.com/dist/llm_speed-0.0.1-py3-none-any.whl
   llm-speed verify
   ```
   Any sha256 mismatch in `requirements.lock` aborts the install before
   any code runs. The lockfile is regenerated on every release with
   `pip-compile --generate-hashes`.

## Identity and contact

Maintainer: `meadow-kun` on GitHub. Single-maintainer open-source project,
Apache-2.0. Contact: privacy@llm-speed.com or
[GitHub issues](https://github.com/meadow-kun/llm-speed/issues).
