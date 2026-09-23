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
2. **Audit trail.** Layered audits, newest first:
   - 2026-05-07 — full-stack forensic audit covering the CLI, the
     ingestion API, the orchestrator, the marketing fleet, the trust
     chain, and identity hygiene:
     [`docs/security/forensic_full_stack_2026-05-07.md`](./docs/security/forensic_full_stack_2026-05-07.md).
     A short follow-up logging the F-N → fix mapping is at
     [`docs/security/launch_fixes_2026-05-07.md`](./docs/security/launch_fixes_2026-05-07.md).
   - 2026-05-04 — installable-CLI pentest + website-routes pentest:
     [`docs/security/pentest_installable_2026-05-04.md`](./docs/security/pentest_installable_2026-05-04.md),
     [`docs/security/website_routes_pentest_2026-05-04.md`](./docs/security/website_routes_pentest_2026-05-04.md).
   - 2026-05-01 — original CLI PII audit and OSS-readiness audit:
     [`docs/security/cli_pii_audit_2026-05-01.md`](./docs/security/cli_pii_audit_2026-05-01.md),
     [`docs/security/cli_oss_ready_2026-05-01.md`](./docs/security/cli_oss_ready_2026-05-01.md).
3. **Privacy contract.** Every field that leaves your machine is
   enumerated in [`docs/PRIVACY.md`](./docs/PRIVACY.md). Anything not on
   that list is a bug — file an issue.
4. **Signed runs.** Every uploaded benchmark is JWS-signed with an
   Ed25519 keypair on your machine. The public key rides in the JWS
   header so anyone can verify it without contacting us.

## Identity and contact

Maintainer: `meadow-kun` on GitHub. Single-maintainer open-source project,
Apache-2.0. Contact: privacy@llm-speed.com or
[GitHub issues](https://github.com/meadow-kun/llm-speed/issues).
