# Trust chain v2 (2026-05-01)

The first-pass trust chain (open source + sha256 sidecar on llm-speed.com)
is necessary but not sufficient. Two attack surfaces remain:

1. **Cloudflare Pages compromise.** An attacker who controls the wheel
   distribution can swap both the wheel AND the sidecar atomically.
2. **No identity-rooted artifact provenance.** The sha256 chain proves
   "this binary matches this digest" but not "the digest was published
   by the maintainer's actual build pipeline."

This doc tracks the v2 work that closes both.

## What's shipped

| Mechanism | Status | Where |
|---|---|---|
| `SECURITY.md` private disclosure channel | ✅ | `/SECURITY.md` |
| Dual-domain sidecar verification (CF + GH Releases) | ✅ | `cli/verify.py:gh_releases_sidecar_url`, `cli/verify.py:verify_wheel(cross_check_url=)` |
| GitHub Releases mirror for v1.0.1 wheel + sidecar | ✅ | https://github.com/meadow-kun/llm-speed/releases/tag/v1.0.1 |
| Hash-pinned transitive dep lockfile | ✅ | `/requirements.lock` (`pip-compile --generate-hashes`) |
| Sigstore-cosign keyless signing on tag push | ✅ workflow staged | `.github/workflows/release-sign.yml` |
| SLSA build-provenance attestation | ✅ workflow staged | same workflow, via `actions/attest-build-provenance@v2` |

The Sigstore + SLSA workflow is wired but won't fire effectively until the
repository is **public** (the OIDC subject claims need to resolve to a
public repo for Sigstore's public-good Fulcio CA to trust them).

## Staged for the maintainer to pull the trigger

These require account access, real keys, or a public repo. None of them
can be done autonomously by the orchestrator.

### 1. Make the repo public

The biggest single trust unlock. Sigstore signing, SLSA attestation, and the
dual-domain GH Release sidecar all require the repo to be public so a third
party can independently inspect the chain.

```sh
gh repo edit meadow-kun/llm-speed --visibility public --accept-visibility-change-consequences
```

After flipping public:
- The README + privacy + about pages already point at this repo, so the
  trust pitch becomes self-evidencing.
- Sigstore OIDC subject claims become verifiable.
- The CI matrix (windows + macos + ubuntu) becomes reasonable to host on
  the free GitHub Actions tier (private repo minutes burn faster).

### 2. Defensive PyPI registrations

Once the repo is public, register PyPI placeholder packages to prevent
typosquatting:

```sh
# 1. Create a PyPI account if you don't have one. Enable 2FA. Generate a
#    project-scoped API token.
# 2. Create empty placeholder packages with the canonical install pointer.
for name in llmspeed llm-speed-cli llm-bench llm_speed_cli llmbench; do
  mkdir -p /tmp/$name && cd /tmp/$name
  cat > pyproject.toml <<EOF
[project]
name = "$name"
version = "0.0.1"
description = "Placeholder. The canonical package is 'llm-speed' — install with: pipx install llm-speed"
authors = [{name = "meadow-kun", email = "2424351+meadow-kun@users.noreply.github.com"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
[project.urls]
Homepage = "https://llm-speed.com"
EOF
  python -m build --sdist
  twine upload dist/*
done
```

Each placeholder takes ~5 min. The README of each placeholder package
should redirect to `pip install llm-speed`.

### 3. Trusted publishing on PyPI (PEP 740 / OIDC)

Register `llm-speed` on PyPI with **trusted publishing** enabled — no
long-lived API tokens, the upload is gated on a specific GitHub Actions
workflow + repo + environment.

```
PyPI account → "Your projects" → llm-speed → "Publishing" → "Add a new pending publisher"
  - Owner: meadow-kun
  - Repository: llm-speed
  - Workflow: release-pypi.yml
  - Environment: pypi-publish
```

After this lands, write a `release-pypi.yml` workflow that publishes via
`pypa/gh-action-pypi-publish@release/v1`. PyPI shows a "verified
publisher" badge next to the package.

### 4. Signed git tags

Set up a GPG or SSH key for signing on whatever machine you tag releases
from:

```sh
# SSH-key signing (cleanest if you already have an SSH key)
git config --local gpg.format ssh
git config --local user.signingkey ~/.ssh/id_ed25519
git config --local commit.gpgsign true
git config --local tag.gpgsign true

# Then re-sign the recent tags
git tag -d v1.0.1 && git tag -s v1.0.1 -m "v1.0.1 — signed"
git push --force-with-lease origin v1.0.1
```

On GitHub the tags will show "Verified" next to the SHA. Combined with the
Sigstore signing workflow, this proves "the tag came from the maintainer's
key AND the artifact came from the public CI."

### 5. macOS code-signing for any future native binary

Out of scope today (we ship a pure-Python wheel). Becomes relevant if we
ever produce a `.pyz` / `.dmg` / standalone binary.

## What this changes for a sceptical reader

Before v2:
> "I trust llm-speed.com to serve a clean wheel."

After v2 (once Sigstore is firing on public repo):
> "Sigstore proves the wheel was built from commit `<hash>` by GitHub
> Actions in workflow `release-sign.yml`. SLSA L3 attestation confirms
> the build env. The sidecar at llm-speed.com matches the one in the
> GitHub release. PyPI's trusted-publishing badge confirms the upload
> came from the same workflow. Two independent infrastructure providers
> agreeing closes the single-point-of-failure attack."

That's the chain a serious enterprise / r/LocalLLaMA security commenter
expects in 2026.

## Order of operations recommended

1. (Maintainer) Make the repo public — biggest single unlock.
2. (Auto) Sigstore + SLSA workflow fires on next tag push.
3. (Maintainer) PyPI account + trusted-publishing config + first publish.
4. (Maintainer) SSH-key signing on git tags.
5. (Maintainer) Defensive placeholder packages on PyPI.
6. (Auto) Update verify command to also check Sigstore signature when
   present.
