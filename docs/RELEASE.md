# `llm-speed` Release Runbook

This document is the end-to-end recipe for shipping a new version of `llm-speed`
to every channel: PyPI, Docker Hub, Homebrew tap, GitHub Releases (standalone
binaries), and npm. Every publish step is gated on a manual approval — nothing
ships without an explicit nod from the maintainer.

## Prerequisites (one-time setup, do this once before the first release)

1. **GitHub Environments.** In repo Settings -> Environments, create:
   - `pypi-publish`     (required reviewer: `meadow-kun`)
   - `docker-publish`   (required reviewer: `meadow-kun`)
   - `homebrew-publish` (required reviewer: `meadow-kun`)
   - `npm-publish`      (required reviewer: `meadow-kun`)

   These environments make the matching `publish-*` jobs in
   `.github/workflows/release.yml` block until you click "Approve and deploy".

2. **PyPI Trusted Publisher.** On https://test.pypi.org (start there), add a
   trusted publisher for project `llm-speed`:
   - Owner: `meadow-kun`
   - Repository: `llm-speed`
   - Workflow: `release.yml`
   - Environment: `pypi-publish`

   When you're satisfied with TestPyPI, switch the `repository-url` in
   `release.yml` (`publish-pypi` job) to the real `https://upload.pypi.org/legacy/`
   and add a trusted publisher there too.

3. **Docker Hub credentials.** Add repo secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN` (a Docker Hub access token, not a password)

4. **npm token.** Add repo secret `NPM_TOKEN` (an automation token for the
   `llm-speed` npm package).

5. **Homebrew tap repo.** Create `meadow-kun/homebrew-tap` (empty repo);
   the publish-homebrew job opens PRs against it. Until the auto-PR script
   exists (Phase 1.5), do step 7 below by hand.

## Per-release steps

### 1. Bump version

Update version in two places:

- `pyproject.toml` -> `[project] version = "X.Y.Z"`
- `npm/package.json` -> `"version": "X.Y.Z"`

Optionally also update the URL in `Formula/llm-speed.rb` to the new tarball
filename (final SHA-256 values come later in step 7).

Commit, push to `main`.

### 2. Tag

```sh
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag does **not** trigger a release — `release.yml` is `workflow_dispatch`
only. The tag is for history, GitHub Releases attachment, and Homebrew URL.

### 3. Trigger the release workflow

```sh
gh workflow run release.yml -f run_publish=false
```

`run_publish=false` builds artifacts only. Use `run_publish=true` to additionally
queue the gated `publish-*` jobs (they will still wait for your approval).

### 4. Inspect the build artifacts

Wait for `build-wheel`, `build-shiv`, and `build-docker` to go green. Download
the artifacts:

```sh
gh run download <run-id> -n dist-pypi -D /tmp/dist
gh run download <run-id> -n shiv-macos-14-arm64 -D /tmp/shiv
gh run download <run-id> -n docker-cpu -D /tmp/docker
```

Smoke-test locally:

```sh
pip install /tmp/dist/llm-speed-*.whl
llm-speed --version
python /tmp/shiv/llm-speed-*.pyz --version
docker load -i /tmp/docker/llm-speed-cpu.tar
docker run --rm llmspeed/llm-speed:cpu-test --version
```

### 5. Approve `publish-pypi`

Once the wheel sniffs OK, go to the GitHub Actions run -> click the
`publish-pypi` job -> "Review deployments" -> Approve.

This pushes to **TestPyPI**. Verify:

```sh
pip install --index-url https://test.pypi.org/simple/ llm-speed==X.Y.Z
llm-speed --version
```

If clean, change `release.yml` `publish-pypi` `repository-url` to real PyPI,
push, and re-run the job. (Recommended: do at least 2 TestPyPI releases before
flipping to real PyPI.)

### 6. Approve `publish-docker`

After PyPI approval, the `publish-docker` job appears with a "Review deployments"
button. Approve. It pushes `llmspeed/llm-speed:cpu`, `:cuda`, `:rocm` to Docker
Hub. Verify:

```sh
docker run --rm llmspeed/llm-speed:cpu --version
docker run --rm --gpus all llmspeed/llm-speed:cuda --version  # on a CUDA host
```

### 7. Update + publish the Homebrew formula

Until the auto-PR script lands (Phase 1.5), do this manually:

1. Compute the source tarball sha256:
   ```sh
   curl -sL https://files.pythonhosted.org/packages/source/l/llm-speed/llm-speed-X.Y.Z.tar.gz \
     | shasum -a 256
   ```
2. Compute the sha256 for every `resource` block. Easiest: install
   `homebrew-pypi-poet`, run it against the wheel:
   ```sh
   pip install homebrew-pypi-poet
   poet -f llm-speed > /tmp/resources.rb
   ```
   Splice the resource blocks (with their real sha256 values) into
   `Formula/llm-speed.rb`, replacing every `PLACEHOLDER_SHA256`.
3. Replace the top-level `url` and `sha256` with the new release values.
4. Copy the file into the tap:
   ```sh
   cp Formula/llm-speed.rb ../homebrew-tap/Formula/llm-speed.rb
   cd ../homebrew-tap
   git add Formula/llm-speed.rb
   git commit -m "llm-speed X.Y.Z"
   git push
   ```
5. Test locally:
   ```sh
   brew tap meadow-kun/tap
   brew install --build-from-source llm-speed
   brew test llm-speed
   ```
6. Approve the `publish-homebrew` job in the workflow run (currently a stub —
   it just logs what it would do).

### 8. Approve `publish-npm`

Once binaries are attached to the GitHub Release (see step 9), approve the
`publish-npm` job. It (will, once un-stubbed) run `npm publish --access public`
from the `npm/` directory.

Smoke-test:

```sh
npm install -g llm-speed@X.Y.Z
llm-speed --version
```

### 9. Attach standalone binaries to the GitHub Release

The `build-shiv` job produced one `.pyz` per `{os, arch}`. Attach them:

```sh
gh release create vX.Y.Z --notes-from-tag
for f in /tmp/shiv-*/llm-speed-*.pyz; do
  zip -j "${f%.pyz}.zip" "$f"
  gh release upload vX.Y.Z "${f%.pyz}.zip"
done
```

(The `npm/install.js` script downloads `*.zip` filenames, hence the wrap.)

### 10. Smoke-test every channel end-to-end

On at least one machine each:

- `pipx install llm-speed && llm-speed --version`
- `uv tool install llm-speed && llm-speed --version`
- `brew install llm-speed/tap/llm-speed && llm-speed --version`
- `docker run --rm llmspeed/llm-speed:cpu --version`
- `npm install -g llm-speed && llm-speed --version`
- Download the `.pyz`/`.zip` from the release page and run it.

If any channel fails, see Rollback.

## Rollback

A bad release. Per channel:

| Channel       | Rollback command                                                                              |
|---------------|-----------------------------------------------------------------------------------------------|
| PyPI          | `pip install yank-cli && yank-cli yank llm-speed X.Y.Z` (or web UI: pypi.org -> manage). PyPI does NOT support deletion; yanking hides from `pip install llm-speed` but keeps it pinnable. |
| TestPyPI      | Same flow on test.pypi.org.                                                                  |
| Docker Hub    | `docker hub rmi llmspeed/llm-speed:cpu` etc. via web UI or API. Re-tag a known-good SHA as the new `:cpu`/`:cuda`/`:rocm`. |
| Homebrew tap  | Revert the commit in `meadow-kun/homebrew-tap`, push. Users get the previous formula on next `brew update`. |
| npm           | `npm deprecate llm-speed@X.Y.Z "broken; use X.Y.(Z-1)"`. (npm allows full unpublish only within 72h; deprecate is the safe default.) |
| GitHub Release| `gh release delete vX.Y.Z --yes` (only if you also want the binary downloads gone — npm wrapper depends on them). |

After rollback, bump to `X.Y.(Z+1)` and re-run the steps above. Don't reuse a
yanked version number — both PyPI and npm will refuse re-uploads.

## Why everything is gated

Every `publish-*` job in `release.yml` carries `if: ${{ inputs.run_publish == 'true' }}`
**and** `environment: <name>` with a required reviewer. That's two locks:

1. The job won't even queue unless you set `run_publish=true` when triggering
   the workflow.
2. Even if it queues, GitHub Environments pauses it on "Waiting for review"
   until you click Approve.

This is the GitHub-native way to require manual approval for sensitive jobs;
it does not depend on any third-party action and survives token rotations.

## Future work

- `scripts/update_brew_formula.sh` to auto-fill resource sha256 values and open
  the tap PR (replaces the manual step 7).
- PyOxidizer-based truly-portable binaries (replaces `.pyz` zipapps for
  no-Python-on-target users; current shiv approach assumes Python 3.10+ exists).
- Auto-trigger on tag push, *after* a few successful manual releases prove the
  pipeline.
