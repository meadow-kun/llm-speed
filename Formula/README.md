# Homebrew Formula

This directory holds the Homebrew formula for `llm-speed`. It is kept in-tree so
contributors can see (and review) the formula alongside source changes, but
Homebrew itself does NOT install from here.

## Publishing flow

When ready to publish a release:

1. Push the version to PyPI first (gated job in `.github/workflows/release.yml`).
2. Compute the source tarball SHA-256:
   ```sh
   curl -sL https://files.pythonhosted.org/packages/source/l/llm-speed/llm-speed-${VERSION}.tar.gz \
     | shasum -a 256
   ```
3. Repeat for every `resource` block (each transitive dep). Tools like
   `homebrew-pypi-poet` automate this; see `docs/RELEASE.md`.
4. Replace every `PLACEHOLDER_SHA256` in `llm-speed.rb` with the real digest.
5. Copy `llm-speed.rb` into the tap repo:
   ```sh
   cp Formula/llm-speed.rb ../homebrew-tap/Formula/llm-speed.rb
   ```
6. In the tap repo, commit + push. The `publish-homebrew` job in
   `.github/workflows/release.yml` automates this **after** the maintainer approves the
   release environment.

End users install with:

```sh
brew install llm-speed/tap/llm-speed
```
