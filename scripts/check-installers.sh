#!/usr/bin/env bash
# Installer release gate: every supported installer must install and upgrade the
# just-published llm-speed release cleanly. Run after release-pypi (and the
# Homebrew tap bump, which lands in the same step) on Linux and macOS; the
# PowerShell twin scripts/check-installers.ps1 covers Windows. CI runs both on
# all three OSes via .github/workflows/installer-gate.yml on the public mirror
# meadow-kun/llm-speed (shipped there by the CLI export). Standalone: it needs
# only bash, curl, python3 and the installers, never a source checkout.
#
#   scripts/check-installers.sh [options] NEW_VERSION PREV_VERSION
#
# Rows (installer x scenario), each in its own throwaway HOME/PIPX_HOME/
# UV_TOOL_DIR/caches under the run root, deleted as soon as the row finishes:
#   fresh    documented install of NEW         pip install / pipx install /
#                                              uv tool install / brew install
#   upgrade  install PREV, then the documented upgrade command
#                                              pip install -U / pipx upgrade /
#                                              uv tool upgrade / brew upgrade
#   autoupdate  the CLI updates itself (0.0.14+, docs/CLI.md "Auto-update"):
#     - with --auto-update-wheel W (all four installers): install the
#       candidate W (llm_speed-NEW-py3-none-any.whl, deps from PyPI) unpinned,
#       copy it to NEW.post1 (make-update-test-wheel.py), run `llm-speed about`
#       on a pseudo-terminal with the test-only hook LLM_SPEED_UPDATE_TEST=1 +
#       LLM_SPEED_UPDATE_TEST_WHEEL=file://<NEW.post1 wheel>, wait for the
#       detached background update, then check `--version` is NEW.post1 and the
#       next run prints `Updated llm-speed NEW -> NEW.post1` once. Homebrew:
#       a local tap (llm-speed-gate/local, `brew tap-new --no-git`, never meadow-kun/tap)
#       gets a formula that installs W (deps from a wheelhouse fetched first),
#       then the formula is switched to NEW.post1 and the background updater
#       must `brew upgrade` to it (test hook names the version; the bits come
#       from the local tap). The tap is removed afterwards.
#       Build W in the same run (CI: `python -m pip wheel --no-deps .` on the
#       release tag); NEW.post1 is made from it here.
#     - otherwise (release, all four incl. brew): install PREV as in the
#       upgrade row, run `llm-speed update` (PyPI / the tap, no hook) instead of
#       the upgrade command. N/A when PREV has no updater (< 0.0.14) or the
#       rows are pinned. N/A is not a failure and not a pass.
# After each: `llm-speed --version` must print NEW (NEW.post1 for the local
# autoupdate row; if that fails but the installed metadata says it, the row is
# FAIL-soft), and `llm-speed doctor --help` must run. Nothing benchmarks,
# submits or uploads anything.
#
# If NEW is not the latest on PyPI (a historical re-check), the unpinned
# documented commands would fetch a later version, so rows use pinned
# equivalents and say "(pinned)"; brew can only test the tap's current formula.
# The brew rows first check the published tap formula is at NEW (read-only),
# and FAIL if it is not. brew installs into the real brew prefix, so it is
# skipped when llm-speed is already installed there (it will not touch that).
#
# Options:
#   --only "pip pipx uv brew"  installers to run (default: all four)
#   --scenarios "fresh upgrade autoupdate"  scenarios (default: all three)
#   --auto-update-wheel W      the locally built NEW wheel for the autoupdate
#                              row (pip/pipx/uv); see above
#   --bootstrap-tools          pip-install pipx and uv into a venv under the
#                              run root when they are missing (deleted at exit)
#   --dry-run                  print every command, run none (no network)
#   --keep                     keep the run directories (debugging)
#   --summary FILE             also append the result table as Markdown to FILE
#                              (e.g. "$GITHUB_STEP_SUMMARY")
#   -h, --help
# Env: INSTALLER_GATE_ROOT (run root; default ~/.cache/llm-speed-installer-gate;
#      never /tmp),
#      PYTHON (default python3), INSTALLER_GATE_TAP (default meadow-kun/tap).
#
# Exit: 0 every row PASS; 1 any FAIL or FAIL-soft; 3 no FAIL but a SKIP
# (a SKIP is not-done for the release); 2 usage error.
set -uo pipefail

usage() { sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; }

ONLY="pip pipx uv brew"
SCENARIOS="fresh upgrade autoupdate"
AU_WHEEL=""
BOOTSTRAP=0
DRY=0
KEEP=0
SUMMARY=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:?--only needs a value}"; shift ;;
    --scenarios) SCENARIOS="${2:?--scenarios needs a value}"; shift ;;
    --auto-update-wheel) AU_WHEEL="${2:?--auto-update-wheel needs a wheel}"; shift ;;
    --bootstrap-tools) BOOTSTRAP=1 ;;
    --dry-run) DRY=1 ;;
    --keep) KEEP=1 ;;
    --summary) SUMMARY="${2:?--summary needs a file}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "check-installers: unknown option $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
[ "${#ARGS[@]}" -eq 2 ] || { usage >&2; exit 2; }
NEW="${ARGS[0]}"
PREV="${ARGS[1]}"
VRE='^[0-9]+(\.[0-9]+)*([a-z]+[0-9]*)?$'
for v in "$NEW" "$PREV"; do
  [[ "$v" =~ $VRE ]] || { echo "check-installers: bad version '$v'" >&2; exit 2; }
done
[ "$NEW" != "$PREV" ] || { echo "check-installers: NEW and PREV are the same" >&2; exit 2; }
for i in $ONLY; do
  case "$i" in pip|pipx|uv|brew) ;; *) echo "check-installers: unknown installer '$i'" >&2; exit 2 ;; esac
done
for i in $SCENARIOS; do
  case "$i" in fresh|upgrade|autoupdate) ;; *) echo "check-installers: unknown scenario '$i'" >&2; exit 2 ;; esac
done
if [ -n "$AU_WHEEL" ]; then
  case "$(basename "$AU_WHEEL")" in
    "llm_speed-$NEW-py3-none-any.whl") ;;
    *) echo "check-installers: --auto-update-wheel must be llm_speed-$NEW-py3-none-any.whl" >&2; exit 2 ;;
  esac
  [ "$DRY" = 1 ] || [ -f "$AU_WHEEL" ] || { echo "check-installers: no such wheel $AU_WHEEL" >&2; exit 2; }
  if [ -d "$(dirname "$AU_WHEEL")" ]; then AU_WHEEL="$(cd "$(dirname "$AU_WHEEL")" && pwd)/$(basename "$AU_WHEEL")"; fi
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
BUMP="$NEW.post1"

PKG=llm-speed
# Rows use their sandbox's own config; a caller's settings must not leak in.
unset LLM_SPEED_CONFIG LLM_SPEED_CACHE LLM_SPEED_DISABLE_AUTOUPDATE LLM_SPEED_UPDATE_TEST LLM_SPEED_UPDATE_TEST_WHEEL
PYTHON="${PYTHON:-python3}"
TAP="${INSTALLER_GATE_TAP:-meadow-kun/tap}"
TAP_REPO="meadow-kun/homebrew-tap"
if [ -n "${INSTALLER_GATE_ROOT:-}" ]; then
  ROOT="$INSTALLER_GATE_ROOT"
else
  ROOT="$HOME/.cache/llm-speed-installer-gate"
fi
case "$ROOT" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
    echo "check-installers: refusing a run root under /tmp ($ROOT)" >&2; exit 2 ;;
esac
RUN="$ROOT/run-$NEW-$(date +%Y%m%d%H%M%S)-$$"

FROM=""
RESULTS=()   # "installer|scenario|status|version|detail"
record() { RESULTS+=("$1|$2|$3|$4|$5"); echo "  -> $1 $2: $3 ${4:+($4)} $5"; }

# Runs a command, or prints it in --dry-run. Output goes to the row log.
LOG=/dev/null
x() {
  if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; return 0; fi
  echo "+ $*" >>"$LOG"
  "$@" >>"$LOG" 2>&1
}

cleanup() {
  if [ "$KEEP" = 0 ] && [ "$DRY" = 0 ] && [ -d "$RUN" ]; then rm -rf "$RUN"; fi
}
trap cleanup EXIT

# Latest version on PyPI (read-only). Decides documented vs pinned commands.
LATEST=""
if [ "$DRY" = 0 ]; then
  LATEST="$(curl -fsSL --proto '=https' --max-time 30 "https://pypi.org/pypi/$PKG/json" 2>/dev/null |
    "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null)" || LATEST=""
else
  LATEST="$NEW"
fi
# Earliest upload time of NEW's files (for the uv upgrade row).
NEW_UPLOADED=""
if [ "$DRY" = 0 ]; then
  NEW_UPLOADED="$(curl -fsSL --proto '=https' --max-time 30 "https://pypi.org/pypi/$PKG/$NEW/json" 2>/dev/null |
    "$PYTHON" -c 'import json,sys; print(min(u["upload_time_iso_8601"] for u in json.load(sys.stdin)["urls"]))' 2>/dev/null)" || NEW_UPLOADED=""
else
  NEW_UPLOADED="<NEW upload time>"
fi
PINNED=0
if [ "$LATEST" != "$NEW" ]; then PINNED=1; fi
SPEC_NEW="$PKG"; [ "$PINNED" = 1 ] && SPEC_NEW="$PKG==$NEW"
SFX=""; [ "$PINNED" = 1 ] && SFX=" (pinned)"

echo "== llm-speed installer gate: NEW=$NEW PREV=$PREV (PyPI latest: ${LATEST:-unknown}) on $(uname -s) $(uname -m)"
echo "   run root: $RUN${SFX:+ ; NEW is not the PyPI latest, rows are pinned}"
[ "$DRY" = 1 ] || mkdir -p "$RUN"

# Optional tool bootstrap: pipx and uv from PyPI into a venv under the run root.
if [ "$BOOTSTRAP" = 1 ]; then
  need=""
  case " $ONLY " in *" pipx "*) command -v pipx >/dev/null || need="$need pipx" ;; esac
  case " $ONLY " in *" uv "*) command -v uv >/dev/null || need="$need uv" ;; esac
  if [ -n "$need" ]; then
    echo "== bootstrapping:$need"
    LOG="$RUN/bootstrap.log"
    # shellcheck disable=SC2086 # $need is a word list
    if x "$PYTHON" -m venv "$RUN/tools" && PIP_CACHE_DIR="$RUN/tools-cache" x "$RUN/tools/bin/python" -m pip install -q $need; then
      PATH="$RUN/tools/bin:$PATH"
    else
      echo "   bootstrap failed (see $LOG)"
    fi
  fi
fi

# Fresh, isolated environment for one row. Sets SB and exports the isolating vars.
new_sandbox() {
  SB="$RUN/$1-$2"
  LOG="$SB.log"
  [ "$DRY" = 1 ] && return 0
  mkdir -p "$SB/home" "$SB/bin" "$SB/cache" "$SB/config" "$SB/data" "$SB/state"
  : >"$LOG"
}
# Runs a command inside the row's sandbox.
sx() {
  HOME="$SB/home" XDG_CACHE_HOME="$SB/cache" XDG_CONFIG_HOME="$SB/config" \
    XDG_DATA_HOME="$SB/data" XDG_STATE_HOME="$SB/state" \
    PIPX_HOME="$SB/pipx" PIPX_BIN_DIR="$SB/bin" PIPX_MAN_DIR="$SB/man" \
    UV_TOOL_DIR="$SB/uvtools" UV_TOOL_BIN_DIR="$SB/bin" UV_CACHE_DIR="$SB/cache/uv" \
    UV_PYTHON_INSTALL_DIR="$SB/uvpython" UV_PYTHON_DOWNLOADS=never \
    PIP_CACHE_DIR="$SB/cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
    PATH="$SB/bin:$SB/venv/bin:$PATH" x "$@"
}
drop_sandbox() {
  [ "$DRY" = 1 ] && return 0
  if [ "$KEEP" = 0 ]; then
    chmod -R u+w "$SB" 2>/dev/null
    rm -rf "$SB"
  fi
}

# The installed version per the tool venv's metadata. Isolated (-I: no cwd or
# script dir on sys.path, no PYTHON* env, no user site) and run from the row's
# sandbox, so a source checkout's llm_speed.egg-info can't shadow the install.
meta_version() {
  [ -x "$1" ] || return 0
  (cd "$SB" 2>/dev/null || cd /; "$1" -I -c "import importlib.metadata as m; print(m.version('$PKG'))" 2>/dev/null) || true
}

# Removes formula $1 with every installed version (brew upgrade under
# HOMEBREW_NO_INSTALL_CLEANUP keeps the old keg, and a plain `brew uninstall`
# removes only the newest), then checks that no llm-speed keg is left.
brew_rm() {
  x brew uninstall --force --formula "$1"
  [ "$DRY" = 1 ] && return 0
  if brew list --formula "$PKG" >/dev/null 2>&1; then
    echo "  warning: $PKG still installed after brew uninstall --force (log $LOG)"
    echo "brew_rm: $PKG still installed" >>"$LOG"
    return 1
  fi
}

# Verifies the installed CLI in the current sandbox. $1=installer $2=scenario
# $3=llm-speed binary $4=python of the tool's venv (for the metadata fallback).
verify() {
  local inst="$1" scen="$2" bin="$3" py="$4" out ver meta from="${FROM:-}" want="${WANT:-$NEW}"
  FROM=""
  WANT=""
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] $bin --version   (must print $PKG $want)"
    echo "  [dry-run] $bin doctor --help"
    record "$inst" "$scen" DRY-RUN "" "not run"
    return
  fi
  out="$(HOME="$SB/home" XDG_CONFIG_HOME="$SB/config" XDG_DATA_HOME="$SB/data" \
    "$bin" --version 2>>"$LOG")" || out=""
  echo "--version: $out" >>"$LOG"
  ver="$(printf '%s\n' "$out" | sed -n 's/^llm-speed[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p' | head -1)"
  if ! HOME="$SB/home" XDG_CONFIG_HOME="$SB/config" XDG_DATA_HOME="$SB/data" \
    "$bin" doctor --help >>"$LOG" 2>&1; then
    record "$inst" "$scen" FAIL "${ver:-?}" "doctor --help failed (log $LOG)"
    return
  fi
  if [ "$ver" = "$want" ]; then
    record "$inst" "$scen" PASS "$from$ver" ""
    return
  fi
  meta=""
  if [ -n "$py" ] && [ -x "$py" ]; then
    meta="$(meta_version "$py")"
  fi
  if [ -z "$ver" ] && [ "$meta" = "$want" ]; then
    record "$inst" "$scen" FAIL-soft "${from}metadata $meta" "--version missing or unparsable"
  else
    record "$inst" "$scen" FAIL "$from${ver:-?}${meta:+ / metadata $meta}" "want $want (log $LOG)"
  fi
}

# Installs PREV and checks it really is PREV before the upgrade.
prev_ok() {
  local bin="$1" py="$2" meta
  [ "$DRY" = 1 ] && return 0
  meta="$(meta_version "$py")"
  echo "PREV installed: ${meta:-none}" >>"$LOG"
  FROM="${meta:-?} -> "
  [ "$meta" = "$PREV" ] && [ -x "$bin" ]
}

# 0 if version $1 ships the self-updater (>= 0.0.14).
has_updater() {
  "$PYTHON" -c 'import re,sys; v=tuple(int(x) for x in re.findall(r"\d+", sys.argv[1])[:3]); sys.exit(0 if v >= (0, 0, 14) else 1)' "$1"
}
# The upgrade step of the upgrade (documented command, "$@") or the release
# autoupdate row (`llm-speed update` with the PREV install's own updater).
up_or_update() {
  local scen="$1" bin="$2"; shift 2
  if [ "$scen" = autoupdate ]; then sx "$bin" update; else sx "$@"; fi
}
# N/A reason for the release autoupdate row, or nothing if it can run.
au_na() {
  if [ "$PINNED" = 1 ]; then echo "NEW is not the PyPI latest: \`llm-speed update\` goes to the latest"; return; fi
  if ! has_updater "$PREV"; then echo "PREV $PREV has no self-updater (it ships in 0.0.14); runs from PREV >= 0.0.14"; fi
}
# Runs the CLI with stdout on a pseudo-terminal (auto-update is off without a
# TTY) and stdin from /dev/null, without the CI variables (auto-update is off in
# CI, and this gate itself runs in CI); its stderr goes to stdout for the caller.
tty_run() {
  "$PYTHON" -c 'import os,pty,subprocess,sys
ci=("CI CONTINUOUS_INTEGRATION GITHUB_ACTIONS GITLAB_CI BUILDKITE CIRCLECI TRAVIS JENKINS_URL "
    "TEAMCITY_VERSION TF_BUILD BITBUCKET_BUILD_NUMBER CODEBUILD_BUILD_ID DRONE APPVEYOR "
    "SEMAPHORE BUILD_BUILDID").split()
env={k:v for k,v in os.environ.items() if k not in ci}
m,s=pty.openpty()
r=subprocess.run(sys.argv[1:],stdin=subprocess.DEVNULL,stdout=s,stderr=subprocess.PIPE,timeout=300,env=env)
os.close(s); os.close(m)
sys.stdout.write(r.stderr.decode("utf-8","replace")); sys.exit(r.returncode)' "$@"
}
# Pre-release autoupdate row with the local candidate wheel. $1=installer
row_autoupdate_local() {
  local inst="$1" py bin bump dir ver i notice
  new_sandbox "$inst" autoupdate
  dir="$(dirname "$AU_WHEEL")"
  case "$inst" in
    pip) py="$SB/venv/bin/python"; bin="$SB/venv/bin/llm-speed"
         x "$PYTHON" -m venv "$SB/venv" &&
         sx "$py" -m pip install --find-links "$dir" "$PKG" ;;
    pipx) py="$SB/pipx/venvs/$PKG/bin/python"; bin="$SB/bin/llm-speed"
         sx pipx install --python "$PYTHON" "$PKG" --pip-args="--find-links=$dir" ;;
    uv) py="$SB/uvtools/$PKG/bin/python"; bin="$SB/bin/llm-speed"
         sx uv tool install --python "$PYTHON" --find-links "$dir" "$PKG" ;;
  esac || { record "$inst" autoupdate FAIL "" "could not install the candidate wheel (log $LOG)"; drop_sandbox; return; }
  if [ "$DRY" = 0 ]; then
    ver="$(meta_version "$py")"
    [ "$ver" = "$NEW" ] || { record "$inst" autoupdate FAIL "${ver:-?}" "candidate is not $NEW (log $LOG)"; drop_sandbox; return; }
    FROM="$ver -> "
  fi
  x "$PYTHON" "$HERE/make-update-test-wheel.py" "$AU_WHEEL" "$SB/bump" ||
    { record "$inst" autoupdate FAIL "" "could not make the $BUMP test wheel (log $LOG)"; drop_sandbox; return; }
  bump="$SB/bump/llm_speed-$BUMP-py3-none-any.whl"
  # The detached background update (first run of the day), then wait for it.
  LLM_SPEED_UPDATE_TEST=1 LLM_SPEED_UPDATE_TEST_WHEEL="file://$(real_path "$bump")" sx tty_run "$bin" about
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] wait up to 300s for $bin --version to print $PKG $BUMP"
  else
    for i in $(seq 1 150); do
      ver="$(HOME="$SB/home" XDG_CONFIG_HOME="$SB/config" "$bin" --version 2>/dev/null | sed -n 's/^llm-speed[[:space:]]\{1,\}//p')"
      [ "$ver" = "$BUMP" ] && break
      sleep 2
    done
    echo "background update after ${i}x2s: ${ver:-?}" >>"$LOG"
  fi
  # The next run prints the one-time notice; the run after it doesn't.
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] tty_run $bin about   (must print 'Updated llm-speed $NEW -> $BUMP' once)"
  else
    notice="$(HOME="$SB/home" XDG_CONFIG_HOME="$SB/config" XDG_DATA_HOME="$SB/data" tty_run "$bin" about 2>&1)"
    echo "notice: $notice" >>"$LOG"
  fi
  if [ "$DRY" = 0 ] && ! printf '%s\n' "$notice" | grep -qx "Updated llm-speed $NEW -> $BUMP"; then
    record "$inst" autoupdate FAIL "${ver:-?}" "no one-time 'Updated llm-speed $NEW -> $BUMP' notice (log $LOG)"; drop_sandbox; return
  fi
  WANT="$BUMP"
  verify "$inst" "autoupdate (local)" "$bin" "$py"
  drop_sandbox
}

row_pip() {
  local scen="$1" na
  if [ "$scen" = autoupdate ]; then
    if [ -n "$AU_WHEEL" ]; then row_autoupdate_local pip; return; fi
    na="$(au_na)"; [ -z "$na" ] || { record pip "$scen" N/A "" "$na"; return; }
  fi
  new_sandbox pip "$scen"
  local py="$SB/venv/bin/python" bin="$SB/venv/bin/llm-speed"
  if ! x "$PYTHON" -m venv "$SB/venv"; then record pip "$scen" FAIL "" "venv failed"; drop_sandbox; return; fi
  if [ "$scen" = fresh ]; then
    sx "$py" -m pip install "$SPEC_NEW" || { record pip "$scen" FAIL "" "pip install$SFX failed (log $LOG)"; drop_sandbox; return; }
  else
    if ! { sx "$py" -m pip install "$PKG==$PREV"; } || ! prev_ok "$bin" "$py"; then
      record pip "$scen" FAIL "" "could not install PREV $PREV"; drop_sandbox; return
    fi
    up_or_update "$scen" "$bin" "$py" -m pip install -U "$SPEC_NEW" || { record pip "$scen" FAIL "" "pip $scen$SFX failed (log $LOG)"; drop_sandbox; return; }
  fi
  verify pip "$scen$SFX" "$bin" "$py"
  drop_sandbox
}

row_pipx() {
  local scen="$1" na
  [ "$DRY" = 1 ] || command -v pipx >/dev/null || { record pipx "$scen" SKIP "" "pipx not installed (try --bootstrap-tools)"; return; }
  if [ "$scen" = autoupdate ]; then
    if [ -n "$AU_WHEEL" ]; then row_autoupdate_local pipx; return; fi
    na="$(au_na)"; [ -z "$na" ] || { record pipx "$scen" N/A "" "$na"; return; }
  fi
  new_sandbox pipx "$scen"
  local py="$SB/pipx/venvs/$PKG/bin/python" bin="$SB/bin/llm-speed"
  if [ "$scen" = fresh ]; then
    sx pipx install --python "$PYTHON" "$SPEC_NEW" || { record pipx "$scen" FAIL "" "pipx install$SFX failed (log $LOG)"; drop_sandbox; return; }
  else
    if ! { sx pipx install --python "$PYTHON" "$PKG==$PREV"; } || ! prev_ok "$bin" "$py"; then
      record pipx "$scen" FAIL "" "could not install PREV $PREV"; drop_sandbox; return
    fi
    if [ "$PINNED" = 1 ]; then
      # pipx upgrade cannot target a version; runpip is the pinned equivalent.
      sx pipx runpip "$PKG" install -U "$PKG==$NEW"
    else
      up_or_update "$scen" "$bin" pipx upgrade "$PKG"
    fi || { record pipx "$scen" FAIL "" "pipx $scen$SFX failed (log $LOG)"; drop_sandbox; return; }
  fi
  verify pipx "$scen$SFX" "$bin" "$py"
  drop_sandbox
}

row_uv() {
  local scen="$1" rc na
  [ "$DRY" = 1 ] || command -v uv >/dev/null || { record uv "$scen" SKIP "" "uv not installed (try --bootstrap-tools)"; return; }
  if [ "$scen" = autoupdate ]; then
    if [ -n "$AU_WHEEL" ]; then row_autoupdate_local uv; return; fi
    na="$(au_na)"; [ -z "$na" ] || { record uv "$scen" N/A "" "$na"; return; }
  fi
  new_sandbox uv "$scen"
  local py="$SB/uvtools/$PKG/bin/python" bin="$SB/bin/llm-speed"
  if [ "$scen" = fresh ]; then
    sx uv tool install --python "$PYTHON" "$SPEC_NEW" || { record uv "$scen" FAIL "" "uv tool install$SFX failed (log $LOG)"; drop_sandbox; return; }
  else
    # `uv tool install llm-speed==PREV` records the pin, and `uv tool upgrade`
    # then refuses to move. A user who followed the docs installed unpinned
    # while PREV was the latest, so reproduce that: unpinned, resolved as of
    # just before NEW was uploaded, then upgrade resolving as of now.
    if [ "$PINNED" = 1 ] || [ -z "$NEW_UPLOADED" ]; then
      sx uv tool install --python "$PYTHON" "$PKG==$PREV"; rc=$?
    else
      sx uv tool install --python "$PYTHON" --exclude-newer "$NEW_UPLOADED" "$PKG"; rc=$?
    fi
    if [ "$rc" != 0 ] || ! prev_ok "$bin" "$py"; then
      record uv "$scen" FAIL "" "could not install PREV $PREV"; drop_sandbox; return
    fi
    if [ "$PINNED" = 1 ] || [ -z "$NEW_UPLOADED" ]; then
      # Pinned PREV: `uv tool upgrade` keeps the pin, so reinstall at NEW.
      sx uv tool install --python "$PYTHON" --reinstall "$PKG==$NEW"
    else
      up_or_update "$scen" "$bin" uv tool upgrade --exclude-newer "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PKG"
    fi || { record uv "$scen" FAIL "" "uv tool $scen$SFX failed (log $LOG)"; drop_sandbox; return; }
  fi
  verify uv "$scen$SFX" "$bin" "$py"
  drop_sandbox
}

# Read-only: the version in the published tap formula.
tap_version() {
  curl -fsSL --proto '=https' --max-time 30 \
    "https://raw.githubusercontent.com/$TAP_REPO/main/Formula/llm-speed.rb" 2>/dev/null |
    sed -n 's/.*llm[-_]speed-\([0-9][0-9a-z.]*\)\.tar\.gz.*/\1/p' | head -1
}

BREW_TAPPED_BY_US=0
# Writes the local gate tap's formula: version $1 from wheel $2, deps from the
# wheelhouse $3 (no network during the build), built with python@$4.
gate_formula() {
  local ver="$1" whl="$2" wh="$3" pv="$4" sha
  sha="$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$whl")"
  cat <<RB
# installer-gate test formula: local tap $GATE_TAP only, never published
class LlmSpeed < Formula
  desc "llm-speed installer-gate test build (local tap, never published)"
  homepage "https://llm-speed.com"
  url "file://$whl"
  version "$ver"
  sha256 "$sha"
  license "Apache-2.0"
  depends_on "python@$pv"

  def install
    system Formula["python@$pv"].opt_bin/"python$pv", "-m", "venv", libexec
    system libexec/"bin/python", "-m", "pip", "install", "--no-index", "--find-links", "$wh", "$PKG==$ver"
    bin.install_symlink libexec/"bin/llm-speed"
  end
end
RB
}

# The hook takes only a fully resolved path (no symlinked directory), e.g. macOS /var.
real_path() { if [ -e "$1" ]; then "$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; else printf '%s\n' "$1"; fi; }

GATE_TAP="llm-speed-gate/local"
# Pre-release brew autoupdate row: a local tap (`brew tap-new --no-git`, never
# meadow-kun/tap) whose formula installs the candidate wheel, then is switched
# to NEW.post1; the background updater must run `brew upgrade` to it.
row_brew_autoupdate_local() {
  local prefix taprepo pv wh bump ver i notice brew_py
  [ "$DRY" = 1 ] || command -v brew >/dev/null || { record brew autoupdate FAIL "" "brew not installed"; return; }
  if [ "$DRY" = 0 ] && brew list --formula "$PKG" >/dev/null 2>&1; then
    record brew autoupdate SKIP "" "$PKG already installed in this brew; not touching it"; return
  fi
  new_sandbox brew autoupdate
  export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ANALYTICS=1 \
    HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_CACHE="$SB/cache/brew"
  prefix="$(brew --prefix 2>/dev/null || echo "<brew prefix>")"
  pv=3.12
  if [ "$DRY" = 0 ]; then
    pv=""
    for v in 3.13 3.12 3.11; do [ -x "$prefix/opt/python@$v/bin/python$v" ] && { pv="$v"; break; }; done
    if [ -z "$pv" ]; then x brew install python@3.12 && pv=3.12; fi
    [ -n "$pv" ] || { record brew autoupdate FAIL "" "no Homebrew python@3.x (log $LOG)"; drop_sandbox; return; }
  fi
  brew_py="$prefix/opt/python@$pv/bin/python$pv"
  wh="$SB/wheelhouse"
  [ "$DRY" = 1 ] || mkdir -p "$wh"
  x "$PYTHON" "$HERE/make-update-test-wheel.py" "$AU_WHEEL" "$wh" ||
    { record brew autoupdate FAIL "" "could not make the $BUMP test wheel (log $LOG)"; drop_sandbox; return; }
  bump="$wh/llm_speed-$BUMP-py3-none-any.whl"
  if ! { x cp "$AU_WHEEL" "$wh/" && x "$brew_py" -m pip download -q --only-binary=:all: -d "$wh" "$AU_WHEEL"; }; then
    record brew autoupdate FAIL "" "could not fetch dependency wheels (log $LOG)"; drop_sandbox; return
  fi
  x brew tap-new --no-git "$GATE_TAP" || { record brew autoupdate FAIL "" "brew tap-new failed (log $LOG)"; drop_sandbox; return; }
  taprepo="$(brew --repository "$GATE_TAP" 2>/dev/null || echo "<gate tap>")"
  if [ "$DRY" = 1 ]; then echo "  [dry-run] write $taprepo/Formula/$PKG.rb ($NEW, $AU_WHEEL)"
  else gate_formula "$NEW" "$wh/$(basename "$AU_WHEEL")" "$wh" "$pv" >"$taprepo/Formula/$PKG.rb"; fi
  if ! x brew install "$GATE_TAP/$PKG"; then
    record brew autoupdate FAIL "" "brew install of the candidate failed (log $LOG)"
    x brew untap "$GATE_TAP"; drop_sandbox; return
  fi
  if [ "$DRY" = 0 ]; then
    ver="$("$prefix/bin/llm-speed" --version 2>/dev/null | sed -n 's/^llm-speed[[:space:]]\{1,\}//p')"
    FROM="${ver:-?} -> "
  fi
  # Switch the formula to NEW.post1; the updater runs `brew upgrade` itself.
  if [ "$DRY" = 1 ]; then echo "  [dry-run] write $taprepo/Formula/$PKG.rb ($BUMP, $bump)"
  else gate_formula "$BUMP" "$bump" "$wh" "$pv" >"$taprepo/Formula/$PKG.rb"; fi
  LLM_SPEED_UPDATE_TEST=1 LLM_SPEED_UPDATE_TEST_WHEEL="file://$(real_path "$bump")" sx tty_run "$prefix/bin/llm-speed" about
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] wait up to 300s for $prefix/bin/llm-speed --version to print $PKG $BUMP"
    echo "  [dry-run] tty_run $prefix/bin/llm-speed about   (must print 'Updated llm-speed $NEW -> $BUMP' once)"
  else
    for i in $(seq 1 150); do
      ver="$("$prefix/bin/llm-speed" --version 2>/dev/null | sed -n 's/^llm-speed[[:space:]]\{1,\}//p')"
      [ "$ver" = "$BUMP" ] && break
      sleep 2
    done
    echo "background update after ${i}x2s: ${ver:-?}" >>"$LOG"
    notice="$(HOME="$SB/home" XDG_CONFIG_HOME="$SB/config" XDG_DATA_HOME="$SB/data" tty_run "$prefix/bin/llm-speed" about 2>&1)"
    echo "notice: $notice" >>"$LOG"
    if ! printf '%s\n' "$notice" | grep -qx "Updated llm-speed $NEW -> $BUMP"; then
      record brew autoupdate FAIL "${ver:-?}" "no one-time 'Updated llm-speed $NEW -> $BUMP' notice (log $LOG)"
      brew_rm "$GATE_TAP/$PKG"; x brew untap "$GATE_TAP"; drop_sandbox; return
    fi
  fi
  WANT="$BUMP"
  verify brew "autoupdate (local tap)" "$prefix/bin/llm-speed" "$prefix/opt/$PKG/libexec/bin/python"
  brew_rm "$GATE_TAP/$PKG"
  x brew untap "$GATE_TAP"
  drop_sandbox
}

row_brew() {
  local scen="$1" tv prefix taprepo orig old na
  if [ "$scen" = autoupdate ]; then
    if [ -n "$AU_WHEEL" ]; then row_brew_autoupdate_local; return; fi
    na="$(au_na)"; [ -z "$na" ] || { record brew "$scen" N/A "" "$na"; return; }
  fi
  if [ "$PINNED" = 1 ]; then
    record brew "$scen" SKIP "" "NEW is not the PyPI latest; the tap only carries the latest formula"
    return
  fi
  if [ "$DRY" = 0 ]; then
    tv="$(tap_version)"
    if [ "$tv" != "$NEW" ]; then
      record brew "$scen" FAIL "tap ${tv:-unreadable}" "$TAP_REPO formula is not at $NEW: the tap bump lags PyPI; re-run after it lands"
      return
    fi
  fi
  [ "$DRY" = 1 ] || command -v brew >/dev/null || { record brew "$scen" SKIP "" "brew not installed"; return; }
  if [ "$DRY" = 0 ] && brew list --formula "$PKG" >/dev/null 2>&1; then
    record brew "$scen" SKIP "" "$PKG already installed in this brew; not touching it"
    return
  fi
  new_sandbox brew "$scen"
  export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ANALYTICS=1 \
    HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_CACHE="$SB/cache/brew"
  if [ "$DRY" = 0 ] && ! brew tap | grep -qx "$TAP"; then BREW_TAPPED_BY_US=1; fi
  x brew tap "$TAP" || { record brew "$scen" FAIL "" "brew tap $TAP failed (log $LOG)"; drop_sandbox; return; }
  prefix="$(brew --prefix 2>/dev/null || echo "<brew prefix>")"
  if [ "$scen" = fresh ]; then
    x brew install "$TAP/$PKG" || { record brew "$scen" FAIL "" "brew install failed (log $LOG)"; brew_rm "$PKG"; drop_sandbox; return; }
  else
    # The tap keeps one formula: install PREV from the tap's history, then go
    # back to the tap's main and run the documented `brew upgrade`.
    taprepo="$(brew --repo "$TAP" 2>/dev/null || echo "<tap repo>")"
    old="<tap commit with $PREV>"; orig=""
    if [ "$DRY" = 0 ]; then
      old=""
      orig="$(git -C "$taprepo" rev-parse HEAD 2>/dev/null)"
      # The tap's previous formula: PREV if the tap carried it, else the last
      # version it did carry before NEW (the tap can skip a PyPI release).
      for c in $(git -C "$taprepo" rev-list HEAD -- Formula/llm-speed.rb 2>/dev/null); do
        tv="$(git -C "$taprepo" show "$c:Formula/llm-speed.rb" 2>/dev/null |
          sed -n 's/.*llm[-_]speed-\([0-9][0-9a-z.]*\)\.tar\.gz.*/\1/p' | head -1)"
        if [ -n "$tv" ] && [ "$tv" != "$NEW" ]; then old="$c"; FROM="$tv -> "; break; fi
      done
      [ -n "$old" ] || { record brew "$scen" FAIL "" "no earlier formula in the tap history"; drop_sandbox; return; }
      [ "$tv" = "$PREV" ] || echo "note: the tap never carried $PREV; upgrading from its previous formula $tv" >>"$LOG"
    fi
    x git -C "$taprepo" -c advice.detachedHead=false checkout -q "$old" -- Formula/llm-speed.rb
    if ! x brew install "$TAP/$PKG"; then
      x git -C "$taprepo" checkout -q "${orig:-HEAD}" -- Formula/llm-speed.rb
      record brew "$scen" FAIL "" "could not brew install PREV $PREV (log $LOG)"; drop_sandbox; return
    fi
    x git -C "$taprepo" checkout -q "${orig:-HEAD}" -- Formula/llm-speed.rb
    if [ "$scen" = autoupdate ]; then
      # The updater runs `brew upgrade meadow-kun/tap/llm-speed` itself.
      sx "$prefix/bin/llm-speed" update
    else
      x brew upgrade "$TAP/$PKG"
    fi || { record brew "$scen" FAIL "" "brew $scen failed (log $LOG)"; brew_rm "$PKG"; drop_sandbox; return; }
  fi
  verify brew "$scen" "$prefix/bin/llm-speed" "$prefix/opt/$PKG/libexec/bin/python"
  brew_rm "$PKG"
  drop_sandbox
}

for inst in $ONLY; do
  for scen in $SCENARIOS; do
    echo "== $inst $scen"
    "row_$inst" "$scen"
  done
done
if [ "$BREW_TAPPED_BY_US" = 1 ]; then LOG=/dev/null; x brew untap "$TAP"; fi

echo
printf '%-6s %-18s %-10s %-26s %s\n' INSTALLER SCENARIO STATUS VERSION DETAIL
fails=0 skips=0 nas=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r a b c d e <<<"$r"
  printf '%-6s %-18s %-10s %-26s %s\n' "$a" "$b" "$c" "${d:--}" "$e"
  case "$c" in FAIL|FAIL-soft) fails=$((fails + 1)) ;; SKIP) skips=$((skips + 1)) ;; N/A) nas=$((nas + 1)) ;; esac
done
if [ "$DRY" = 1 ]; then verdict="dry run: nothing was installed"; rc=0
elif [ "$fails" -gt 0 ]; then verdict="installer gate: FAIL ($fails failing, $skips skipped)"; rc=1
elif [ "$skips" -gt 0 ]; then verdict="installer gate: NOT DONE ($skips skipped; a SKIP is not a pass)"; rc=3
else verdict="installer gate: PASS"; rc=0
fi
[ "$nas" = 0 ] || verdict="$verdict; $nas N/A (not applicable to this pair, see DETAIL)"
echo "$verdict"
if [ -n "$SUMMARY" ]; then
  {
    echo "#### $(uname -s) $(uname -m): $PREV -> $NEW (PyPI latest: ${LATEST:-unknown})"
    echo
    echo "| Installer | Scenario | Status | Version | Detail |"
    echo "| --- | --- | --- | --- | --- |"
    for r in "${RESULTS[@]}"; do
      IFS='|' read -r a b c d e <<<"$r"
      echo "| $a | $b | $c | ${d:--} | $(printf '%s' "$e" | tr '|' '/') |"
    done
    echo
    echo "**$verdict**"
    echo
  } >>"$SUMMARY"
fi
[ "$rc" = 0 ] || exit "$rc"
