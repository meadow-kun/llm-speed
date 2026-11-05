<#
.SYNOPSIS
  Installer release gate, PowerShell twin of scripts/check-installers.sh (Windows;
  also runs under pwsh on macOS/Linux).

.DESCRIPTION
  For pip (fresh venv), pipx and uv tool: a fresh install of NEW, and an install
  of PREV followed by the documented upgrade command (pip install -U, pipx
  upgrade, uv tool upgrade). After each, `llm-speed --version` must print NEW
  (FAIL-soft if only the installed metadata says NEW) and `llm-speed doctor
  --help` must run. Nothing benchmarks, submits or uploads anything. Each row
  runs in its own throwaway HOME/USERPROFILE/APPDATA/PIPX_HOME/UV_TOOL_DIR under
  the run root and is deleted as soon as it finishes. Homebrew is covered by the
  bash script on macOS/Linux. CI runs it on windows-latest via
  .github/workflows/installer-gate.yml on the public mirror meadow-kun/llm-speed.
  Standalone: needs only pwsh, Python and the installers, never a source checkout.

  Scenario autoupdate (0.0.14+, CLI.md "Auto-update"):
  - with -AutoUpdateWheel W (pre-release): install the candidate W
    (llm_speed-NEW-py3-none-any.whl, deps from PyPI) unpinned, copy it to NEW.post1
    (make-update-test-wheel.py next to this script), run `llm-speed about` in its
    own console (auto-update is off without one) with the test-only hook
    LLM_SPEED_UPDATE_TEST=1 + LLM_SPEED_UPDATE_TEST_WHEEL=file:///<NEW.post1 wheel>
    and the CI variables removed, wait for the detached background update (on
    Windows it starts once that llm-speed has exited, so the running .exe is not
    locked), then check `--version` is NEW.post1 and the next run prints
    `Updated llm-speed NEW -> NEW.post1`.
  - otherwise (release): install PREV as in the upgrade row and run
    `llm-speed update` instead of the upgrade command. N/A when PREV has no updater
    (< 0.0.14) or the rows are pinned; N/A is neither a pass nor a failure.

  Run root: $env:INSTALLER_GATE_ROOT, else ~/.cache/llm-speed-installer-gate
  (Join-Path $HOME '.cache/llm-speed-installer-gate', the same as the bash
  script; on Windows $HOME is the user profile). Never /tmp.

  -Summary FILE also appends the result table as Markdown (e.g. $env:GITHUB_STEP_SUMMARY).

  Exit: 0 every row PASS; 1 any FAIL or FAIL-soft; 3 a SKIP and no FAIL (a SKIP
  is not-done for the release); 2 usage error.

.EXAMPLE
  pwsh -File scripts/check-installers.ps1 -New 0.0.11 -Prev 0.0.10 -BootstrapTools
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$New,
  [Parameter(Mandatory = $true)][string]$Prev,
  [string[]]$Only = @('pip', 'pipx', 'uv'),
  [string[]]$Scenarios = @('fresh', 'upgrade', 'autoupdate'),
  [string]$AutoUpdateWheel = '',
  [switch]$BootstrapTools,
  [switch]$DryRun,
  [switch]$Keep,
  [string]$Python = '',
  [string]$Summary = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$Pkg = 'llm-speed'
$vre = '^[0-9]+(\.[0-9]+)*([a-z]+[0-9]*)?$'
foreach ($v in @($New, $Prev)) {
  if ($v -notmatch $vre) { Write-Error "check-installers: bad version '$v'"; exit 2 }
}
if ($New -eq $Prev) { Write-Error 'check-installers: NEW and PREV are the same'; exit 2 }
$Only = @($Only | ForEach-Object { $_ -split '[ ,]+' } | Where-Object { $_ })
foreach ($i in $Only) {
  if ($i -notin @('pip', 'pipx', 'uv')) { Write-Error "check-installers: unknown installer '$i' (brew: use the bash script)"; exit 2 }
}

$Scenarios = @($Scenarios | ForEach-Object { $_ -split '[ ,]+' } | Where-Object { $_ })
foreach ($i in $Scenarios) {
  if ($i -notin @('fresh', 'upgrade', 'autoupdate')) { Write-Error "check-installers: unknown scenario '$i'"; exit 2 }
}
if ($AutoUpdateWheel) {
  if ((Split-Path -Leaf $AutoUpdateWheel) -ne "llm_speed-$New-py3-none-any.whl") {
    Write-Error "check-installers: -AutoUpdateWheel must be llm_speed-$New-py3-none-any.whl"; exit 2
  }
  if (-not $DryRun -and -not (Test-Path -PathType Leaf $AutoUpdateWheel)) { Write-Error "check-installers: no such wheel $AutoUpdateWheel"; exit 2 }
  if (Test-Path $AutoUpdateWheel) { $AutoUpdateWheel = (Resolve-Path $AutoUpdateWheel).Path }
}
$Bump = "$New.post1"
# Rows use their sandbox's own config; a caller's settings must not leak in.
foreach ($n in 'LLM_SPEED_CONFIG', 'LLM_SPEED_CACHE', 'LLM_SPEED_DISABLE_AUTOUPDATE', 'LLM_SPEED_UPDATE_TEST', 'LLM_SPEED_UPDATE_TEST_WHEEL') {
  Remove-Item "Env:$n" -ErrorAction SilentlyContinue
}
$CiVars = @('CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'GITLAB_CI', 'BUILDKITE', 'CIRCLECI', 'TRAVIS', 'JENKINS_URL',
  'TEAMCITY_VERSION', 'TF_BUILD', 'BITBUCKET_BUILD_NUMBER', 'CODEBUILD_BUILD_ID', 'DRONE', 'APPVEYOR', 'SEMAPHORE', 'BUILD_BUILDID')

$OnWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or $IsWindows
$BinSub = if ($OnWindows) { 'Scripts' } else { 'bin' }
$Exe = if ($OnWindows) { '.exe' } else { '' }
if (-not $Python) { $Python = if ($OnWindows) { 'python' } else { 'python3' } }

if ($env:INSTALLER_GATE_ROOT) { $Root = $env:INSTALLER_GATE_ROOT }
else { $Root = Join-Path (Join-Path $HOME '.cache') 'llm-speed-installer-gate' }
if ($Root -match '^(/private)?/(var/)?tmp(/|$)') { Write-Error "check-installers: refusing a run root under /tmp ($Root)"; exit 2 }
$Run = Join-Path $Root ("run-$New-" + (Get-Date -Format 'yyyyMMddHHmmss') + "-$PID")

$Results = New-Object System.Collections.Generic.List[object]
function Add-Result($inst, $scen, $status, $ver, $detail) {
  $Results.Add([pscustomobject]@{ Installer = $inst; Scenario = $scen; Status = $status; Version = $ver; Detail = $detail })
  Write-Host "  -> $inst ${scen}: $status $ver $detail"
  if ($status -in 'FAIL', 'FAIL-soft' -and $script:Log -and (Test-Path $script:Log)) {
    Write-Host "::group::$inst $scen row log (last 200 lines)"
    Get-Content -Path $script:Log -Tail 200 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    Write-Host "::endgroup::"
  }
}

$script:Log = $null
# Runs a native command (or prints it in -DryRun); output goes to the row log.
function X {
  param([string]$Cmd, [Parameter(ValueFromRemainingArguments = $true)][string[]]$CmdArgs)
  if ($DryRun) { Write-Host "  [dry-run] $Cmd $($CmdArgs -join ' ')"; return $true }
  if ($script:Log) { Add-Content -Path $script:Log -Value "+ $Cmd $($CmdArgs -join ' ')" }
  $out = & $Cmd @CmdArgs 2>&1
  $rc = $LASTEXITCODE
  if ($script:Log) { $out | ForEach-Object { "$_" } | Add-Content -Path $script:Log }
  return ($rc -eq 0)
}

function Get-Json($url) {
  try { return Invoke-RestMethod -Uri $url -TimeoutSec 30 } catch { return $null }
}
$Latest = $New; $NewUploaded = '<NEW upload time>'
if (-not $DryRun) {
  $j = Get-Json "https://pypi.org/pypi/$Pkg/json"
  $Latest = if ($j) { $j.info.version } else { '' }
  $j2 = Get-Json "https://pypi.org/pypi/$Pkg/$New/json"
  $NewUploaded = if ($j2 -and $j2.urls) { ($j2.urls | ForEach-Object {
      # Invoke-RestMethod may already have parsed the timestamp into a DateTime.
      $t = $_.upload_time_iso_8601
      if ($t -is [datetime]) { $t.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ') } else { "$t" }
    } | Sort-Object)[0] } else { '' }
}
$Pinned = ($Latest -ne $New)
$SpecNew = if ($Pinned) { "$Pkg==$New" } else { $Pkg }
$Sfx = if ($Pinned) { ' (pinned)' } else { '' }

Write-Host "== llm-speed installer gate: NEW=$New PREV=$Prev (PyPI latest: $Latest) on $([System.Environment]::OSVersion.VersionString)"
Write-Host "   run root: $Run$(if ($Pinned) { ' ; NEW is not the PyPI latest, rows are pinned' })"
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $Run | Out-Null }

$SavedPath = $env:PATH
if ($BootstrapTools) {
  $need = @()
  if ('pipx' -in $Only -and -not (Get-Command pipx -ErrorAction SilentlyContinue)) { $need += 'pipx' }
  if ('uv' -in $Only -and -not (Get-Command uv -ErrorAction SilentlyContinue)) { $need += 'uv' }
  if ($need.Count -gt 0) {
    Write-Host "== bootstrapping: $($need -join ' ')"
    $script:Log = Join-Path $Run 'bootstrap.log'
    $tools = Join-Path $Run 'tools'
    $env:PIP_CACHE_DIR = Join-Path $Run 'tools-cache'
    if ((X $Python -m venv $tools) -and (X (Join-Path $tools "$BinSub/python$Exe") -m pip install -q @need)) {
      $env:PATH = (Join-Path $tools $BinSub) + [IO.Path]::PathSeparator + $env:PATH
      $SavedPath = $env:PATH
    } else { Write-Host "   bootstrap failed (see $script:Log)" }
    Remove-Item Env:PIP_CACHE_DIR -ErrorAction SilentlyContinue
  }
}

$IsoVars = @('HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
  'XDG_STATE_HOME', 'PIPX_HOME', 'PIPX_BIN_DIR', 'PIPX_MAN_DIR', 'UV_TOOL_DIR', 'UV_TOOL_BIN_DIR', 'UV_CACHE_DIR',
  'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_DOWNLOADS', 'PIP_CACHE_DIR', 'PIP_DISABLE_PIP_VERSION_CHECK', 'PIP_NO_INPUT')
$Saved = @{}
foreach ($n in $IsoVars) { $Saved[$n] = [Environment]::GetEnvironmentVariable($n) }

function Enter-Sandbox($inst, $scen) {
  $script:SB = Join-Path $Run "$inst-$scen"
  $script:Log = "$script:SB.log"
  if ($DryRun) { return }
  foreach ($d in 'home', 'bin', 'cache', 'config', 'data', 'state', 'appdata', 'localappdata') {
    New-Item -ItemType Directory -Force -Path (Join-Path $script:SB $d) | Out-Null
  }
  Set-Content -Path $script:Log -Value ''
  $sb = $script:SB
  # The tool resolves its Python from PATH, so set these only after it is chosen.
  $env:HOME = "$sb/home"; $env:USERPROFILE = "$sb/home"
  $env:APPDATA = "$sb/appdata"; $env:LOCALAPPDATA = "$sb/localappdata"
  $env:XDG_CACHE_HOME = "$sb/cache"; $env:XDG_CONFIG_HOME = "$sb/config"
  $env:XDG_DATA_HOME = "$sb/data"; $env:XDG_STATE_HOME = "$sb/state"
  $env:PIPX_HOME = "$sb/pipx"; $env:PIPX_BIN_DIR = "$sb/bin"; $env:PIPX_MAN_DIR = "$sb/man"
  $env:UV_TOOL_DIR = "$sb/uvtools"; $env:UV_TOOL_BIN_DIR = "$sb/bin"; $env:UV_CACHE_DIR = "$sb/cache/uv"
  $env:UV_PYTHON_INSTALL_DIR = "$sb/uvpython"; $env:UV_PYTHON_DOWNLOADS = 'never'
  $env:PIP_CACHE_DIR = "$sb/cache/pip"; $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'; $env:PIP_NO_INPUT = '1'
  $env:PATH = "$sb/bin" + [IO.Path]::PathSeparator + $SavedPath
}
function Exit-Sandbox {
  foreach ($n in $IsoVars) { [Environment]::SetEnvironmentVariable($n, $Saved[$n]) }
  $env:PATH = $SavedPath
  if (-not $DryRun -and -not $Keep -and (Test-Path $script:SB)) {
    Remove-Item -Recurse -Force -Path $script:SB -ErrorAction SilentlyContinue
  }
}

function Get-MetaVersion($py) {
  if (-not (Test-Path $py)) { return '' }
  # Isolated (-I) and from the row's sandbox, so a source checkout's
  # llm_speed.egg-info in the current directory can't shadow the install.
  $here = if ($script:SB -and (Test-Path $script:SB)) { $script:SB } else { [System.IO.Path]::GetTempPath() }
  Push-Location $here
  try { $v = & $py -I -c "import importlib.metadata as m; print(m.version('$Pkg'))" 2>$null; $rc = $LASTEXITCODE }
  finally { Pop-Location }
  if ($rc -eq 0) { return "$v".Trim() } else { return '' }
}

$script:From = ''
function Test-Prev($bin, $py) {
  if ($DryRun) { return $true }
  $m = Get-MetaVersion $py
  Add-Content -Path $script:Log -Value "PREV installed: $m"
  $script:From = "$m -> "
  return (($m -eq $Prev) -and (Test-Path $bin))
}

function Test-Installed($inst, $scen, $bin, $py, $want = $New) {
  $from = $script:From; $script:From = ''
  if ($DryRun) {
    Write-Host "  [dry-run] $bin --version   (must print $Pkg $want)"
    Write-Host "  [dry-run] $bin doctor --help"
    Add-Result $inst $scen 'DRY-RUN' '' 'not run'; return
  }
  $out = ''
  if (Test-Path $bin) { $out = (& $bin --version 2>>$script:Log | Out-String).Trim() }
  Add-Content -Path $script:Log -Value "--version: $out"
  $ver = ''
  if ($out -match '^llm-speed\s+(\S+)') { $ver = $Matches[1] }
  $docOk = $false
  if (Test-Path $bin) { $docOk = X $bin doctor --help }
  if (-not $docOk) { Add-Result $inst $scen 'FAIL' "$from$ver" "doctor --help failed (log $script:Log)"; return }
  if ($ver -eq $want) { Add-Result $inst $scen 'PASS' "$from$ver" ''; return }
  $meta = Get-MetaVersion $py
  if (-not $ver -and $meta -eq $want) { Add-Result $inst $scen 'FAIL-soft' "${from}metadata $meta" '--version missing or unparsable' }
  else { Add-Result $inst $scen 'FAIL' "$from$ver / metadata $meta" "want $want (log $script:Log)" }
}

# 0.0.14+ ships the self-updater.
function Test-HasUpdater($v) {
  $n = @([regex]::Matches($v, '\d+') | Select-Object -First 3 | ForEach-Object { [int]$_.Value })
  while ($n.Count -lt 3) { $n += 0 }
  return ([version]::new($n[0], $n[1], $n[2]) -ge [version]::new(0, 0, 14))
}
function Get-AutoUpdateNA {
  if ($Pinned) { return 'NEW is not the PyPI latest: `llm-speed update` goes to the latest' }
  if (-not (Test-HasUpdater $Prev)) { return "PREV $Prev has no self-updater (it ships in 0.0.14); runs from PREV >= 0.0.14" }
  return ''
}
# Runs `llm-speed about` in its own console (stdout a terminal), without the CI
# variables; returns its stderr.
function Invoke-Console($bin, [hashtable]$extra = @{}) {
  $err = Join-Path $script:SB 'console.err'
  $saved = @{}
  foreach ($n in $CiVars + @($extra.Keys)) { $saved[$n] = [Environment]::GetEnvironmentVariable($n) }
  try {
    foreach ($n in $CiVars) { [Environment]::SetEnvironmentVariable($n, $null) }
    foreach ($k in $extra.Keys) { [Environment]::SetEnvironmentVariable($k, $extra[$k]) }
    if ($OnWindows) {
      Start-Process -FilePath $bin -ArgumentList 'about' -Wait -RedirectStandardError $err | Out-Null
      $e = if (Test-Path $err) { Get-Content -Raw $err } else { '' }
    } else {
      $code = 'import os,pty,subprocess,sys' + "`n" + 'm,s=pty.openpty()' + "`n" +
        'r=subprocess.run(sys.argv[1:],stdin=subprocess.DEVNULL,stdout=s,stderr=subprocess.PIPE,timeout=300)' + "`n" +
        'os.close(s); os.close(m); sys.stdout.write(r.stderr.decode("utf-8","replace"))'
      $e = (& $Python -c $code $bin about | Out-String)
    }
  } finally {
    foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) }
  }
  Add-Content -Path $script:Log -Value "console stderr: $e"
  return "$e"
}
function Invoke-AutoUpdateLocal($inst) {
  Enter-Sandbox $inst 'autoupdate'
  $sb = $script:SB
  try {
    $dir = Split-Path -Parent $AutoUpdateWheel
    switch ($inst) {
      'pip' { $py = "$sb/venv/$BinSub/python$Exe"; $bin = "$sb/venv/$BinSub/llm-speed$Exe"
              $ok = (X $Python -m venv "$sb/venv") -and (X $py -m pip install --find-links $dir $Pkg) }
      'pipx' { $py = "$sb/pipx/venvs/$Pkg/$BinSub/python$Exe"; $bin = "$sb/bin/llm-speed$Exe"
               $ok = X pipx install --python $Python $Pkg "--pip-args=--find-links=$dir" }
      'uv' { $py = "$sb/uvtools/$Pkg/$BinSub/python$Exe"; $bin = "$sb/bin/llm-speed$Exe"
             $ok = X uv tool install --python $Python --find-links $dir $Pkg }
    }
    if (-not $ok) { Add-Result $inst 'autoupdate' 'FAIL' '' "could not install the candidate wheel (log $script:Log)"; return }
    if (-not $DryRun) {
      $m = Get-MetaVersion $py
      if ($m -ne $New) { Add-Result $inst 'autoupdate' 'FAIL' $m "candidate is not $New (log $script:Log)"; return }
      $script:From = "$m -> "
    }
    if (-not (X $Python (Join-Path $PSScriptRoot 'make-update-test-wheel.py') $AutoUpdateWheel "$sb/bump")) {
      Add-Result $inst 'autoupdate' 'FAIL' '' "could not make the $Bump test wheel (log $script:Log)"; return
    }
    $wheel = (Join-Path "$sb/bump" "llm_speed-$Bump-py3-none-any.whl")
    # the hook takes only a fully resolved path (no symlinked directory)
    if (Test-Path $wheel) { $wheel = (& $Python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' $wheel | Out-String).Trim() }
    $uri = ([System.Uri]::new($wheel)).AbsoluteUri
    if ($DryRun) {
      Write-Host "  [dry-run] $bin about   (own console, LLM_SPEED_UPDATE_TEST=1, LLM_SPEED_UPDATE_TEST_WHEEL=$uri)"
      Write-Host "  [dry-run] wait up to 300s for $bin --version to print $Pkg $Bump"
      Write-Host "  [dry-run] $bin about   (must print 'Updated llm-speed $New -> $Bump' once)"
    } else {
      Invoke-Console $bin @{ LLM_SPEED_UPDATE_TEST = '1'; LLM_SPEED_UPDATE_TEST_WHEEL = $uri } | Out-Null
      $ver = ''
      for ($i = 0; $i -lt 150; $i++) {
        $out = (& $bin --version 2>$null | Out-String).Trim()
        if ($out -match '^llm-speed\s+(\S+)') { $ver = $Matches[1] }
        if ($ver -eq $Bump) { break }
        Start-Sleep -Seconds 2
      }
      Add-Content -Path $script:Log -Value "background update: $ver"
      $notice = Invoke-Console $bin
      if ($notice -notmatch [regex]::Escape("Updated llm-speed $New -> $Bump")) {
        Add-Result $inst 'autoupdate' 'FAIL' $ver "no one-time 'Updated llm-speed $New -> $Bump' notice (log $script:Log)"; return
      }
    }
    Test-Installed $inst 'autoupdate (local)' $bin $py $Bump
  } finally { Exit-Sandbox }
}

# Pre-release row (mirrors check-installers.sh row_autoupdate_fallback_nopkg):
# block `packaging` in the candidate venv so cli.autoupdate takes the regex /
# PyPI-JSON path. On 0.0.14 that path printed _RegexVersion(...) and never updated.
# R-144: the inline python runs -I from inside the sandbox, and `update --check`
# must exit 0 and print a version.
$FallbackPy = @'
import json, urllib.request
from cli import autoupdate as au
try:
    import packaging.version  # noqa: F401
    raise SystemExit("packaging still importable")
except ImportError:
    pass
assert isinstance(au.parse_version("0.0.15"), au._RegexVersion)
data = json.load(urllib.request.urlopen("https://pypi.org/pypi/llm-speed/json", timeout=30))
v = au.pick_version(data, prereleases=False)
assert isinstance(v, str) and v and "_RegexVersion" not in v, repr(v)
assert au.parse_version(v) is not None
assert str(au.parse_version(v)) == v
print("fallback-ok", v)
'@
$BlockerInit = @'
# installer-gate: force cli.autoupdate's regex fallback (no packaging).
raise ImportError("installer-gate: packaging blocked")
'@
$BlockerSite = @'
# Belt-and-suspenders: also refuse packaging via meta_path.
import sys
class _NoPackaging:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "packaging" or fullname.startswith("packaging."):
            raise ImportError("installer-gate: packaging blocked")
        return None
sys.meta_path.insert(0, _NoPackaging())
'@
function Invoke-AutoUpdateFallback {
  $scen = 'autoupdate-fallback'
  Enter-Sandbox pip $scen
  $sb = $script:SB
  try {
    $dir = Split-Path -Parent $AutoUpdateWheel
    $py = "$sb/venv/$BinSub/python$Exe"; $bin = "$sb/venv/$BinSub/llm-speed$Exe"
    if (-not ((X $Python -m venv "$sb/venv") -and (X $py -m pip install --find-links $dir $Pkg))) {
      Add-Result pip $scen 'FAIL' '' "could not install the candidate wheel (log $script:Log)"; return
    }
    if ($DryRun) {
      Write-Host "  [dry-run] plant packaging blocker + sitecustomize in $sb/venv"
      Write-Host "  [dry-run] (cd $sb) $py -I -   (pick_version(PyPI JSON) with packaging blocked)"
      Write-Host "  [dry-run] (cd $sb) $bin update --check   (exit 0, prints a version, no _RegexVersion)"
      Add-Result pip $scen 'DRY-RUN' '' 'not run'; return
    }
    $m = Get-MetaVersion $py
    if ($m -ne $New) { Add-Result pip $scen 'FAIL' $m "candidate is not $New (log $script:Log)"; return }
    Push-Location $sb
    try { $site = (& $py -I -c 'import site; print(site.getsitepackages()[-1])' | Out-String).Trim() } finally { Pop-Location }
    X $py -m pip uninstall -y packaging | Out-Null
    Get-ChildItem -Path $site -Filter 'packaging*' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    New-Item -ItemType Directory -Force -Path (Join-Path $site 'packaging') | Out-Null
    Set-Content -Path (Join-Path (Join-Path $site 'packaging') '__init__.py') -Value $BlockerInit
    Set-Content -Path (Join-Path $site 'sitecustomize.py') -Value $BlockerSite
    $pyFile = Join-Path $sb 'fallback-check.py'
    Set-Content -Path $pyFile -Value $FallbackPy
    Push-Location $sb
    try { $ok = X $py -I $pyFile } finally { Pop-Location }
    if (-not $ok) { Add-Result pip $scen 'FAIL' '' "forced packaging-fallback check failed (log $script:Log)"; return }
    Push-Location $sb
    try { $notice = (& $bin update --check 2>&1 | Out-String); $rc = $LASTEXITCODE } finally { Pop-Location }
    Add-Content -Path $script:Log -Value "update --check (exit $rc): $notice"
    if ($rc -ne 0) { Add-Result pip $scen 'FAIL' '' "update --check exited $rc (log $script:Log)"; return }
    if ($notice -match '_RegexVersion') { Add-Result pip $scen 'FAIL' '' "update --check still prints _RegexVersion (log $script:Log)"; return }
    if ($notice -notmatch '\d+\.\d+\.\d+') { Add-Result pip $scen 'FAIL' '' "update --check printed no version (log $script:Log)"; return }
    Add-Result pip $scen 'PASS' $New 'packaging blocked; JSON/regex path ok'
  } finally { Exit-Sandbox }
}

function Invoke-Row($inst, $scen) {
  if ($inst -ne 'pip' -and -not (Get-Command $inst -ErrorAction SilentlyContinue)) {
    Add-Result $inst $scen 'SKIP' '' "$inst not installed (try -BootstrapTools)"; return
  }
  if ($scen -eq 'autoupdate') {
    if ($AutoUpdateWheel) {
      Invoke-AutoUpdateLocal $inst
      if ($inst -eq 'pip') { Invoke-AutoUpdateFallback }
      return
    }
    $na = Get-AutoUpdateNA
    if ($na) { Add-Result $inst $scen 'N/A' '' $na; return }
  }
  Enter-Sandbox $inst $scen
  $sb = $script:SB
  try {
    switch ($inst) {
      'pip' {
        $py = "$sb/venv/$BinSub/python$Exe"; $bin = "$sb/venv/$BinSub/llm-speed$Exe"
        if (-not (X $Python -m venv "$sb/venv")) { Add-Result pip $scen 'FAIL' '' 'venv failed'; return }
        if ($scen -eq 'fresh') {
          if (-not (X $py -m pip install $SpecNew)) { Add-Result pip $scen 'FAIL' '' "pip install$Sfx failed (log $script:Log)"; return }
        } else {
          if (-not ((X $py -m pip install "$Pkg==$Prev") -and (Test-Prev $bin $py))) { Add-Result pip $scen 'FAIL' '' "could not install PREV $Prev"; return }
          $up = if ($scen -eq 'autoupdate') { X $bin update } else { X $py -m pip install -U $SpecNew }
          if (-not $up) { Add-Result pip $scen 'FAIL' '' "pip install -U$Sfx failed (log $script:Log)"; return }
        }
      }
      'pipx' {
        $py = "$sb/pipx/venvs/$Pkg/$BinSub/python$Exe"; $bin = "$sb/bin/llm-speed$Exe"
        if ($scen -eq 'fresh') {
          if (-not (X pipx install --python $Python $SpecNew)) { Add-Result pipx $scen 'FAIL' '' "pipx install$Sfx failed (log $script:Log)"; return }
        } else {
          if (-not ((X pipx install --python $Python "$Pkg==$Prev") -and (Test-Prev $bin $py))) { Add-Result pipx $scen 'FAIL' '' "could not install PREV $Prev"; return }
          # pipx upgrade cannot target a version; runpip is the pinned equivalent.
          $ok = if ($Pinned) { X pipx runpip $Pkg install -U "$Pkg==$New" } elseif ($scen -eq 'autoupdate') { X $bin update } else { X pipx upgrade $Pkg }
          if (-not $ok) { Add-Result pipx $scen 'FAIL' '' "pipx upgrade$Sfx failed (log $script:Log)"; return }
        }
      }
      'uv' {
        $py = "$sb/uvtools/$Pkg/$BinSub/python$Exe"; $bin = "$sb/bin/llm-speed$Exe"
        if ($scen -eq 'fresh') {
          if (-not (X uv tool install --python $Python $SpecNew)) { Add-Result uv $scen 'FAIL' '' "uv tool install$Sfx failed (log $script:Log)"; return }
        } else {
          # `uv tool install llm-speed==PREV` records the pin and `uv tool upgrade`
          # then refuses to move, so install unpinned as of just before NEW.
          $ok = if ($Pinned -or -not $NewUploaded) { X uv tool install --python $Python "$Pkg==$Prev" }
                else { X uv tool install --python $Python --exclude-newer $NewUploaded $Pkg }
          if (-not ($ok -and (Test-Prev $bin $py))) { Add-Result uv $scen 'FAIL' '' "could not install PREV $Prev"; return }
          $now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
          $ok = if ($Pinned -or -not $NewUploaded) { X uv tool install --python $Python --reinstall "$Pkg==$New" }
                elseif ($scen -eq 'autoupdate') { X $bin update }
                else { X uv tool upgrade --exclude-newer $now $Pkg }
          if (-not $ok) { Add-Result uv $scen 'FAIL' '' "uv tool upgrade$Sfx failed (log $script:Log)"; return }
        }
      }
    }
    Test-Installed $inst "$scen$Sfx" $bin $py
  } finally { Exit-Sandbox }
}

foreach ($inst in $Only) { foreach ($scen in $Scenarios) { Write-Host "== $inst $scen"; Invoke-Row $inst $scen } }

Write-Host ''
$Results | Format-Table -AutoSize Installer, Scenario, Status, Version, Detail | Out-String -Width 200 | Write-Host
$fails = @($Results | Where-Object { $_.Status -in 'FAIL', 'FAIL-soft' }).Count
$skips = @($Results | Where-Object { $_.Status -eq 'SKIP' }).Count
if ($DryRun) { $verdict = 'dry run: nothing was installed'; $rc = 0 }
elseif ($fails -gt 0) { $verdict = "installer gate: FAIL ($fails failing, $skips skipped)"; $rc = 1 }
elseif ($skips -gt 0) { $verdict = "installer gate: NOT DONE ($skips skipped; a SKIP is not a pass)"; $rc = 3 }
else { $verdict = 'installer gate: PASS'; $rc = 0 }
# Keep *.log on failure for upload-artifact; otherwise remove the run root.
if (-not $DryRun -and -not $Keep -and (Test-Path $Run)) {
  if ($fails -gt 0) {
    Get-ChildItem -Path $Run -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  } else {
    Remove-Item -Recurse -Force -Path $Run -ErrorAction SilentlyContinue
  }
}
$nas = @($Results | Where-Object { $_.Status -eq 'N/A' }).Count
if ($nas -gt 0) { $verdict = "$verdict; $nas N/A (not applicable to this pair, see Detail)" }
Write-Host $verdict
if ($Summary) {
  $md = @("#### $([System.Environment]::OSVersion.Platform) (pwsh $($PSVersionTable.PSVersion)): $Prev -> $New (PyPI latest: $Latest)", '',
    '| Installer | Scenario | Status | Version | Detail |', '| --- | --- | --- | --- | --- |')
  foreach ($r in $Results) {
    $v = if ($r.Version) { $r.Version } else { '-' }
    $md += "| $($r.Installer) | $($r.Scenario) | $($r.Status) | $v | $("$($r.Detail)" -replace '\|', '/') |"
  }
  $md += @('', "**$verdict**", '')
  Add-Content -Path $Summary -Value $md
}
exit $rc
