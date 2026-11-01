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
  $v = & $py -c "import importlib.metadata as m; print(m.version('$Pkg'))" 2>$null
  if ($LASTEXITCODE -eq 0) { return "$v".Trim() } else { return '' }
}

$script:From = ''
function Test-Prev($bin, $py) {
  if ($DryRun) { return $true }
  $m = Get-MetaVersion $py
  Add-Content -Path $script:Log -Value "PREV installed: $m"
  $script:From = "$m -> "
  return (($m -eq $Prev) -and (Test-Path $bin))
}

function Test-Installed($inst, $scen, $bin, $py) {
  $from = $script:From; $script:From = ''
  if ($DryRun) {
    Write-Host "  [dry-run] $bin --version   (must print $Pkg $New)"
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
  if ($ver -eq $New) { Add-Result $inst $scen 'PASS' "$from$ver" ''; return }
  $meta = Get-MetaVersion $py
  if (-not $ver -and $meta -eq $New) { Add-Result $inst $scen 'FAIL-soft' "${from}metadata $meta" '--version missing or unparsable' }
  else { Add-Result $inst $scen 'FAIL' "$from$ver / metadata $meta" "want $New (log $script:Log)" }
}

function Invoke-Row($inst, $scen) {
  if ($inst -ne 'pip' -and -not (Get-Command $inst -ErrorAction SilentlyContinue)) {
    Add-Result $inst $scen 'SKIP' '' "$inst not installed (try -BootstrapTools)"; return
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
          if (-not (X $py -m pip install -U $SpecNew)) { Add-Result pip $scen 'FAIL' '' "pip install -U$Sfx failed (log $script:Log)"; return }
        }
      }
      'pipx' {
        $py = "$sb/pipx/venvs/$Pkg/$BinSub/python$Exe"; $bin = "$sb/bin/llm-speed$Exe"
        if ($scen -eq 'fresh') {
          if (-not (X pipx install --python $Python $SpecNew)) { Add-Result pipx $scen 'FAIL' '' "pipx install$Sfx failed (log $script:Log)"; return }
        } else {
          if (-not ((X pipx install --python $Python "$Pkg==$Prev") -and (Test-Prev $bin $py))) { Add-Result pipx $scen 'FAIL' '' "could not install PREV $Prev"; return }
          # pipx upgrade cannot target a version; runpip is the pinned equivalent.
          $ok = if ($Pinned) { X pipx runpip $Pkg install -U "$Pkg==$New" } else { X pipx upgrade $Pkg }
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
                else { X uv tool upgrade --exclude-newer $now $Pkg }
          if (-not $ok) { Add-Result uv $scen 'FAIL' '' "uv tool upgrade$Sfx failed (log $script:Log)"; return }
        }
      }
    }
    Test-Installed $inst "$scen$Sfx" $bin $py
  } finally { Exit-Sandbox }
}

foreach ($inst in $Only) { foreach ($scen in 'fresh', 'upgrade') { Write-Host "== $inst $scen"; Invoke-Row $inst $scen } }
if (-not $DryRun -and -not $Keep -and (Test-Path $Run)) { Remove-Item -Recurse -Force -Path $Run -ErrorAction SilentlyContinue }

Write-Host ''
$Results | Format-Table -AutoSize Installer, Scenario, Status, Version, Detail | Out-String -Width 200 | Write-Host
$fails = @($Results | Where-Object { $_.Status -in 'FAIL', 'FAIL-soft' }).Count
$skips = @($Results | Where-Object { $_.Status -eq 'SKIP' }).Count
if ($DryRun) { $verdict = 'dry run: nothing was installed'; $rc = 0 }
elseif ($fails -gt 0) { $verdict = "installer gate: FAIL ($fails failing, $skips skipped)"; $rc = 1 }
elseif ($skips -gt 0) { $verdict = "installer gate: NOT DONE ($skips skipped; a SKIP is not a pass)"; $rc = 3 }
else { $verdict = 'installer gate: PASS'; $rc = 0 }
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
