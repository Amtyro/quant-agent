<#
.SYNOPSIS
    Commit any changes in this repository and push them to GitHub (Amtyro/quant-agent).

.DESCRIPTION
    Run daily at 23:00 by the Windows scheduled task "quant-agent GitHub sync"; can also be run by hand.
    Files matched by .gitignore are never committed. Results are appended to sync.log at the repository root.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\auto-sync.ps1
#>

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$log  = Join-Path $repo 'sync.log'

function Write-Log([string]$msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -FilePath $log -Append -Encoding utf8
}

Set-Location -LiteralPath $repo

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-Null
    Write-Log 'Committed local changes.'
} else {
    Write-Log 'No local changes.'
}

# Never wait for a login prompt in the background, and abort stalled connections
# (under 1 KB/s for 20 s) so a flaky network just means "retry", not a hang.
$env:GCM_INTERACTIVE     = 'never'
$env:GIT_TERMINAL_PROMPT = '0'
$net = @('-c', 'http.lowSpeedLimit=1000', '-c', 'http.lowSpeedTime=20')

$ok = $false
for ($i = 1; $i -le 3 -and -not $ok; $i++) {
    git @net push origin main 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        # Remote may have commits we don't have (e.g. edited on github.com): rebase, then retry.
        git @net pull --rebase origin main 2>&1 | Out-Null
        git @net push origin main 2>&1 | Out-Null
    }
    if ($LASTEXITCODE -eq 0) {
        $ok = $true
    } else {
        Write-Log "Push attempt $i failed (exit $LASTEXITCODE)."
        Start-Sleep -Seconds 30
    }
}

if ($ok) { Write-Log 'Push OK.' } else { Write-Log 'Push FAILED after 3 attempts; will retry at next run.' }
