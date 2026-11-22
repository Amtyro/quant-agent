# Commits any changes in this folder and pushes them to GitHub (Amtyro/quant-agent).
# Run by the Windows scheduled task "quant-agent GitHub sync" every day at 23:00.
# You can also run it by hand at any time.

$ErrorActionPreference = 'Continue'
$repo = $PSScriptRoot
$log  = Join-Path $repo 'sync.log'

function Write-Log($msg) {
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

git push origin main 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    # Remote has commits we don't have (e.g. edited on github.com): rebase, then retry.
    git pull --rebase origin main 2>&1 | Out-Null
    git push origin main 2>&1 | Out-Null
}

if ($LASTEXITCODE -eq 0) { Write-Log 'Push OK.' } else { Write-Log "Push FAILED (exit $LASTEXITCODE)." }
