# quant agent

Code for the quant agent project.

## Auto-sync

Everything in this folder is committed and pushed to GitHub automatically every day at 23:00
by the Windows scheduled task `quant-agent GitHub sync`, which runs `auto-sync.ps1`.

- Run a sync now: `powershell -ExecutionPolicy Bypass -File auto-sync.ps1`
- Results are logged to `sync.log` (not pushed).
- Files listed in `.gitignore` (secrets, `.env`, data files, `data/`) are never pushed.
