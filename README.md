# quant agent

[![CI](https://github.com/Amtyro/quant-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Amtyro/quant-agent/actions/workflows/ci.yml)

Research code on market microstructure and manipulation, studied with agent-based limit-order-book simulations.

## Projects

| Folder | Question | Status |
|---|---|---|
| [`a1_iceberg_spoofing/`](a1_iceberg_spoofing/) | Does hiding a spoofer's genuine order as an iceberg change the profitability and harm of spoofing against learning traders? | Pilot + pre-specified confirmation done; H1 not supported ([diagnosis](a1_iceberg_spoofing/experiments/A1_DIAGNOSIS.md)) |

Each project folder is self-contained, with its own README covering setup, how to reproduce the results, and tests.

## Repository layout

```
.
├── a1_iceberg_spoofing/    A1 experiment: simulator, runners, analysis, results, report
├── tools/auto-sync.ps1     daily commit-and-push to GitHub (Windows scheduled task)
└── .github/workflows/      CI: lint and tests on every push
```

## Conventions

- Every simulation is seeded and deterministic; committed results can be regenerated exactly from the commands in
  each project's README.
- Large raw outputs (`*.pkl`, `data/`) and secrets (`.env`, keys) are git-ignored and never pushed.
- Code is linted with [ruff](https://docs.astral.sh/ruff/) and tested with pytest in CI.

## Auto-sync

The Windows scheduled task `quant-agent GitHub sync` runs [`tools/auto-sync.ps1`](tools/auto-sync.ps1) every day at
23:00: it commits any changes and pushes them to this repository, logging to `sync.log` (not committed). To sync now:

```powershell
powershell -ExecutionPolicy Bypass -File tools\auto-sync.ps1
```
