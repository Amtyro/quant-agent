#!/usr/bin/env bash
# Rerun the pilot and confirmation designs with the fill-time surplus fields (about 2 x 1 hour on 15 workers).
#
# Usage: scripts/run_nr.sh [python]      (default: python; PyMarketSim must be on PYTHONPATH)
# Writes experiments/results_a1/{pilot1,confirm1}_nr.{pkl,log}.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
OUT=experiments/results_a1

"$PY" -m experiments.a1_pilot --seeds 400 --seed0 100000 --envs A1,B2,C3 --G 10 --offsets 1,50 --workers 15 \
  --out "$OUT/pilot1_nr.pkl" > "$OUT/pilot1_nr.log" 2>&1
"$PY" -m experiments.a1_pilot --seeds 400 --seed0 200000 --envs A1,B2,C3 --G 10 --offsets 1,25,50,100 --core-only \
  --workers 15 --out "$OUT/confirm1_nr.pkl" > "$OUT/confirm1_nr.log" 2>&1
