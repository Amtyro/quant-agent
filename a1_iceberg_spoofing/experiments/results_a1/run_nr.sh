#!/usr/bin/env bash
# Rerun pilot and confirmation designs with the fill-time surplus fields.
set -e
PY="$1"
"$PY" -m experiments.a1_pilot --seeds 400 --seed0 100000 --envs A1,B2,C3 --G 10 --offsets 1,50 --workers 15 \
  --out experiments/results_a1/pilot1_nr.pkl > experiments/results_a1/pilot1_nr.log 2>&1
"$PY" -m experiments.a1_pilot --seeds 400 --seed0 200000 --envs A1,B2,C3 --G 10 --offsets 1,25,50,100 --core-only \
  --workers 15 --out experiments/results_a1/confirm1_nr.pkl > experiments/results_a1/confirm1_nr.log 2>&1
