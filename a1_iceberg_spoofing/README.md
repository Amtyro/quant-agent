# A1: iceberg genuine leg x spoofing (PyMarketSim pilot)

This experiment asks whether hiding a spoofer's genuine sell order as an iceberg changes how well spoofing works
against heuristic belief learning (HBL) traders. It builds on Gu et al. (ICAIF 2024) and Wang & Wellman (2017).

## Contents
| Path | What it is |
|---|---|
| `experiments/a1sim.py` | Simulator: price-time order book, synthetic iceberg, HBL that learns from a public ledger, ladder market maker, spoofer |
| `experiments/test_a1sim.py` | Correctness checks (FIFO, level pricing, conservation, iceberg invariants, belief equal to upstream, paired arms) |
| `experiments/a1_pilot.py` | Runs the paired arms x liquidity environments x seeds |
| `experiments/a1_analysis.py` | Seed-paired effects with bootstrap CIs; writes `*_summary.md/.json` and figures |
| `experiments/a1_diagnose.py` | Breaks down the noise and channels behind the null interaction; writes `results_a1/diagnosis.md` |
| `experiments/A1_DESIGN.md` | Design, departures from upstream, and the claims fixed before the confirmation run |
| `experiments/A1_DIAGNOSIS.md` | Why H1 was not supported, and the rerun vs EGTA comparison |
| `experiments/results_a1/` | Summaries, logs and figures for the pilot (seeds 100000-100399) and confirmation (seeds 200000-200399) runs |
| `report/` | One-page research brief (HTML) |

The raw `*.pkl` result files (about 65 MB) are not committed. Re-create them with the commands below.

## Setup
The code imports PyMarketSim's `marketsim` package and uses it unmodified (`HBLAgent`, constants and private values).
Put a PyMarketSim checkout on `PYTHONPATH`, then run the commands from this folder:

```
pip install numpy scipy matplotlib pytest torch   # torch is imported by marketsim
set PYTHONPATH=<path to PyMarketSim>;.
python -m pytest experiments/test_a1sim.py -q
python -m experiments.a1_pilot --seeds 400 --seed0 100000 --out experiments/results_a1/pilot1.pkl
python -m experiments.a1_pilot --seeds 400 --seed0 200000 --offsets 1,25,50,100 --core-only --out experiments/results_a1/confirm1.pkl
python experiments/a1_analysis.py experiments/results_a1/confirm1.pkl
python experiments/a1_diagnose.py
```

The confirmation run is 19,200 simulations and took about 56 minutes on 15 worker processes.
