# A1: iceberg genuine leg × spoofing

Does hiding a spoofer's genuine sell order as an **iceberg** change how profitable and how harmful spoofing is against
heuristic belief learning (HBL) traders? This agent-based experiment builds on Gu et al. (ICAIF 2024, *The Effect of
Liquidity on the Spoofability of Financial Markets*) and Wang & Wellman (2017), using PyMarketSim's HBL pricing.

**Result in one line:** the pre-registered interaction H1 was not supported. The pilot's positive result was a false
positive driven by terminal-value noise, and the design makes the true interaction structurally small. See
[`experiments/A1_DIAGNOSIS.md`](experiments/A1_DIAGNOSIS.md) and the one-page brief in [`report/`](report/).

## Design at a glance

| | |
|---|---|
| Market | 12 ZI + 12 HBL background traders, ladder market maker, one strategic seller; T = 10,000 |
| Liquidity | Gu et al. environments A1 (most liquid), B2, C3 (least liquid) |
| Arms | spoof bid on/off × genuine leg of G = 10 units displayed (V) or iceberg with peak 1 (I), plus Gu's G = 1 arms |
| Estimand | Γ = (spI − nsI) − (spV − nsV), paired seed by seed, with bootstrap 95% CIs |
| Seeds | pilot 100000–100399, pre-specified confirmation 200000–200399 (400 per arm) |

Every source of randomness has its own seeded stream, so arms that share a seed differ only through the seller's
behaviour, and every run is exactly repeatable on the same machine (see the reproducibility note below). Full design and pre-specified claims:
[`experiments/A1_DESIGN.md`](experiments/A1_DESIGN.md). A plain-language walkthrough in Chinese:
[`experiments/A1_EXPLAINER_zh.md`](experiments/A1_EXPLAINER_zh.md).

## Layout

```
a1_iceberg_spoofing/
├── experiments/
│   ├── a1sim.py            simulator: price-time order book, synthetic iceberg, ledger-based HBL, MM, spoofer
│   ├── a1_pilot.py         parallel runner: arms × environments × seeds -> one pickle
│   ├── common.py           shared analysis helpers: loading, seed-paired contrasts, bootstrap CIs, formatting
│   ├── a1_analysis.py      summary tables (md/json) and figures for one runner pickle
│   ├── a1_diagnose.py      why H1 failed: terminal-value noise and channel breakdown
│   ├── a1_rerun_check.py   checks the pre-specified claims of the fill-time-valuation rerun
│   ├── a1_decompose.py     splits the seller's fill-time surplus into genuine sells and spoof-bid purchases
│   ├── test_a1sim.py       correctness checks (FIFO, level pricing, conservation, iceberg, belief = upstream, pairing)
│   ├── test_regression.py  golden-output regression test (fixture: golden_a1sim.json)
│   ├── A1_DESIGN.md · A1_DIAGNOSIS.md · A1_EXPLAINER_zh.md
│   └── results_a1/         summaries, logs and figures of the committed runs
├── report/                 research brief (HTML / PDF)
├── scripts/run_nr.sh       reruns pilot + confirmation with fill-time valuation
├── pyproject.toml          pytest and ruff configuration
└── requirements.txt        pinned environment that produced the results
```

The raw `*.pkl` result files (about 135 MB) are not committed; the commands below re-create them (exactly, on the machine that produced them).

## Setup

The simulator imports PyMarketSim's `marketsim` package unmodified (`HBLAgent`, constants and private values).
PyMarketSim is not on PyPI, so clone it at the commit the results were produced with and put it on `PYTHONPATH`:

```bash
git clone https://github.com/dipplestix/pymarketsim.git && git -C pymarketsim checkout c300eec
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/pymarketsim                        # Windows: set PYTHONPATH=<path>\pymarketsim
```

Run everything from this folder (`a1_iceberg_spoofing/`).

## Reproduce

```bash
python -m pytest                                          # 11 tests, about 15 s

# simulations (the confirmation run is 19,200 markets: about 56 min on 15 workers)
python -m experiments.a1_pilot --seeds 400 --seed0 100000 --out experiments/results_a1/pilot1.pkl
python -m experiments.a1_pilot --seeds 400 --seed0 200000 --offsets 1,25,50,100 --core-only \
    --out experiments/results_a1/confirm1.pkl
scripts/run_nr.sh                                         # fill-time-valuation reruns (*_nr.pkl)

# analysis (needs only numpy and matplotlib)
python -m experiments.a1_analysis experiments/results_a1/confirm1.pkl
python -m experiments.a1_diagnose
python -m experiments.a1_rerun_check
python -m experiments.a1_decompose --env C3 --offset 50 --out experiments/results_a1/decompose_C3_o50.md
```

The analysis scripts read from `experiments/results_a1/` by default; set `A1_RESULTS` to use another folder.

## Development

```bash
ruff check .                                              # lint (configured in pyproject.toml)
A1_UPDATE_GOLDEN=1 python -m pytest experiments/test_regression.py   # only when a change is meant to alter outputs
```

Refactors must leave `test_regression.py` passing: the committed results depend on the simulator being bit-for-bit
deterministic.

**Reproducibility note.** Runs are bit-for-bit repeatable on the same machine, but not across machines: PyMarketSim's
HBL price optimiser uses compiled numerical code whose floating-point results differ slightly between operating
systems and CPUs, and the differences compound over a market. On another machine, expect the same conclusions with
slightly different numbers in the tables. The committed results were produced on Windows with the pinned
environment above, so the golden regression test is pinned there only and skips elsewhere (including CI, which runs
lint and the correctness tests).
