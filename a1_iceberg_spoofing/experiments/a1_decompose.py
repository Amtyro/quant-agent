"""Split the spoofer's fill-time surplus into its genuine sells and its (accidental) spoof-bid purchases.

Usage::

    python -m experiments.a1_decompose --env C3 --offset 50 --out experiments/results_a1/decompose_C3_o50.md

Re-simulates the four core arms on the pilot and confirmation seeds (the same seeds as before; the simulator is
deterministic, so the runs are identical to the originals).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Tuple

import numpy as np

from experiments.a1sim import ENVS, SELL, SPOOFER, Config, Sim
from experiments.common import fmt_ci, stat

ARMS = {"nsV": dict(spoof=False, iceberg=False), "spV": dict(spoof=True, iceberg=False),
        "nsI": dict(spoof=False, iceberg=True), "spI": dict(spoof=True, iceberg=True)}
PILOT_SEEDS = range(100_000, 100_400)
CONFIRM_SEEDS = range(200_000, 200_400)


def one(args: Tuple[Config, int]) -> Tuple[int, float, float, float, int]:
    """(seed, genuine-sell surplus, spoof-bid surplus, r~ drift on sells, spoof units bought) for one run."""
    cfg, seed = args
    sim = Sim(cfg, seed)
    sim.run()
    sell = buy = drift = 0.0
    n_buy = 0
    for r in sim.ledger:
        if r.owner != SPOOFER:
            continue
        for (t, q, p) in r.fills:
            v = r.side * q * (float(sim.est[t]) - p)
            if r.side == SELL:
                sell += v
                drift += q * (float(sim.est[r.time]) - float(sim.est[t]))  # r~ change between posting and fill
            else:
                buy += v
                n_buy += q
    return seed, sell, buy, drift, n_buy


def main() -> None:
    ap = argparse.ArgumentParser(description="Split the spoofer's fill-time surplus by leg.")
    ap.add_argument("--env", default="C3", choices=sorted(ENVS))
    ap.add_argument("--offset", type=int, default=50)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", required=True, help="output markdown path")
    a = ap.parse_args()
    xi, om = ENVS[a.env]
    seeds = list(PILOT_SEEDS) + list(CONFIRM_SEEDS)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for arm, kw in ARMS.items():
            cfg = replace(Config(mm_xi=xi, mm_omega=om), spoofer=True, sell_qty=10, sell_offset=a.offset, **kw)
            res[arm] = {s: x for s, *x in ex.map(one, [(cfg, s) for s in seeds], chunksize=8)}
    rng = np.random.default_rng(0)

    def ci(x: np.ndarray) -> str:
        return fmt_ci(stat(x, rng=rng), 2)

    def gam(j: int) -> np.ndarray:
        return np.array([(res["spI"][s][j] - res["nsI"][s][j]) - (res["spV"][s][j] - res["nsV"][s][j]) for s in seeds])

    lines = [f"# Spoofer fill-time surplus, split by leg ({a.env}, offset {a.offset}, {len(seeds)} seeds)", "",
             "| part | Gamma = (spI - nsI) - (spV - nsV) | mean in spV | mean in spI |", "|---|---|---|---|"]
    for j, lab in enumerate(["genuine sells", "spoof-bid purchases", "r~ drift on sells", "spoof units bought"]):
        lines.append(f"| {lab} | {ci(gam(j))} | {np.mean([res['spV'][s][j] for s in seeds]):+.2f} | "
                     f"{np.mean([res['spI'][s][j] for s in seeds]):+.2f} |")
    tot = gam(0) + gam(1)
    lines += ["", f"Total Gamma (sells + purchases): {ci(tot)}"]
    buys = [res[arm][s] for arm in ("spV", "spI") for s in seeds if res[arm][s][3] > 0]
    lines.append(f"Mean gain per spoof unit bought: {np.sum([b[1] for b in buys]) / np.sum([b[3] for b in buys]):.1f}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
