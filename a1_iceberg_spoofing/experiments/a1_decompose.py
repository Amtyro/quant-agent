"""Split the spoofer's fill-time surplus into its genuine sells and its (accidental) spoof-bid purchases.

Usage: python -m experiments.a1_decompose --env C3 --offset 50 --out experiments/results_a1/decompose_C3_o50.md
Re-simulates the four core arms on the pilot and confirmation seeds (same seeds as before, deterministic).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from experiments.a1sim import Config, ENVS, SELL, SPOOFER, Sim

ARMS = {"nsV": dict(spoof=False, iceberg=False), "spV": dict(spoof=True, iceberg=False),
        "nsI": dict(spoof=False, iceberg=True), "spI": dict(spoof=True, iceberg=True)}


def one(args):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="C3")
    ap.add_argument("--offset", type=int, default=50)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xi, om = ENVS[a.env]
    seeds = list(range(100000, 100400)) + list(range(200000, 200400))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for arm, kw in ARMS.items():
            cfg = replace(Config(mm_xi=xi, mm_omega=om), spoofer=True, sell_qty=10, sell_offset=a.offset, **kw)
            res[arm] = {s: x for s, *x in ex.map(one, [(cfg, s) for s in seeds], chunksize=8)}
    rng = np.random.default_rng(0)

    def ci(x):
        b = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return f"{x.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else "")

    def gam(j):
        return np.array([(res["spI"][s][j] - res["nsI"][s][j]) - (res["spV"][s][j] - res["nsV"][s][j]) for s in seeds])

    lines = [f"# Spoofer fill-time surplus, split by leg ({a.env}, offset {a.offset}, 800 seeds)", "",
             "| part | Gamma = (spI - nsI) - (spV - nsV) | mean in spV | mean in spI |", "|---|---|---|---|"]
    for j, lab in enumerate(["genuine sells", "spoof-bid purchases", "r~ drift on sells", "spoof units bought"]):
        lines.append(f"| {lab} | {ci(gam(j))} | {np.mean([res['spV'][s][j] for s in seeds]):+.2f} | "
                     f"{np.mean([res['spI'][s][j] for s in seeds]):+.2f} |")
    tot = gam(0) + gam(1)
    lines += ["", f"Total Gamma (sells + purchases): {ci(tot)}"]
    buys = [res[arm][s] for arm in ("spV", "spI") for s in seeds if res[arm][s][3] > 0]
    lines.append(f"Mean gain per spoof unit bought: {np.sum([b[1] for b in buys]) / np.sum([b[3] for b in buys]):.1f}")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
