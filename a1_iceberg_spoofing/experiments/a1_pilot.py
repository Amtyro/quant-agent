"""A1 pilot runner: paired arms x liquidity environments x seeds.

Usage: python -m experiments.a1_pilot --seeds 400 --out experiments/results_a1/pilot1.pkl
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

from experiments.a1sim import Config, ENVS, run


def arms(G: int, offsets, core_only=False):
    out = {} if core_only else {"base": dict(spoofer=False)}
    for o in offsets:
        if not core_only:
            out[f"nsV1_o{o}"] = dict(spoofer=True, spoof=False, sell_qty=1, sell_offset=o)
            out[f"spV1_o{o}"] = dict(spoofer=True, spoof=True, sell_qty=1, sell_offset=o)
        for ice in (False, True):
            tag = "I" if ice else "V"
            out[f"ns{tag}{G}_o{o}"] = dict(spoofer=True, spoof=False, sell_qty=G, iceberg=ice, sell_offset=o)
            out[f"sp{tag}{G}_o{o}"] = dict(spoofer=True, spoof=True, sell_qty=G, iceberg=ice, sell_offset=o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=100_000)
    ap.add_argument("--envs", default="A1,B2,C3")
    ap.add_argument("--G", type=int, default=10)
    ap.add_argument("--offsets", default="1,50")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--core-only", action="store_true", help="only the G-unit V/I x spoof arms")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    offsets = [int(x) for x in a.offsets.split(",")]
    jobs, keys = [], []
    for env in a.envs.split(","):
        xi, om = ENVS[env]
        for name, kw in arms(a.G, offsets, a.core_only).items():
            cfg = replace(Config(mm_xi=xi, mm_omega=om), **kw)
            for s in range(a.seed0, a.seed0 + a.seeds):
                jobs.append((cfg, s))
                keys.append((env, name))
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)}  {el / 60:.1f} min  eta {(len(jobs) - i - 1) * el / (i + 1) / 60:.1f} min",
                      flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(keys=keys, results=results, args=vars(a),
                         configs={k: asdict(c) for k, c in zip(keys, (j[0] for j in jobs))}), f)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
