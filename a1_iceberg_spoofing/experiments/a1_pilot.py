"""A1 runner: paired arms x liquidity environments x seeds, in parallel.

Every arm of an environment is run on the same seeds, so arms can be compared seed by seed.

Usage::

    python -m experiments.a1_pilot --seeds 400 --seed0 100000 --out experiments/results_a1/pilot1.pkl
    python -m experiments.a1_pilot --seeds 400 --seed0 200000 --offsets 1,25,50,100 --core-only \\
        --out experiments/results_a1/confirm1.pkl

The output pickle holds ``keys`` [(env, arm)], ``results`` [per-run summary dict, see ``Sim._summary``],
``args`` (the command-line arguments) and ``configs`` {(env, arm): Config as a dict}.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Tuple

from experiments.a1sim import ENVS, Config, run

log = logging.getLogger(__name__)


def arms(G: int, offsets: Iterable[int], core_only: bool = False) -> Dict[str, dict]:
    """Arm name -> Config overrides.

    Names: ``ns``/``sp`` = without/with the spoof bid; ``V``/``I`` = visible/iceberg genuine leg; then the
    leg size G and ``_o<offset>``. Unless ``core_only``, also the no-seller ``base`` arm and the G=1
    visible arms that replicate Gu et al.
    """
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


def build_jobs(envs: Iterable[str], G: int, offsets: List[int], core_only: bool, seed0: int,
               n_seeds: int) -> Tuple[List[Tuple[Config, int]], List[Tuple[str, str]]]:
    """All (config, seed) jobs and their (env, arm) keys, in a fixed order."""
    jobs, keys = [], []
    for env in envs:
        xi, om = ENVS[env]
        for name, kw in arms(G, offsets, core_only).items():
            cfg = replace(Config(mm_xi=xi, mm_omega=om), **kw)
            for s in range(seed0, seed0 + n_seeds):
                jobs.append((cfg, s))
                keys.append((env, name))
    return jobs, keys


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=400, help="seeds per arm (default 400)")
    ap.add_argument("--seed0", type=int, default=100_000, help="first seed (default 100000)")
    ap.add_argument("--envs", default="A1,B2,C3", help=f"comma-separated, from {','.join(ENVS)}")
    ap.add_argument("--G", type=int, default=10, help="genuine-leg size (default 10)")
    ap.add_argument("--offsets", default="1,50", help="comma-separated sell offsets above r~ (default 1,50)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes (default: CPU count - 1)")
    ap.add_argument("--core-only", action="store_true", help="only the G-unit V/I x spoof arms")
    ap.add_argument("--out", required=True, help="output pickle path")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args(argv)
    offsets = [int(x) for x in a.offsets.split(",")]
    jobs, keys = build_jobs(a.envs.split(","), a.G, offsets, a.core_only, a.seed0, a.seeds)
    log.info("%d runs on %d workers", len(jobs), a.workers)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                eta = (len(jobs) - i - 1) * el / (i + 1)
                log.info("%d/%d  %.1f min  eta %.1f min", i + 1, len(jobs), el / 60, eta / 60)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(keys=keys, results=results, args=vars(a),
                         configs={k: asdict(c) for k, c in zip(keys, (j[0] for j in jobs), strict=True)}), f)
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, a.out)


if __name__ == "__main__":
    main()
