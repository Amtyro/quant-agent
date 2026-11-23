"""Shared helpers for the A1 analysis scripts: loading result pickles, seed-paired contrasts,
bootstrap confidence intervals and table formatting.

All bootstrap draws come from the module-level ``RNG`` (seed 0) unless a generator is passed in, so a
script that makes the same calls in the same order always prints the same intervals.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

RNG = np.random.default_rng(0)

#: Default location of the result pickles; override with the ``A1_RESULTS`` environment variable.
RESULTS_DIR = Path(os.environ.get("A1_RESULTS", Path(__file__).resolve().parent / "results_a1"))

#: (env, arm) -> seed -> per-run result dict
Table = Dict[Tuple[str, str], Dict[int, dict]]


def load(path) -> Tuple[dict, Table]:
    """Load a runner pickle and index its results by (env, arm) and seed."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    table: Table = defaultdict(dict)
    for (env, arm), r in zip(d["keys"], d["results"], strict=True):
        table[(env, arm)][r["seed"]] = r
    return d, table


def envs_of(d: dict) -> list:
    """Environments in a runner pickle, in run order."""
    return list(dict.fromkeys(k[0] for k in d["keys"]))


def offsets_of(d: dict) -> list:
    """Sell offsets recorded in a runner pickle's arguments."""
    return [int(x) for x in d["args"]["offsets"].split(",")]


def core_arms(G: int, o) -> Tuple[str, str, str, str]:
    """Names of the four core arms (spoof visible, spoof iceberg, no-spoof visible, no-spoof iceberg)."""
    return f"spV{G}_o{o}", f"spI{G}_o{o}", f"nsV{G}_o{o}", f"nsI{G}_o{o}"


def paired(table: Table, env: str, a: str, b: str, key: str,
           c: Optional[str] = None, d: Optional[str] = None) -> np.ndarray:
    """Seed-level differences ``a - b`` (or ``(a - b) - (c - d)`` if ``c, d`` are given), on the seeds
    present in every arm and with a finite value of ``key`` in all of them."""
    arms = [a, b] + ([c, d] if c else [])
    seeds = set.intersection(*[set(table[(env, x)]) for x in arms])
    vals = []
    for s in sorted(seeds):
        v = [table[(env, x)][s][key] for x in arms]
        if all(np.isfinite(v)):
            vals.append(v[0] - v[1] if not c else (v[0] - v[1]) - (v[2] - v[3]))
    return np.asarray(vals, dtype=float)


def stat(x: np.ndarray, B: int = 2000, rng: Optional[np.random.Generator] = None) -> dict:
    """Mean, standard error and percentile-bootstrap 95% CI of ``x``; ``p_pos`` is the bootstrap
    share of positive means."""
    n = len(x)
    if n < 2:
        return dict(n=n, mean=np.nan, se=np.nan, lo=np.nan, hi=np.nan, p_pos=np.nan)
    rng = RNG if rng is None else rng
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    boot = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, mean=m, se=se, lo=float(lo), hi=float(hi), p_pos=float((boot > 0).mean()))


def significant(s: dict) -> bool:
    """True if the 95% CI excludes zero."""
    return s["lo"] > 0 or s["hi"] < 0


def fmt_ci(s: dict, d: int = 1) -> str:
    """``mean [lo, hi]`` with ``d`` decimals and a trailing ``*`` when the CI excludes zero."""
    if not np.isfinite(s["mean"]):
        return "n/a"
    return f"{s['mean']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]" + ("*" if significant(s) else "")


def fmt_mean(s: dict, d: int = 2) -> str:
    """``mean`` with ``d`` decimals and a trailing ``*`` when the CI excludes zero."""
    if not np.isfinite(s["mean"]):
        return "n/a"
    return f"{s['mean']:+.{d}f}" + ("*" if significant(s) else "")
