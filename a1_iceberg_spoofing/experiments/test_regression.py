"""Golden-output regression test for the A1 simulator.

The simulator is deterministic given (config, seed), and the committed results depend on that. This test pins the
summary of a few short runs, so any change that alters simulated markets fails loudly.

Exact values are machine-specific: PyMarketSim's HBL price optimiser relies on compiled numerical code whose
floating-point results differ slightly between operating systems and CPUs, and the differences compound over a run
(on GitHub's Linux runners they even differ from one runner to the next). Runs are bit-for-bit repeatable on the
same machine. The fixture therefore holds values per ``sys.platform`` and is only populated for Windows (``win32``),
where the committed results in ``results_a1/`` were produced; other platforms skip this test.

To (re)generate the entry for the current platform, e.g. when a change is meant to alter outputs:

    A1_UPDATE_GOLDEN=1 python -m pytest experiments/test_regression.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.a1sim import ENVS, Config, Sim

GOLDEN = Path(os.environ.get("A1_GOLDEN", Path(__file__).with_name("golden_a1sim.json")))
PLATFORM = sys.platform

CASES = {
    "B2_base": ("B2", dict(spoofer=False), 7),
    "A1_spoof_G1_o1": ("A1", dict(spoofer=True, spoof=True, sell_qty=1, sell_offset=1), 11),
    "C3_nospoof_vis_G10_o50": ("C3", dict(spoofer=True, spoof=False, sell_qty=10, sell_offset=50), 13),
    "C3_spoof_ice_G10_o50": ("C3", dict(spoofer=True, spoof=True, sell_qty=10, iceberg=True, sell_offset=50), 13),
}


def _summarise(env: str, overrides: dict, seed: int) -> dict:
    xi, om = ENVS[env]
    out = Sim(replace(Config(mm_xi=xi, mm_omega=om, T=4000), **overrides), seed).run()
    mids = out.pop("mids")
    out["mids_len"] = int(len(mids))
    out["mids_nansum"] = float(np.nansum(mids))
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in out.items()}


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name):
    got = _summarise(*CASES[name])
    if os.environ.get("A1_UPDATE_GOLDEN"):
        data = _load()
        data.setdefault(PLATFORM, {})[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"golden output for {PLATFORM} updated")
    golden = _load().get(PLATFORM)
    if golden is None:
        pytest.skip(f"no golden outputs for platform {PLATFORM!r}; generate them with A1_UPDATE_GOLDEN=1")
    want = golden[name]
    assert got.keys() == want.keys()
    diffs = {k: (want[k], got[k]) for k in want if not _close(want[k], got[k])}
    assert not diffs, f"outputs changed (expected, got): {diffs}"
