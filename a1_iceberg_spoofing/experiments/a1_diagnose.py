"""Why H1 (iceberg x spoofing interaction on spoofer surplus) was not supported: noise and channel diagnostics.

Usage::

    python -m experiments.a1_diagnose

Reads ``pilot1.pkl``, ``confirm1.pkl`` and ``pilot1_summary.json`` from the results directory
(``experiments/results_a1``, or ``$A1_RESULTS``) and writes ``diagnosis.md`` there.

Key identity. The spoofer's surplus is cash + pos * r_T. Fundamental paths are shared by all arms of a seed, so the
interaction Gamma(surplus) = Gamma(cash) + Gamma(pos) * r_T. The term Gamma(pos) * (r_T - mean r_T) has mean ~0 but
carries the terminal-value noise; removing it is a proxy for marking each fill to r~ at execution time.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

from experiments.common import RESULTS_DIR, core_arms, envs_of, fmt_ci, load, offsets_of, paired, significant, stat


def table(path, G: int = 10) -> List[str]:
    """Markdown rows of the noise / channel breakdown for every (env, offset) cell of one runner pickle."""
    d, T = load(path)
    rows = ["| env | offset | Gamma surplus (raw) | SD per seed | R^2 of terminal noise "
            "| Gamma surplus (noise removed) | SD | Gamma units sold | offset x Gamma sold | D_V sold | D_I sold |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in envs_of(d):
        for o in offsets_of(d):
            V, I, nV, nI = core_arms(G, o)
            gs = paired(T, e, I, nI, "sp_surplus", V, nV)
            gpos = paired(T, e, I, nI, "sp_position", V, nV)
            seeds = sorted(set.intersection(*[set(T[(e, x)]) for x in (I, nI, V, nV)]))
            rT = np.array([T[(e, V)][s]["final_fund"] for s in seeds])
            noise = gpos * (rT - rT.mean())
            r2 = np.corrcoef(gs, noise)[0, 1] ** 2 if gs.std() > 0 else np.nan
            resid = gs - noise
            gsold = stat(paired(T, e, I, nI, "sp_sold", V, nV))
            dv, di = stat(paired(T, e, V, nV, "sp_sold")), stat(paired(T, e, I, nI, "sp_sold"))
            rows.append(f"| {e} | {o} | {fmt_ci(stat(gs))} | {gs.std():.0f} | {r2:.2f} | {fmt_ci(stat(resid))} | "
                        f"{resid.std():.0f} | {fmt_ci(gsold, 3)} | {o * gsold['mean']:+.2f} | {fmt_ci(dv, 2)} | "
                        f"{fmt_ci(di, 2)} |")
    return rows


def main() -> None:
    out = ["# A1 diagnosis: why the interaction was not found", "",
           "Gamma = (spI - nsI) - (spV - nsV), seed-paired; bootstrap 95% CI; * = CI excludes 0. "
           "'Noise removed' subtracts Gamma(position) x (r_T - mean r_T), the part of the surplus contrast that comes "
           "only from the terminal fundamental (a proxy for valuing each fill at r~ when it trades).", ""]
    for name in ("pilot1", "confirm1"):
        out += [f"## {name}", ""] + table(RESULTS_DIR / f"{name}.pkl") + [""]
    _, Tp = load(RESULTS_DIR / "pilot1.pkl")
    _, Tc = load(RESULTS_DIR / "confirm1.pkl")
    out += ["## Pilot + confirmation pooled (C3, 800 seeds)", ""]
    for o in (1, 50):
        V, I, nV, nI = core_arms(10, o)
        args = ("C3", I, nI, "sp_surplus", V, nV)
        x = np.concatenate([paired(Tp, *args), paired(Tc, *args)])
        out.append(f"- offset {o}: Gamma surplus {fmt_ci(stat(x))}")
    with open(RESULTS_DIR / "pilot1_summary.json") as f:
        js = json.load(f)
    g = [v["Gamma"] for v in js.values() if np.isfinite(v["Gamma"]["mean"])]
    nsig = sum(1 for s in g if significant(s))
    out += ["", f"Pilot interaction tests: {len(g)} (env x offset x outcome); CIs excluding 0: {nsig} "
                f"(expected by chance at 5%: {0.05 * len(g):.1f})."]
    with open(RESULTS_DIR / "diagnosis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
