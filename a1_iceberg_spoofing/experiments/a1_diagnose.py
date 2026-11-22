"""Why H1 (iceberg x spoofing interaction on spoofer surplus) was not supported: noise and channel diagnostics.

Usage (system Python): python experiments/a1_diagnose.py
Reads results_a1/pilot1.pkl and confirm1.pkl, writes results_a1/diagnosis.md.

Key identity. The spoofer's surplus is cash + pos * r_T. Fundamental paths are shared by all arms of a seed, so the
interaction Gamma(surplus) = Gamma(cash) + Gamma(pos) * r_T. The term Gamma(pos) * (r_T - mean r_T) has mean ~0 but
carries the terminal-value noise; removing it is a proxy for marking each fill to r~ at execution time.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.a1_analysis import load, paired, stat  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_a1")


def ci(s, d=1):
    return f"{s['mean']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]" + ("*" if s["lo"] > 0 or s["hi"] < 0 else "")


def table(path, G=10):
    d, T = load(path)
    envs = list(dict.fromkeys(k[0] for k in d["keys"]))
    offs = [int(x) for x in d["args"]["offsets"].split(",")]
    rows = ["| env | offset | Gamma surplus (raw) | SD per seed | R^2 of terminal noise | Gamma surplus (noise removed) | SD | "
            "Gamma units sold | offset x Gamma sold | D_V sold | D_I sold |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in envs:
        for o in offs:
            V, I, nV, nI = f"spV{G}_o{o}", f"spI{G}_o{o}", f"nsV{G}_o{o}", f"nsI{G}_o{o}"
            gs = paired(T, e, I, nI, "sp_surplus", V, nV)
            gpos = paired(T, e, I, nI, "sp_position", V, nV)
            seeds = sorted(set.intersection(*[set(T[(e, x)]) for x in (I, nI, V, nV)]))
            rT = np.array([T[(e, V)][s]["final_fund"] for s in seeds])
            noise = gpos * (rT - rT.mean())
            r2 = np.corrcoef(gs, noise)[0, 1] ** 2 if gs.std() > 0 else np.nan
            resid = gs - noise
            gsold = stat(paired(T, e, I, nI, "sp_sold", V, nV))
            dv, di = stat(paired(T, e, V, nV, "sp_sold")), stat(paired(T, e, I, nI, "sp_sold"))
            rows.append(f"| {e} | {o} | {ci(stat(gs))} | {gs.std():.0f} | {r2:.2f} | {ci(stat(resid))} | {resid.std():.0f} | "
                        f"{ci(gsold, 3)} | {o * gsold['mean']:+.2f} | {ci(dv, 2)} | {ci(di, 2)} |")
    return rows


def main():
    out = ["# A1 diagnosis: why the interaction was not found", "",
           "Gamma = (spI - nsI) - (spV - nsV), seed-paired; bootstrap 95% CI; * = CI excludes 0. "
           "'Noise removed' subtracts Gamma(position) x (r_T - mean r_T), the part of the surplus contrast that comes only "
           "from the terminal fundamental (a proxy for valuing each fill at r~ when it trades).", ""]
    for name in ("pilot1", "confirm1"):
        out += [f"## {name}", ""] + table(os.path.join(HERE, name + ".pkl")) + [""]
    dp, Tp = load(os.path.join(HERE, "pilot1.pkl"))
    dc, Tc = load(os.path.join(HERE, "confirm1.pkl"))
    out += ["## Pilot + confirmation pooled (C3, 800 seeds)", ""]
    for o in (1, 50):
        args = ("C3", f"spI10_o{o}", f"nsI10_o{o}", "sp_surplus", f"spV10_o{o}", f"nsV10_o{o}")
        x = np.concatenate([paired(Tp, *args), paired(Tc, *args)])
        out.append(f"- offset {o}: Gamma surplus {ci(stat(x))}")
    import json
    js = json.load(open(os.path.join(HERE, "pilot1_summary.json")))
    g = [v["Gamma"] for v in js.values() if np.isfinite(v["Gamma"]["mean"])]
    nsig = sum(1 for s in g if s["lo"] > 0 or s["hi"] < 0)
    out += ["", f"Pilot interaction tests: {len(g)} (env x offset x outcome); CIs excluding 0: {nsig} "
                f"(expected by chance at 5%: {0.05 * len(g):.1f})."]
    with open(os.path.join(HERE, "diagnosis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
