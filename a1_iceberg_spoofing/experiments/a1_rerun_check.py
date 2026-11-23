"""Checks the pre-specified claims of the fill-time-valuation rerun (A1_DESIGN.md, last section).

Usage::

    python -m experiments.a1_rerun_check

Reads ``{pilot1,confirm1}{,_nr}.pkl`` from the results directory (``experiments/results_a1``, or
``$A1_RESULTS``) and writes ``rerun_nr.md`` and ``rerun_nr_gamma.png`` there.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from experiments.common import RESULTS_DIR, core_arms, envs_of, fmt_ci, load, offsets_of, paired, significant, stat

G = 10
#: Outputs that existed before the rerun; the rerun must reproduce them exactly.
ORIG = ["sp_surplus", "sp_sold", "sp_bought", "sp_position", "hbl_mean", "zi_mean", "mm_surplus", "total_bg",
        "n_trades", "mid_dev", "vwap_dev_post", "final_fund", "diag_buy_dev", "diag_sell_dev"]


def same(a, b) -> bool:
    """Exact equality of two arrays/scalars, treating NaN as equal to NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def cells(d: dict) -> List[Tuple[str, int]]:
    """All (env, offset) cells of a runner pickle."""
    return [(e, o) for e in envs_of(d) for o in offsets_of(d)]


def main() -> None:
    out = ["# A1 rerun with fill-time valuation: pre-specified checks", "",
           "Seed-paired means [bootstrap 95% CI]; * = CI excludes 0. 'raw' values each trader's final position at the "
           "terminal fundamental r_T; 'fill-time' values each unit at r~ when it trades.", ""]
    sets = {}
    for name in ("pilot1", "confirm1"):
        _, T0 = load(RESULTS_DIR / f"{name}.pkl")
        d1, T1 = load(RESULTS_DIR / f"{name}_nr.pkl")
        sets[name] = (d1, T1)
        # claim 1: determinism
        n = bad = 0
        for key in T0:
            for s, r0 in T0[key].items():
                r1 = T1[key][s]
                n += 1
                if not (all(same(r0[k], r1[k]) for k in ORIG) and same(r0["mids"], r1["mids"])):
                    bad += 1
        out += [f"## {name}", "", f"**Claim 1 (determinism):** {n - bad} of {n} runs reproduce every original output "
                                  f"exactly ({bad} differ).", ""]

        # claims 2-4: Gamma raw vs fill-time
        out += ["| env | offset | Gamma raw | SD raw | Gamma fill-time | SD fill-time | SD ratio | raw - fill-time |",
                "|---|---|---|---|---|---|---|---|"]
        c2 = c2n = 0
        for e, o in cells(d1):
            V, I, nV, nI = core_arms(G, o)
            raw = paired(T1, e, I, nI, "sp_surplus", V, nV)
            nr = paired(T1, e, I, nI, "sp_surplus_nr", V, nV)
            diff = stat(raw - nr)
            if raw.std() > 0:
                c2n += 1
                c2 += significant(diff)
            ratio = raw.std() / nr.std() if nr.std() > 0 else np.nan
            out.append(f"| {e} | {o} | {fmt_ci(stat(raw))} | {raw.std():.0f} | {fmt_ci(stat(nr))} | {nr.std():.0f} | "
                       f"{ratio:.1f} | {fmt_ci(diff)} |")
        out += ["", f"**Claim 2 (same expectation):** raw - fill-time Gamma has a CI excluding 0 in {c2} of {c2n} "
                    "non-degenerate cells.", ""]

        # claim 5 and the display / spoofing effects on fill-time measures
        out += ["| env | offset | outcome (fill-time) | D_V = spV - nsV | D_I = spI - nsI | Gamma | nsI - nsV | "
                "spI - spV |",
                "|---|---|---|---|---|---|---|---|"]
        for e, o in cells(d1):
            V, I, nV, nI = core_arms(G, o)
            for k in ("sp_surplus_nr", "hbl_mean_nr", "zi_mean_nr", "total_bg_nr"):
                row = [stat(paired(T1, e, V, nV, k)), stat(paired(T1, e, I, nI, k)),
                       stat(paired(T1, e, I, nI, k, V, nV)), stat(paired(T1, e, nI, nV, k)),
                       stat(paired(T1, e, I, V, k))]
                out.append(f"| {e} | {o} | {k} | " + " | ".join(fmt_ci(s) for s in row) + " |")
        out += ["", "Hidden seller, no spoofing (nsI - nsV): fill-time surplus vs offset x extra units sold.", "",
                "| env | offset | surplus gain | extra units | offset x extra units |", "|---|---|---|---|---|"]
        for e, o in cells(d1):
            V, I, nV, nI = core_arms(G, o)
            g, u = stat(paired(T1, e, nI, nV, "sp_surplus_nr")), stat(paired(T1, e, nI, nV, "sp_sold"))
            out.append(f"| {e} | {o} | {fmt_ci(g)} | {fmt_ci(u, 2)} | {o * u['mean']:+.1f} |")
        out.append("")

        if name == "pilot1":
            out += ["Replication of Gu et al. (G = 1) on fill-time measures.", "",
                    "| env | offset | HBL: spV1 - base | ZI: spV1 - base | total bg: spV1 - base "
                    "| spoofer: spV1 - nsV1 |",
                    "|---|---|---|---|---|---|"]
            for e, o in cells(d1):
                a = f"spV1_o{o}"
                row = [stat(paired(T1, e, a, "base", k)) for k in ("hbl_mean_nr", "zi_mean_nr", "total_bg_nr")]
                row.append(stat(paired(T1, e, a, f"nsV1_o{o}", "sp_surplus_nr")))
                out.append(f"| {e} | {o} | " + " | ".join(fmt_ci(s) for s in row) + " |")
            out.append("")

    # claim 4: H1 re-test pooled
    out += ["## Claim 4 (H1 re-test): C3, Gamma of spoofer fill-time surplus", ""]
    for o in (1, 50):
        xs = []
        for name in ("pilot1", "confirm1"):
            _, T1 = sets[name]
            V, I, nV, nI = core_arms(G, o)
            x = paired(T1, "C3", I, nI, "sp_surplus_nr", V, nV)
            xs.append(x)
            out.append(f"- offset {o}, {name}: {fmt_ci(stat(x))}")
        out.append(f"- offset {o}, pooled 800 seeds: {fmt_ci(stat(np.concatenate(xs)))}")
    with open(RESULTS_DIR / "rerun_nr.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    figure(sets)


def figure(sets) -> None:
    """Forest plot of the interaction on spoofer surplus, raw vs fill-time valuation, per cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for name, (d1, T1) in sets.items():
        for e, o in cells(d1):
            V, I, nV, nI = core_arms(G, o)
            raw = paired(T1, e, I, nI, "sp_surplus", V, nV)
            if raw.std() == 0:
                continue
            rows.append((f"{name[:-1]} {e} o{o}", stat(raw), stat(paired(T1, e, I, nI, "sp_surplus_nr", V, nV))))
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for j, (lab, col) in enumerate((("raw (terminal value)", "#9aa0a6"), ("fill-time value", "#1a73e8"))):
        m = np.array([r[1 + j]["mean"] for r in rows])
        lo = np.array([r[1 + j]["lo"] for r in rows])
        hi = np.array([r[1 + j]["hi"] for r in rows])
        ax.errorbar(m, y + (0.15 if j == 0 else -0.15), xerr=[m - lo, hi - m], fmt="o", ms=3.5, capsize=2,
                    color=col, label=lab)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("interaction Gamma on spoofer surplus (95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rerun_nr_gamma.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
