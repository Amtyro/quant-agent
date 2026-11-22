"""Paired, seed-level analysis of the A1 pilot.

Usage: python -m experiments.a1_analysis experiments/results_a1/pilot1.pkl
Writes <stem>_summary.md, <stem>_summary.json and figures next to the pickle.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

OUTCOMES = ["sp_surplus", "hbl_mean", "zi_mean", "mm_surplus", "total_bg", "mid_dev", "vwap_dev_post",
            "sp_sold", "sp_bought", "n_trades"]
LABELS = {"sp_surplus": "spoofer surplus", "hbl_mean": "HBL surplus (per agent)",
          "zi_mean": "ZI surplus (per agent)", "mm_surplus": "MM surplus", "total_bg": "total background surplus",
          "mid_dev": "midprice - r~ (post t0)", "vwap_dev_post": "VWAP - r~ (post t0)",
          "sp_sold": "units sold by spoofer", "sp_bought": "spoof units executed", "n_trades": "units traded"}
RNG = np.random.default_rng(0)


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    table = defaultdict(dict)  # (env, arm) -> seed -> result
    for (env, arm), r in zip(d["keys"], d["results"]):
        table[(env, arm)][r["seed"]] = r
    return d, table


def paired(table, env, a, b, key, c=None, d=None):
    """Seed-level differences a-b (or (a-b)-(c-d) if c,d given) on seeds present and finite in all arms."""
    arms = [a, b] + ([c, d] if c else [])
    seeds = set.intersection(*[set(table[(env, x)]) for x in arms])
    vals = []
    for s in sorted(seeds):
        v = [table[(env, x)][s][key] for x in arms]
        if all(np.isfinite(v)):
            vals.append(v[0] - v[1] if not c else (v[0] - v[1]) - (v[2] - v[3]))
    return np.asarray(vals, dtype=float)


def stat(x, B=2000):
    n = len(x)
    if n < 2:
        return dict(n=n, mean=np.nan, se=np.nan, lo=np.nan, hi=np.nan, p_pos=np.nan)
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    boot = x[RNG.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, mean=m, se=se, lo=float(lo), hi=float(hi), p_pos=float((boot > 0).mean()))


def fmt(s):
    if not np.isfinite(s["mean"]):
        return "n/a"
    star = "*" if (s["lo"] > 0 or s["hi"] < 0) else ""
    return f"{s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}]{star}"


def fmt2(s):
    if not np.isfinite(s["mean"]):
        return "n/a"
    star = "*" if (s["lo"] > 0 or s["hi"] < 0) else ""
    return f"{s['mean']:+.2f}{star}"


def main(path):
    d, table = load(path)
    envs = list(dict.fromkeys(k[0] for k in d["keys"]))
    G = d["args"]["G"]
    offsets = [int(x) for x in d["args"]["offsets"].split(",")]
    stem = os.path.splitext(path)[0]
    lines = [f"# A1 pilot summary ({os.path.basename(path)})", "",
             f"Seeds per arm: {d['args']['seeds']}; genuine-leg size G={G}; offsets {offsets}. "
             "Entries: seed-paired mean difference [bootstrap 95% CI]; * = CI excludes 0.", ""]
    js = {}

    # 0. levels
    lines += ["## Levels (mean over seeds)", "",
              "| env | arm | spoofer surplus | units sold | spoof units hit | HBL surplus | ZI surplus | MM surplus | mid-r~ | trades |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for env in envs:
        for arm in dict.fromkeys(k[1] for k in d["keys"] if k[0] == env):
            rs = list(table[(env, arm)].values())
            g = lambda k: np.nanmean([r[k] for r in rs])
            lines.append(f"| {env} | {arm} | {g('sp_surplus'):.1f} | {g('sp_sold'):.2f} | {g('sp_bought'):.2f} | "
                         f"{g('hbl_mean'):.1f} | {g('zi_mean'):.1f} | {g('mm_surplus'):.1f} | {g('mid_dev'):.1f} | {g('n_trades'):.1f} |")
    lines.append("")

    for o in offsets:
        # 1. replication of Gu et al.: spoofing vs no spoofer, G=1
        lines += [f"## Offset {o}: replication of Gu et al. (G=1)", "",
                  "| env | HBL: spV1 - base | ZI: spV1 - base | mid: spV1 - base | spoofer: spV1 - nsV1 (value of spoofing) |",
                  "|---|---|---|---|---|"]
        for env in envs:
            a = f"spV1_o{o}"
            row = [fmt(stat(paired(table, env, a, "base", k))) for k in ("hbl_mean", "zi_mean", "mid_dev")]
            row.append(fmt(stat(paired(table, env, a, f"nsV1_o{o}", "sp_surplus"))))
            lines.append(f"| {env} | " + " | ".join(row) + " |")
        lines.append("")

        # 2. A1 core: interaction
        V, I, nV, nI = f"spV{G}_o{o}", f"spI{G}_o{o}", f"nsV{G}_o{o}", f"nsI{G}_o{o}"
        lines += [f"## Offset {o}: spoofing x display (G={G})", "",
                  "Effect of spoofing with a visible genuine leg D_V = spV - nsV, with an iceberg leg D_I = spI - nsI, "
                  "and the interaction Gamma = D_I - D_V. Display effects: spI - spV (with spoofing), nsI - nsV (without).", "",
                  "| env | outcome | D_V | D_I | Gamma | spI - spV | nsI - nsV |", "|---|---|---|---|---|---|---|"]
        for env in envs:
            for k in OUTCOMES:
                s = {
                    "D_V": stat(paired(table, env, V, nV, k)),
                    "D_I": stat(paired(table, env, I, nI, k)),
                    "Gamma": stat(paired(table, env, I, nI, k, V, nV)),
                    "spI-spV": stat(paired(table, env, I, V, k)),
                    "nsI-nsV": stat(paired(table, env, nI, nV, k)),
                }
                js[f"{env}|o{o}|{k}"] = s
                lines.append(f"| {env} | {LABELS[k]} | " + " | ".join(fmt(v) for v in s.values()) + " |")
        lines.append("")

        # 2b. who trades with the seller, and who else sells (counterparty composition)
        cp = ["sp_to_HBL", "sp_to_ZI", "sp_to_MM", "sp_other_sell_HBL", "sp_other_sell_ZI", "sp_other_sell_MM"]
        if all(k in next(iter(table[(envs[0], V)].values())) for k in cp):
            lines += [f"## Offset {o}: counterparties (units per market, post t0)", "",
                      "Seller's units bought by each kind, and units sold by other kinds; last columns: paired I - V.", "",
                      "| env | arm | to HBL | to ZI | to MM | HBL sells | ZI sells | MM sells |", "|---|---|---|---|---|---|---|---|"]
            for env in envs:
                for arm in (nV, nI, V, I):
                    rs = list(table[(env, arm)].values())
                    lines.append(f"| {env} | {arm} | " + " | ".join(f"{np.mean([r[k] for r in rs]):.2f}" for k in cp) + " |")
                for (x, y, lab) in ((nI, nV, "nsI - nsV"), (I, V, "spI - spV")):
                    lines.append(f"| {env} | {lab} | " + " | ".join(fmt2(stat(paired(table, env, x, y, k))) for k in cp) + " |")
            lines.append("")

        # 3. mechanism: what HBL sees
        lines += [f"## Offset {o}: mechanism diagnostics (HBL decisions after t0)", "",
                  "| env | arm | spoofer share of ask qty in memory | spoofer share of bid qty | HBL buy quote - r~ | HBL sell quote - r~ |",
                  "|---|---|---|---|---|---|"]
        for env in envs:
            for arm in ("base", nV, nI, V, I):
                rs = list(table[(env, arm)].values())
                g = lambda k: np.nanmean([r[k] for r in rs])
                lines.append(f"| {env} | {arm} | {g('diag_sp_ask_share'):.3f} | {g('diag_sp_bid_share'):.3f} | "
                             f"{g('diag_buy_dev'):.1f} | {g('diag_sell_dev'):.1f} |")
        lines.append("")

    with open(stem + "_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(stem + "_summary.json", "w") as f:
        json.dump(js, f, indent=1)
    figures(d, table, envs, G, offsets, stem)
    print("\n".join(lines))


def figures(d, table, envs, G, offsets, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.arange(1000, 10001, 50)
    has_base = all(table.get((e, "base")) for e in envs)
    for o in offsets:
        if has_base:
            fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), sharey=False)
            axes = np.atleast_1d(axes)
            for ax, env in zip(axes, envs):
                seeds = sorted(table[(env, "base")])
                base = np.array([table[(env, "base")][s]["mids"] for s in seeds])
                for arm, lab, st in ((f"spV1_o{o}", "spoof, G=1 (Gu)", ":"), (f"spV{G}_o{o}", f"spoof, visible G={G}", "-"),
                                     (f"spI{G}_o{o}", f"spoof, iceberg G={G}", "--"), (f"nsV{G}_o{o}", f"no spoof, visible G={G}", "-.")):
                    if (env, arm) not in table:
                        continue
                    m = np.array([table[(env, arm)][s]["mids"] for s in seeds])
                    diff = np.nanmean(m - base, axis=0)
                    ax.plot(grid[: len(diff)], diff, st, label=lab)
                ax.axhline(0, color="grey", lw=0.6)
                ax.set_title(f"{env}: midprice vs no-spoofer twin")
                ax.set_xlabel("time")
            axes[0].set_ylabel("mean paired midprice difference")
            axes[-1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(f"{stem}_mid_o{o}.png", dpi=140)
            plt.close(fig)

        # interaction bars
        keys = ["sp_surplus", "hbl_mean", "zi_mean", "mid_dev"]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2))
        for ax, k in zip(axes, keys):
            V, I, nV, nI = f"spV{G}_o{o}", f"spI{G}_o{o}", f"nsV{G}_o{o}", f"nsI{G}_o{o}"
            x = np.arange(len(envs))
            for j, (lab, args) in enumerate((("D_V", (V, nV)), ("D_I", (I, nI)))):
                ss = [stat(paired(table, e, *args, k)) for e in envs]
                ax.errorbar(x + (j - 0.5) * 0.25, [s["mean"] for s in ss],
                            yerr=[[s["mean"] - s["lo"] for s in ss], [s["hi"] - s["mean"] for s in ss]],
                            fmt="o", capsize=3, label=lab)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(envs)
            ax.set_title(LABELS[k], fontsize=9)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Effect of spoofing with visible (D_V) vs iceberg (D_I) genuine leg, G={G}, offset {o}", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{stem}_effects_o{o}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
