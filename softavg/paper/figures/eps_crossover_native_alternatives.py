"""Cheaper spellings vs exact softavg across eps (IMDb, Amazon), single-column 2x2 figure.

Reads experiments/crossover/results.json only (no numbers typed here).
Top row: fraction of the 200 queries for which a spelling is acceptable (every group answered,
every group within 1% of its value range). Bottom row: median over queries of the worst
covered-group error, in units of the group's value range (clipped at 1e-6 for readability).
"Best truncated AVG" is, per eps, the best of all top-k + AVG, threshold + AVG and sample + AVG
variants in results.json (maximum acceptable fraction; minimum median error).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[2] / "experiments/crossover/results.json"
R = json.load(open(SRC))
TOL = R["meta"]["tolerance"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
SER = [("avg", "AVG", "#2a78d6", "-"),
       ("topk_soft", "top-10k + softmax", "#1baf7a", "--"),
       ("best_trunc", "best truncated AVG", "#eb6834", ":")]


def series(C, key, field):
    keys = [k for k in C["methods"]["plain_avg"]["per_eps"] if k != "inf"]
    xs = [float(k) for k in keys]
    alts = [m for m in C["methods"] if m != "plain_avg" and "_avg_" in m]

    def val(m, k):
        v = C["methods"][m]["per_eps"][k][field]
        return float("nan") if v is None else v

    if key == "avg":
        ys = [val("plain_avg", k) for k in keys]
    elif key == "topk_soft":
        ys = [val("topk_softavg_k10000", k) for k in keys]
    else:
        pick = max if field == "frac_queries_acceptable" else min
        ys = []
        for k in keys:
            vals = [val(a, k) for a in alts if not math.isnan(val(a, k))]
            ys.append(pick(vals) if vals else float("nan"))
    return xs, ys


corpora = [("imdb", "IMDb"), ("amazon", "Amazon")]
fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.0), sharex="col")
for j, (name, label) in enumerate(corpora):
    C = R["corpora"][name]
    for i, field in enumerate(["frac_queries_acceptable", "worst_facet_err_median_q"]):
        ax = axes[i, j]
        for key, lab, col, ls in SER:
            xs, ys = series(C, key, field)
            if i == 1:
                ys = [max(y, 1e-6) if not math.isnan(y) else y for y in ys]
            ax.plot(xs, ys, color=col, ls=ls, lw=1.4, label=lab)
        ax.set_xscale("log")
        if i == 0:
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"({'ab'[j]}) {label}, acceptable", fontsize=7, color=INK, loc="left")
        else:
            ax.set_yscale("log")
            ax.axhline(TOL, color=MUTED, lw=0.8, ls=(0, (4, 2)))
            ax.set_title(f"({'cd'[j]}) {label}, worst-group error", fontsize=7, color=INK, loc="left")
            ax.set_xlabel(r"temperature $\varepsilon$", fontsize=7, color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=6.5)
axes[0, 0].set_ylabel("fraction of queries", fontsize=7, color=INK)
axes[1, 0].set_ylabel("error / range", fontsize=7, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.4, w_pad=0.6)
fig.savefig(Path(__file__).with_suffix(".pdf"))
