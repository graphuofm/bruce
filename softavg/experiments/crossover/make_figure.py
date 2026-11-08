"""Figure for the eps crossover. Reads results.json only (no numbers typed here).

Top row (per corpus): fraction of the 200 queries for which the alternative is acceptable
(all facets covered, every facet error <= 1% of the facet's range); "best AVG alternative"
is, per eps, the maximum over all top-k + AVG, threshold + AVG and sample + AVG variants.
Middle row: median over queries of the worst covered-facet error (units of the facet value
range), clipped at 1e-6 for readability; dashed line = the 1% bar.
Bottom row: median ESS / n_f over (query, facet) vs eps, the diagnostic that tracks the
crossover. Writes figures/crossover.pdf and figures/crossover.png.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
SERIES = [("plain_avg", "plain AVG", "#2a78d6"),
          ("topk_avg_k10000", "top-10k + AVG", "#eb6834"),
          ("topk_softavg_k10000", "top-10k + client softavg", "#1baf7a"),
          ("thresh_avg_tau0.4", "s >= 0.4 + AVG", "#6250d6")]
TOL = R["meta"]["tolerance"]


def eps_vals(keys):
    return [float(k) for k in keys if k != "inf"]


corpora = list(R["corpora"])
fig, axes = plt.subplots(3, len(corpora), figsize=(4.2 * len(corpora), 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 3, 2]})
ACC = [("plain_avg", "plain AVG", "#2a78d6"),
       ("topk_softavg_k10000", "top-10k + client softavg", "#1baf7a"),
       ("BEST_AVG", "best AVG alternative", "#e34948")]
for j, name in enumerate(corpora):
    C = R["corpora"][name]
    ax0 = axes[0, j]
    avg_alts = [m for m in C["methods"] if m != "plain_avg" and "_avg_" in m]
    for m, lab, col in ACC:
        ks = [k for k in C["methods"]["plain_avg"]["per_eps"] if k != "inf"]
        if m == "BEST_AVG":
            ys = [max(C["methods"][a]["per_eps"][k]["frac_queries_acceptable"] for a in avg_alts) for k in ks]
        else:
            ys = [C["methods"][m]["per_eps"][k]["frac_queries_acceptable"] for k in ks]
        ax0.plot(eps_vals(ks), ys, color=col, lw=2, label=lab)
    ax0.set_xscale("log"); ax0.set_ylim(-0.03, 1.03)
    ax0.set_title(f"{name}: {C['rows_admitted']:,} rows, {C['facets']} facets", fontsize=10)
    if j == 0:
        ax0.set_ylabel("queries acceptable\n(fraction)")
    ax0.grid(True, which="major", color="#e4e3df", lw=0.6)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)
    ax = axes[1, j]
    for m, lab, col in SERIES:
        pe = C["methods"][m]["per_eps"]
        ks = [k for k in pe if k != "inf"]
        xs = eps_vals(ks)
        ys = [pe[k]["worst_facet_err_median_q"] for k in ks]
        ax.plot(xs, [max(y, 1e-6) if y is not None else float("nan") for y in ys], color=col, lw=2, label=lab)
    ax.axhline(TOL, color="#52514e", lw=1, ls="--")
    ax.text(xs[0], TOL * 1.15, "1% of range", color="#52514e", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    if j == 0:
        ax.set_ylabel("worst-facet error / range\n(median over queries)")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax2 = axes[2, j]
    E = C["ess_by_eps"]
    ks = [k for k in E if k != "inf"]
    ax2.plot(eps_vals(ks), [E[k]["ess_frac_median"] for k in ks], color="#0b0b0b", lw=2, label="median")
    ax2.fill_between(eps_vals(ks), [E[k]["ess_frac_p10"] for k in ks], [E[k]["ess_frac_p90"] for k in ks],
                     color="#0b0b0b", alpha=0.12, lw=0, label="p10-p90")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("eps")
    if j == 0:
        ax2.set_ylabel("ESS / facet rows")
    ax2.grid(True, which="major", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
axes[0, 0].legend(fontsize=8, frameon=False, loc="center left")
axes[1, 0].legend(fontsize=8, frameon=False, loc="lower left")
axes[2, 0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"crossover.{ext}"), dpi=160)
print("wrote figures/crossover.{pdf,png}")
