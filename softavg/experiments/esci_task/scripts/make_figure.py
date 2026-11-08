"""Figure: findings detected vs false findings on the test queries, one curve
per estimator family swept over its parameter (softavg: eps; threshold: tau;
global top-k: k; per-facet top-b: b).  The dev-chosen point of each family is
ringed.  Every number is read from ../results.json.

Writes ../figures/findings_tradeoff.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
R = json.loads((HERE / "results.json").read_text())
keys = R["figure_settings"]
FAM = [("softavg", "softavg (eps)", "#2a78d6"), ("thresh", "AVG, s >= tau", "#eb6834"),
       ("topk", "global top-k + AVG", "#1baf7a"), ("facet_topb", "per-facet top-b + AVG", "#eda100"),
       ("thresh_relf", "AVG, s >= max_facet(s) - d", "#e87ba4")]

fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.9), squeeze=False)
for ax, key in zip(axes[0], keys):
    S = R["settings"][key]
    for fam, label, col in FAM:
        g = S["grid"][fam]
        x = [p["test"]["findings_false"] for p in g]
        y = [p["test"]["findings_detected"] for p in g]
        ax.plot(x, y, "-o", color=col, lw=2, ms=4, label=label)
        ch = S["chosen"][fam]["param"]
        for p, xi, yi in zip(g, x, y):
            if p["param"] == ch:
                ax.plot([xi], [yi], "o", ms=11, mfc="none", mec=col, mew=2)
    ax.axhline(S["chosen"]["avg"]["test"]["findings_gt"], color="#888", lw=1, ls="--")
    ax.text(ax.get_xlim()[1], S["chosen"]["avg"]["test"]["findings_gt"], "GT findings",
            ha="right", va="bottom", fontsize=8, color="#555")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("false findings (test)")
    ax.set_ylabel("true findings detected (test)")
    pool, facet, value, rel = key.split("|")
    ax.set_title(f"{pool}, facet={facet}\nvalue={value}, relevant={rel}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=len(FAM))
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = HERE / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "findings_tradeoff.pdf")
fig.savefig(out / "findings_tradeoff.png", dpi=150)
print("wrote", out / "findings_tradeoff.pdf")
