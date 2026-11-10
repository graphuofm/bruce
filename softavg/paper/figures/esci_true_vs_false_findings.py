"""Case 1 (ESCI): true vs false findings on the test queries, one curve per estimator family swept
over its parameter (softavg: eps; threshold: tau; global top-k: k; per-group top-b: b). The point
chosen on the dev queries is ringed. Reads experiments/esci_task/results.json only.

The paper compares four families, so the relative-threshold family stored in results.json
("thresh_relf") is not drawn.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[2] / "experiments/esci_task/results.json"
R = json.loads(SRC.read_text())
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
FAM = [("softavg", r"softavg, swept $\varepsilon$", "#2a78d6", "-"),
       ("thresh", r"AVG over $s\geq\tau$", "#eb6834", "--"),
       ("topk", r"global top-$k$ + AVG", "#1baf7a", "-."),
       ("facet_topb", r"per-group top-$b$ + AVG", "#7a52c7", ":")]
POOL = {"labelled": "labelled pool", "labelled+distractors": "with distractors"}

keys = R["figure_settings"]
fig, axes = plt.subplots(1, len(keys), figsize=(7.0, 2.25), squeeze=False)
for i, (ax, key) in enumerate(zip(axes[0], keys)):
    S = R["settings"][key]
    for fam, label, col, ls in FAM:
        g = S["grid"][fam]
        x = [p["test"]["findings_false"] for p in g]
        y = [p["test"]["findings_detected"] for p in g]
        ax.plot(x, y, color=col, ls=ls, lw=1.4, label=label)
        ch = S["chosen"][fam]["param"]
        for p, xi, yi in zip(g, x, y):
            if p["param"] == ch:
                ax.plot([xi], [yi], "o", ms=7, mfc="none", mec=col, mew=1.4)
    gt = S["chosen"]["avg"]["test"]["findings_gt"]
    ax.axhline(gt, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.text(ax.get_xlim()[1], gt, "ground-truth findings", ha="right", va="bottom", fontsize=6, color=MUTED)
    pool, facet, value, rel = key.split("|")
    facet = "category" if facet == "main_category" else facet
    value = "log price" if value == "log_price" else value
    ax.set_title(f"({'abc'[i]}) {facet}, {value}, {rel}, {POOL.get(pool, pool)}", fontsize=7, color=INK, loc="left")
    ax.set_xlabel("false findings", fontsize=7, color=INK)
    if i == 0:
        ax.set_ylabel("true findings detected", fontsize=7, color=INK)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=len(FAM))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
fig.savefig(Path(__file__).with_suffix(".pdf"))
