"""Case 3: per-delete latency vs. fan-out (memberships of the deleted movie) for multi-label
maintenance in PostgreSQL. Reads ../../experiments/multilabel/results_maint_join.json (measured,
smoke timings). Colours: dataviz reference categorical slots 1-3 (fixed order).
Regenerate: python3 case3_fanout_latency.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[2] / "experiments/multilabel/results_maint_join.json"
R = json.load(open(SRC))
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
SERIES = [("softavg_ivm_join", "0.1", "#2a78d6", "-", "ours, join view, eps 0.1"),
          ("softavg_ivm_join", "0.02", "#2a78d6", (0, (3, 2)), "ours, join view, eps 0.02"),
          ("pg_ivm_linear_join", "0.1", "#eb6834", "-", "pg_ivm linear rewrite, eps 0.1"),
          ("softavg_ivm_single", "0.1", "#1baf7a", "-", "single-label view (one group per edit)")]

fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=6.5)
for arm, eps, col, ls, name in SERIES:
    a = next(x for x in R["arms"] if x["arm"] == arm and x["eps"] == eps)
    per = a["fanout_sweep"]["per_fanout"]
    f = sorted((int(k) for k in per), key=int)
    y = [per[str(k)]["delete"]["median_ms"] for k in f]
    ax.plot(f, y, color=col, ls=ls, lw=1.6, marker="o", ms=3, label=name)
ax.set_xlabel("memberships of the deleted movie (fan-out)", fontsize=7, color=INK)
ax.set_ylabel("ms per delete (median)", fontsize=7, color=INK)
ax.set_ylim(bottom=0)
ax.legend(fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_suffix(".pdf"))
