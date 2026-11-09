"""Window cost of each ladder rung vs frame size k (Case 2 data: Amazon reviews in time order,
eps = 0.02, query 0), in PostgreSQL 18.6 and SQLite 3.37.

Reads the measured runs; nothing is typed by hand:
  experiments/softmax_ladder/pg_window.jsonl      (PostgreSQL engine run, median_ms per rung)
  experiments/softmax_ladder/sqlite_timing.jsonl  (SQLite interleaved timing, median_ms per rung)
Timings are from a shared machine ("smoke, machine shared" in the files); compare rungs within
a panel. Prints the ratios quoted in the paper.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "experiments/softmax_ladder"
DATASET, EPS, Q = "amazon", 0.02, 0
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
COL = {"S1": "#eb6834", "S2": "#7a52c7", "S3": "#2a78d6", "S4": "#1baf7a"}
NAME = {"S1": "S1 shifted SQL (host restarts frame)", "S2": "S2 online, refold frame",
        "S3": "S3 removal, no certificate", "S4": "S4 certified removal (ours)"}
MARK = {"S1": "s", "S2": "D", "S3": "o", "S4": "^"}


def pg_runs():
    out = {}
    for line in open(ROOT / "pg_window.jsonl"):
        r = json.loads(line)
        if r.get("engine") == "postgres" and "median_ms" in r and r.get("error") is None:
            out[(r["dataset"], r["eps"], r["q"], r["k"], r["rung"])] = r["median_ms"]
    return out


def sqlite_runs():
    out = {}
    for line in open(ROOT / "sqlite_timing.jsonl"):
        r = json.loads(line)
        if "median_ms" in r:
            for rung, ms in r["median_ms"].items():
                out[(r["dataset"], r["eps"], r["q"], r["k"], rung)] = ms
    return out


pg, sq = pg_runs(), sqlite_runs()
ks = [10, 100, 1000]

# ratios quoted in the text (all row sets and eps measured in the PG engine run)
s2s4 = [pg[(d, e, q, 1000, "S2")] / pg[(d, e, q, 1000, "S4")]
        for (d, e, q, k, r) in pg if k == 1000 and r == "S4" and (d, e, q, 1000, "S2") in pg]
s4s3 = sorted(pg[(d, e, q, k, "S4")] / pg[(d, e, q, k, "S3")]
              for (d, e, q, k, r) in pg if r == "S4" and (d, e, q, k, "S3") in pg)
print(f"PG S2/S4 at k=1000: {min(s2s4):.0f}..{max(s2s4):.0f}x over {len(s2s4)} cells")
print(f"PG S4/S3: median {s4s3[len(s4s3)//2]:.2f}, range {s4s3[0]:.2f}..{s4s3[-1]:.2f} over {len(s4s3)} cells")
mod = sorted(pg[(d, e, q, k, "S4")] / pg[(d, e, q, k, "S3")]
             for (d, e, q, k, r) in pg if r == "S4" and e >= 0.02 and (d, e, q, k, "S3") in pg)
print(f"PG S4/S3 at eps >= 0.02: median {mod[len(mod)//2]:.2f} over {len(mod)} cells")
for r in ("S1", "S2", "S3", "S4"):
    print("PG", r, [round(pg.get((DATASET, EPS, Q, k, r), float("nan"))) for k in ks])
for r in ("S2", "S3", "S4"):
    print("SQLite", r, [round(sq.get((DATASET, EPS, Q, k, r), float("nan"))) for k in ks])

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
for ax, (title, runs, rungs) in zip(axes, [("(a) PostgreSQL 18.6", pg, ["S1", "S2", "S3", "S4"]),
                                          ("(b) SQLite 3.37", sq, ["S2", "S3", "S4"])]):
    for r in rungs:
        ys = [runs[(DATASET, EPS, Q, k, r)] / 1000 for k in ks]
        ax.plot(ks, ys, color=COL[r], lw=1.4, marker=MARK[r], ms=4, label=NAME[r])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.5)
    ax.set_xlabel("frame rows k", fontsize=7, color=INK)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
axes[0].set_ylabel("query time (s)", fontsize=7, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
fig.savefig(Path(__file__).with_suffix(".pdf"))
