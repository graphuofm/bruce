"""Case 2 trace: rolling softavg over Amazon reviews in time order (SQLite 3.37.2 engine runs).
Reads ../../experiments/softmax_ladder/trace.json (measured); nothing is typed by hand.
Colours: first three slots of the dataviz reference categorical palette (pre-validated order).
Regenerate: python3 ladder_rolling_trace.py"""
import json, math, datetime as dt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[2] / "experiments/softmax_ladder/trace.json"
T = json.load(open(SRC))
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
COL = {"sqlite_S1": "#eb6834", "sqlite_S3": "#2a78d6", "sqlite_S4": "#1baf7a"}
NAME = {"sqlite_S1": "S1 constant shift", "sqlite_S3": "S3 removal, no certificate",
        "sqlite_S4": "S4 certified (ours)"}
FLOOR = 1e-17

def trace(k, eps):
    for tr in T["traces"]:
        if tr["k"] == k and abs(tr["eps"] - eps) < 1e-12:
            return tr
    raise KeyError((k, eps))

def times(tr):
    return [dt.datetime.utcfromtimestamp(x / 1000) for x in tr["ts_ms"]]

def err(tr, key):
    out = []
    for x, r, m in zip(tr["series"][key], tr["reference"], tr["vmax"]):
        if x is None or r is None or not math.isfinite(x):
            out.append(float("nan"))
        else:
            out.append(max(abs(x - r) / m, FLOOR))
    return out

fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.9), gridspec_kw={"height_ratios": [1.1, 1, 1]})
for ax in axes:
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.5)

# (a) values, k = 50, eps = 0.02: plausible but wrong
tr = trace(50, 0.02); t = times(tr)
e3 = err(tr, "sqlite_S3")
w = max(range(len(e3)), key=lambda i: e3[i] if e3[i] == e3[i] else -1)   # worst S3 row
lo, hi = max(0, w - 60), min(len(t), w + 60)
a = axes[0]
a.plot(t[lo:hi], tr["reference"][lo:hi], color=INK, lw=1.6, label="exact")
a.plot(t[lo:hi], tr["series"]["sqlite_S4"][lo:hi], color=COL["sqlite_S4"], lw=0.9, ls=(0, (2, 2)), label=NAME["sqlite_S4"])
a.plot(t[lo:hi], tr["series"]["sqlite_S3"][lo:hi], color=COL["sqlite_S3"], lw=1.0, label=NAME["sqlite_S3"])
a.set_ylabel("rolling rating", fontsize=7, color=INK)
a.set_title(f"(a) values, k = {tr['k']}, eps = {tr['eps']}", fontsize=7.5, color=INK, loc="left")
a.legend(fontsize=5.8, frameon=False, loc="lower left")
loc = matplotlib.dates.AutoDateLocator(maxticks=4)
a.xaxis.set_major_locator(loc); a.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))

for ax, (k, eps), tag in ((axes[1], (50, 0.02), "(b)"), (axes[2], (50, 0.001), "(c)")):
    tr = trace(k, eps); t = times(tr)
    for key, lw in (("sqlite_S1", 2.0), ("sqlite_S3", 0.7), ("sqlite_S4", 0.7)):
        ax.plot(t, err(tr, key), color=COL[key], lw=lw, label=NAME[key])
    ax.set_yscale("log")
    ax.set_ylabel("abs err / max|v|", fontsize=7, color=INK)
    ax.set_title(f"{tag} error, k = {k}, eps = {eps}", fontsize=7.5, color=INK, loc="left")
    loc = matplotlib.dates.AutoDateLocator(maxticks=4)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))
axes[2].legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
               handlelength=1.4, columnspacing=0.8)
for ax in axes: ax.xaxis.get_offset_text().set_fontsize(6); ax.xaxis.get_offset_text().set_color(MUTED)
fig.tight_layout(pad=0.3, h_pad=0.6)
fig.savefig(Path(__file__).with_suffix(".pdf"))
