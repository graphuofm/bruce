"""Schematic (not data): how one window frame slide is executed by the shared core in the
three hosts. Concept figure drawn with matplotlib because draw.io is not installed here;
regenerate with `python3 host_contracts_execution.py`."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, CORE, OK, BAD, HOST = "#222222", "#dbe8f5", "#e3f1e3", "#f8e1dc", "#f2f2f2"
fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

def box(x, y, w, h, text, fc, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.15)

def arrow(x1, y1, x2, y2, text=None, dx=0.8, dy=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color=INK))
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, fontsize=6.3, color=INK,
                ha="left", va="center")

# shared core
box(18, 40.5, 64, 6, "shared Rust core: InvAcc = online softmax state $(\\mu,z,y)$ + error bounds $(n,e_z,e_y,v_{\\max})$\n"
    "absorb / merge / remove;  certified iff $B \\leq \\tau_n$", CORE, fs=7.0)

cols = [(2, "PostgreSQL"), (35, "SQLite"), (68, "DuckDB")]
for x, name in cols:
    box(x, 32.5, 30, 4.2, name, HOST, fs=7.6, bold=True)
    arrow(50, 40.3, x + 15, 36.9)

# PostgreSQL column
box(2, 23.5, 30, 6, "frame slides: MINVFUNC\ncalls remove", OK)
box(2, 13, 14, 6.5, "certified:\nreturn state", OK, fs=6.8)
box(18, 13, 14, 6.5, "not certified:\nreturn NULL", BAD, fs=6.8)
box(2, 2, 30, 7, "executor restarts the frame\n(re-aggregates live rows)", HOST, fs=6.8)
arrow(17, 32.3, 17, 29.7)
arrow(10, 23.3, 9, 19.7); arrow(24, 23.3, 25, 19.7)
arrow(25, 12.8, 22, 9.2)

# SQLite column
box(35, 23.5, 30, 6, "frame slides: xInverse\ncalls remove (cannot refuse)", OK)
box(35, 13, 14, 6.5, "certified:\nappend to\nframe shadow", OK, fs=6.6)
box(51, 13, 14, 6.5, "not certified:\nflag refold", BAD, fs=6.8)
box(35, 2, 30, 7, "xValue: extension refolds\nfrom its own frame shadow", HOST, fs=6.8)
arrow(50, 32.3, 50, 29.7)
arrow(43, 23.3, 42, 19.7); arrow(57, 23.3, 58, 19.7)
arrow(58, 12.8, 55, 9.2)

# DuckDB column
box(68, 23.5, 30, 6, "no inverse in the C API:\nsegment tree of partial states", OK)
box(68, 13, 30, 6.5, "each frame = merge of\n$O(\\log n)$ nodes; remove never called", OK, fs=6.8)
box(68, 2, 30, 7, "online state (S2) suffices:\nnothing is subtracted", HOST, fs=6.8)
arrow(83, 32.3, 83, 29.7); arrow(83, 23.3, 83, 19.7); arrow(83, 12.8, 83, 9.2)

fig.tight_layout(pad=0.1)
fig.savefig("host_contracts_execution.pdf")
