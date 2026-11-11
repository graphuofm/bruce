"""Schematic (no data): lifecycle of the removable softmax state.
Updates touch a constant-size state; Value runs the error test and either returns the value or
asks the host to refold the live rows into a fresh state.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42})
INK, MUTED = "#222222", "#6b6b6b"
CORE, OK, BAD, HOST = "#dbe8f6", "#e3f0e0", "#f7e1dc", "#f2f2f2"

fig, ax = plt.subplots(figsize=(3.45, 1.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, text, fc, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x0, y0, x1, y1, text=None, dx=0, dy=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, fontsize=5.8, color=MUTED, ha="center", va="center")


box(1, 37, 20, 9, "absorb row\nmerge states", HOST)
box(1, 21, 20, 9, "remove row\n(window, delete)", HOST)
box(31, 25, 26, 17, "state\n$(\\mu, z, y)$\n$(n, e_z, e_y, v_{\\max})$", CORE, bold=False)
box(67, 29, 30, 9, "error test\n$B \\leq \\tau_n$", CORE)
box(67, 43, 30, 6, "return $y/z$", OK)
box(67, 14, 30, 8, "refold: host folds\nthe live rows again", BAD)
arrow(21, 41.5, 31, 37)
arrow(21, 25.5, 31, 30)
arrow(57, 33.5, 67, 33.5, "Value", dy=2.6)
arrow(82, 38, 82, 43, "pass", dx=6)
arrow(82, 29, 82, 22, "fail", dx=5.5)
ax.annotate("", xy=(44, 25), xytext=(67, 16.5),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, connectionstyle="angle3,angleA=180,angleB=90"))
ax.text(50, 13.5, "fresh state", fontsize=5.8, color=MUTED, ha="center")
ax.text(44, 4.5, "constant size, $O(1)$ arithmetic per update;\nonly a failed test costs a pass over the live rows",
        fontsize=5.8, color=MUTED, ha="center", va="center")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(Path(__file__).with_suffix(".pdf"))
