#!/usr/bin/env python3
"""Draw the pipeline diagram (docs/img/00_pipeline.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

SURFACE, INK, INK2, AXIS, BLUE, BLUE_L, ORANGE_L = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#2a78d6", "#e6f0fd", "#fdeee7"
plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "sans-serif"]})
fig, ax = plt.subplots(figsize=(11, 12.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.set_xlim(0, 110); ax.set_ylim(0, 124); ax.axis("off")

def box(x, y, w, h, role, lines, fill=SURFACE, edge=AXIS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge, lw=1.8 if bold else 1.2))
    ax.text(x + 1.6, y + h - 1.9, role, fontsize=9.5, fontweight="bold", color=BLUE if bold else INK2, va="top")
    ax.text(x + 1.6, y + h - 5.0, "\n".join(lines), fontsize=9.6, color=INK, va="top", linespacing=1.45)

def diamond(cx, cy, text):
    ax.add_patch(Polygon([(cx, cy + 5.5), (cx + 13, cy), (cx, cy - 5.5), (cx - 13, cy)], fc=SURFACE, ec=INK2, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.6, color=INK)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, shrinkA=0, shrinkB=0))
    if label: ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=9.5, color=INK2, ha="center", va="center", fontweight="bold")

L, W = 4, 50                                    # left column
box(L, 113, W, 8, "INPUT", ["One JPEG photo, any size"])
box(L, 97, W, 12, "PREPROCESS A", ["Resize shortest side to 96, centre crop", "Scale with ImageNet mean and std", "Out: tensor 1 x 3 x 96 x 96"])
box(L, 75, W, 18, "MODEL 1 · FAST GATE  (.pte, ExecuTorch + XNNPACK)", ["MobileNetV2, distilled from model 2, 32-bit", "2.22 million weights · 8.9 MB file", "In: 1 x 3 x 96 x 96    Out: 1280 numbers", "Laptop: about 1.5 ms per photo"], fill=BLUE_L, edge=BLUE, bold=True)
box(L, 61, W, 10, "DECISION TABLE 1  (NumPy)", ["6 labels x 1280 weights", "Out: probability for each of 6 labels"])
diamond(L + W / 2, 50, "Top probability\nat least 0.90?")
R = 60                                          # right column
box(R, 97, 46, 12, "PREPROCESS B", ["Resize shortest side to 256, centre crop", "Scale pixels to 0-1", "Out: tensor 1 x 3 x 256 x 256"])
box(R, 75, 46, 18, "MODEL 2 · CAREFUL  (.pte, ExecuTorch + XNNPACK)", ["Apple MobileCLIP2-S0 image tower, fine-tuned", "11.4 million weights · 45.5 MB file, 32-bit", "In: 1 x 3 x 256 x 256    Out: 512 numbers", "Laptop: about 59 ms per photo"], fill=BLUE_L, edge=BLUE, bold=True)
box(R, 61, 46, 10, "DECISION TABLE 2  (NumPy)", ["6 labels x 512 weights", "Out: probability for each of 6 labels"])
diamond(R + 23, 50, "Top probability\nat least 0.70?")
box(L, 22, W, 16, "OUTPUT", ["Label, e.g. paper", "Probability for every label", "Tags, e.g. recyclable, dry_waste, paper_stream", "One line in the audit log (model hash, time, path)"])
box(R, 24, 46, 12, "REVIEW", ["Photo moved to the review folder", "Logged as not sure, with the best guess"], fill=ORANGE_L, edge="#eb6834")

arrow(L + W / 2, 113, L + W / 2, 109.6); arrow(L + W / 2, 97, L + W / 2, 93.6); arrow(L + W / 2, 75, L + W / 2, 71.6); arrow(L + W / 2, 61, L + W / 2, 55.5)
arrow(L + W / 2, 44.5, L + W / 2, 38.6, "yes  (65% of photos)", lx=14)
ax.plot([L + W / 2 + 13, 57, 57, R + 23, R + 23], [50, 50, 111.5, 111.5, 109.6], color=INK2, lw=1.3)
ax.annotate("", (R + 23, 109.4), (R + 23, 111.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3)); ax.text(57.8, 80, "no", fontsize=9.5, color=INK2, fontweight="bold", rotation=90, va="center")
arrow(R + 23, 97, R + 23, 93.6); arrow(R + 23, 75, R + 23, 71.6); arrow(R + 23, 61, R + 23, 55.5)
arrow(R + 23, 44.5, R + 23, 36.6, "no", lx=3)
ax.plot([76.5, 59, 59], [47.25, 47.25, 30], color=INK2, lw=1.3)
ax.annotate("", (L + W + 0.5, 30), (59, 30), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))
ax.text(66, 45.2, "yes", fontsize=9.5, color=INK2, fontweight="bold")
ax.text(L, 14, "Measured on 240 held-out TrashNet photos, laptop CPU, 4 threads:  95.8% correct (230 of 240).\nNot yet measured on the Raspberry Pi.", fontsize=10, color=INK2, va="top", linespacing=1.5)
fig.savefig(Path(__file__).parent / "img" / "00_pipeline.png", dpi=150, bbox_inches="tight", facecolor=SURFACE); print("wrote 00_pipeline.png")
