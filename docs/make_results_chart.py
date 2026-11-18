"""Draw docs/images/results.png from the numbers printed by compare.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Original", "+ Strong augmentation", "+ Borrowed gliomas"]
METRICS = {  # from `python compare.py` (shared 1,562-image test set)
    "Overall accuracy": [97.4, 97.9, 98.0],
    "Glioma recall": [90.6, 94.5, 96.4],
    "Other-source glioma recall": [23.7, 55.3, 76.3],
}
LIGHT, DARK = "#9ec5f4", "#2a78d6"  # blue ramp 200 / 450
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), facecolor=SURFACE)
for ax, (title, vals) in zip(axes, METRICS.items()):
    ax.set_facecolor(SURFACE)
    colors = [LIGHT, LIGHT, DARK]
    ax.barh(range(3), vals, color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=10, color=INK,
                fontweight="bold" if i == 2 else "normal")
    ax.set_xlim(0, 112)
    ax.set_yticks(range(3), MODELS if ax is axes[0] else [""] * 3, color=MUTED, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle("Fixing the glioma shortcut: same 1,562 held-out test scans", x=0.01, ha="left",
             fontsize=12.5, color=INK, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.93))
out = Path(__file__).parent / "images" / "results.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("Saved", out)
