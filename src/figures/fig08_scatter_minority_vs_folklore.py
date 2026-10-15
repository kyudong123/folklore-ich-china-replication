# -*- coding: utf-8 -*-
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_DIR = Path(__file__).resolve().parents[2]
FIGURE_OUTPUT = PACKAGE_DIR / "results" / "figures"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_env import setup_environment
setup_environment()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from remote_style_common import PROVINCE_EN, PALETTE, load_analysis_panel, ols_fit, save_figure, set_style


def main():
    set_style()
    df = load_analysis_panel()
    x = df["minority_share"].astype(float).to_numpy()
    y = df["festival_per_million"].astype(float).to_numpy()
    intercept, slope, fitted, residual, r2 = ols_fit(x, y)
    xline = np.linspace(x.min(), x.max(), 200)
    yline = intercept + slope * xline
    n = len(x)
    serr = np.sqrt(np.sum(residual ** 2) / (n - 2))
    ci = 2.045 * serr * np.sqrt(1 / n + (xline - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig = plt.figure(figsize=(7.15, 5.25))
    gs = GridSpec(2, 2, width_ratios=[4.7, 1.05], height_ratios=[1.0, 4.0], hspace=0.04, wspace=0.04)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    scatter = ax.scatter(x, y, c=df["festival_count"], cmap="viridis", s=58,
                         edgecolor="white", linewidth=0.65, zorder=4)
    ax.plot(xline, yline, color=PALETTE["line"], lw=1.6, zorder=5)
    ax.fill_between(xline, yline - ci, yline + ci, color=PALETTE["line"], alpha=0.12, zorder=2)
    label_offsets = {
        "西藏": (8, 3),
        "青海": (8, 8),
        "新疆": (8, -12),
        "贵州": (8, 8),
        "山西": (-2, 13),
        "浙江": (13, 3),
        "福建": (10, -14),
    }
    for prov, offset in label_offsets.items():
        row = df[df["province"] == prov]
        if not row.empty:
            px = float(row["minority_share"].iloc[0])
            py = float(row["festival_per_million"].iloc[0])
            ann = ax.annotate(PROVINCE_EN[prov], (px, py),
                              xytext=offset, textcoords="offset points", fontsize=8.2,
                              zorder=7)
            ann.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
    ax_top.hist(x, bins=12, color="#A6CEE3", edgecolor="white")
    ax_right.hist(y, bins=12, orientation="horizontal", color="#FDBF6F", edgecolor="white")
    ax_top.axis("off")
    ax_right.axis("off")
    ax.set_xlabel("Minority population share")
    ax.set_ylabel("Folklore ICH density (items per million persons)")
    ax.grid(True, color=PALETTE["grid"], lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.03, 0.95, f"OLS: R2 = {r2:.3f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=10.2, bbox=dict(fc="white", ec="#D1D5DB", lw=0.6, pad=4))
    cbar = fig.colorbar(scatter, ax=ax_right, fraction=0.80, pad=0.10)
    cbar.set_label("Folklore records", fontsize=9)
    save_figure(fig, "fig08_scatter_minority_vs_folklore", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

