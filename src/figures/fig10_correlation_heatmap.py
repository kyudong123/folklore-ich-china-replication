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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from remote_style_common import load_analysis_panel, save_figure, set_style


def main():
    set_style()
    df = load_analysis_panel()
    pop_col = "pop_2020" if "pop_2020" in df.columns else "pop_2010"
    pop_label = "Population"
    cols = ["festival_count", "festival_per_million", "minority_share", pop_col, "minority_pop", "han_pct"]
    labels = ["Folklore\nrecords", "ICH density", "Minority\nshare", pop_label, "Minority\npopulation", "Han share"]
    corr = df[cols].astype(float).corr().values
    n = len(cols)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(n):
        for j in range(n):
            r = corr[i, j]
            ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor="#FAFBFD", edgecolor="#E5E7EB", linewidth=0.65))
            if i < j:
                continue
            if i == j:
                continue
            ax.add_patch(mpatches.Circle((j, i), radius=0.42 * abs(r),
                                         facecolor=cmap(norm(r)), edgecolor="white", linewidth=0.6))
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=8.4,
                    color="white" if abs(r) > 0.58 else "#111827")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Pearson r")
    save_figure(fig, "fig10_correlation_heatmap", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

