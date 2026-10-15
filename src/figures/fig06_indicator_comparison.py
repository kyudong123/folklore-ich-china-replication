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
from remote_style_common import PROVINCE_EN, PALETTE, load_panel, save_figure, set_style


def main():
    set_style()
    df = load_panel().dropna(subset=["pop_2020", "folk_per_million"]).copy()
    df["folk_per_million_minority"] = df["folk_count"] / (df["minority_pop"] / 1_000_000.0)
    df = df.sort_values(["folk_count", "folk_per_million"], ascending=[False, False]).reset_index(drop=True)
    labels = df["province"].map(PROVINCE_EN).tolist()
    y = np.arange(len(df))

    count_values = df["folk_count"].astype(float).to_numpy()
    density_values = df["folk_per_million"].astype(float).to_numpy()
    minority_density = df["folk_per_million_minority"].astype(float).to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, sharey=True, figsize=(8.27, 8.75),
        gridspec_kw={"width_ratios": [1.04, 0.92, 1.05], "wspace": 0.060}
    )

    count_colors = ["#2878B5" if v < count_values.max() else "#0F4C81" for v in count_values]
    density_colors = ["#E69F00" if v < 1.0 else "#D55E00" for v in density_values]
    minority_colors = ["#009E73" if np.isfinite(v) and v < 50 else "#0072B2" for v in minority_density]

    ax1.barh(y, count_values, height=0.68, color=count_colors, edgecolor="white", linewidth=0.35)
    ax2.barh(y, density_values, height=0.68, color=density_colors, edgecolor="white", linewidth=0.35)
    valid_minority = np.isfinite(minority_density)
    ax3.barh(y[valid_minority], minority_density[valid_minority], height=0.68,
             color=np.array(minority_colors, dtype=object)[valid_minority],
             edgecolor="white", linewidth=0.35)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Folklore ICH records")
    ax2.set_xlabel("Records per million\npersons")
    ax3.set_xlabel("Records per million\nminority persons")

    ax1.set_xlim(0, max(count_values) * 1.18)
    ax2.set_xlim(0, max(density_values) * 1.15)
    ax3.set_xlim(0, np.nanmax(minority_density[valid_minority]) * 1.10)

    for yi, val in zip(y, count_values):
        ax1.text(val + max(count_values) * 0.018, yi, f"{int(val)}",
                 ha="left", va="center", fontsize=7.5, color="#263238")

    for yi, val in zip(y, density_values):
        ax2.text(val + max(density_values) * 0.018, yi, f"{val:.3f}",
                 ha="left", va="center", fontsize=7.5, color="#263238")

    max_minority = np.nanmax(minority_density[valid_minority])
    for yi, val, ok in zip(y, minority_density, valid_minority):
        if ok:
            ax3.text(val + max_minority * 0.018, yi, f"{val:.1f}",
                     ha="left", va="center", fontsize=7.0, color="#263238")
        else:
            ax3.text(max_minority * 0.020, yi, "n.a.",
                     ha="left", va="center", fontsize=7.0, color="#6B7280")

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="x", color=PALETTE["grid"], lw=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ax2.tick_params(axis="y", labelleft=False)
    ax3.tick_params(axis="y", labelleft=False)
    ax1.text(0.00, 1.012, "(a) Absolute records", transform=ax1.transAxes,
             ha="left", va="bottom", fontsize=9.2, fontweight="bold")
    ax2.text(0.00, 1.012, "(b) Total-population density", transform=ax2.transAxes,
             ha="left", va="bottom", fontsize=9.2, fontweight="bold")
    ax3.text(0.00, 1.012, "(c) Minority-population density", transform=ax3.transAxes,
             ha="left", va="bottom", fontsize=9.2, fontweight="bold")
    fig.subplots_adjust(left=0.185, right=0.975, top=0.965, bottom=0.082)
    save_figure(fig, "fig06_indicator_comparison", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

