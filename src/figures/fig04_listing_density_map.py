# -*- coding: utf-8 -*-
"""Map the observed 2020 population-standardized listing density for 31 mainland units."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE_DIR = Path(__file__).resolve().parents[2]
PANEL = PACKAGE_DIR / "data" / "processed" / "provincial_analysis_panel_2020.csv"
FIGURE_OUTPUT = PACKAGE_DIR / "results" / "figures"
sys.path.insert(0, str(HERE))

from plot_env import setup_environment  # noqa: E402

setup_environment()

from remote_style_common import (  # noqa: E402
    add_map_frame,
    add_map_labels,
    add_north_arrow,
    add_scale_bar,
    add_south_china_inset,
    decode_topojson,
    draw_base_map,
    draw_nine_dash,
    china_lambert,
    province_centroid,
    set_style,
    save_figure,
)


def main() -> None:
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")
    if len(panel) != 31 or panel["province"].nunique() != 31:
        raise ValueError("Expected one row for each of 31 mainland provincial units")
    if panel["folklore_count"].sum() != 471:
        raise ValueError("The analysis panel must contain 471 listed records")
    calculated = panel["folklore_count"] / panel["pop_2020"] * 1_000_000
    if not ((calculated - panel["folklore_per_million"]).abs() < 1e-9).all():
        raise ValueError("Listing density does not match counts and population")

    values = dict(zip(panel["province"], calculated.astype(float)))
    set_style()
    provinces = decode_topojson()
    cmap = plt.get_cmap("YlGnBu")
    norm = mcolors.Normalize(vmin=0, vmax=3.5)

    fig = plt.figure(figsize=(8.27, 6.05))
    ax = fig.add_axes([0.075, 0.185, 0.840, 0.675])
    draw_base_map(ax, provinces, values, cmap=cmap, norm=norm,
                  edge="#334155", lw=0.40, missing="#FFFFFF")
    add_map_labels(ax, provinces, fontsize=5.8, skip_extra={"西藏", "青海"})
    for key, label in [("西藏", "Tibet"), ("青海", "Qinghai")]:
        province = next(p for p in provinces if p["key"] == key)
        lon, lat = province_centroid(province)
        x, y = china_lambert(lon, lat)
        ax.text(float(x), float(y), label, ha="center", va="center",
                fontsize=5.8, color="white", zorder=6)
    draw_nine_dash(ax, lw=0.65)
    add_north_arrow(ax)
    add_scale_bar(ax)
    add_south_china_inset(fig, ax, provinces, values, cmap=cmap, norm=norm)
    add_map_frame(ax)

    cax = fig.add_axes([0.24, 0.075, 0.52, 0.018])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=cax, orientation="horizontal",
                            ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7.5, length=2)
    colorbar.set_label("National folklore ICH records per million residents", fontsize=8.5)

    save_figure(fig, "fig04_listing_density_map", FIGURE_OUTPUT)
    print(f"Saved Figure 4 for {len(values)} units and {panel['folklore_count'].sum()} records")
    print(f"Density range: {calculated.min():.6f}–{calculated.max():.6f} per million")


if __name__ == "__main__":
    main()
