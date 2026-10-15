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
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from remote_style_common import (
    add_horizontal_colorbar, add_map_frame, add_map_labels, add_north_arrow, add_scale_bar,
    add_south_china_inset, china_lambert, decode_topojson, draw_base_map, draw_nine_dash,
    load_analysis_panel, load_items, save_figure, set_style,
)


def main():
    set_style()
    items = load_items()
    provinces = decode_topojson()

    fig = plt.figure(figsize=(8.27, 6.05))
    ax = fig.add_axes([0.075, 0.140, 0.840, 0.720])
    draw_base_map(ax, provinces, value_map=None, missing="#FFF7C2", edge="#334155", lw=0.40)
    point_df = items.dropna(subset=["x", "y"]).copy()
    point_x, point_y = china_lambert(point_df["x"].to_numpy(), point_df["y"].to_numpy())
    ax.scatter(point_x, point_y, s=5.8, c="#8B1E1E", alpha=0.72, linewidths=0,
               zorder=6, rasterized=True)
    add_map_labels(ax, provinces, fontsize=6.0)
    draw_nine_dash(ax, lw=0.65)
    add_north_arrow(ax)
    add_scale_bar(ax)
    axins = add_south_china_inset(fig, ax, provinces, value_map=None, cmap="YlOrRd", norm=None)
    inset_items = point_df[
        (point_df["x"].between(106, 125.5)) & (point_df["y"].between(3.0, 25.5))
    ]
    if not inset_items.empty:
        ix, iy = china_lambert(inset_items["x"].to_numpy(), inset_items["y"].to_numpy())
        axins.scatter(ix, iy, s=3.0, c="#8B1E1E", alpha=0.78, linewidths=0,
                      zorder=6, rasterized=True)
    add_map_frame(ax)
    point_handle = mlines.Line2D([], [], color="#8B1E1E", marker="o", linestyle="None",
                                markersize=4.5, label="Folklore ICH record")
    ax.legend(handles=[point_handle], loc="lower left", bbox_to_anchor=(0.070, 0.165),
              frameon=False, borderaxespad=0.0, fontsize=8.0, handletextpad=0.45)
    save_figure(fig, "fig02_folklore_point_map", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()


