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

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from remote_style_common import (
    add_map_frame, add_map_labels, add_north_arrow, add_scale_bar, add_south_china_inset,
    decode_topojson, draw_base_map, draw_nine_dash, load_analysis_panel, local_lisa, make_adjacency,
    save_figure, set_style,
)


def main():
    set_style()
    package_dir = Path(__file__).resolve().parents[2]
    local_path = package_dir / "results" / "spatial" / "06_local_moran_province_results.csv"
    local = pd.read_csv(local_path, encoding="utf-8-sig")
    local = local.loc[(local["indicator"] == "crude rate") & (local["weights"] == "Queen row-standardized")]
    if len(local) != 31:
        raise ValueError("Expected 31 local Moran results for the primary Queen specification")
    provinces = decode_topojson()
    clusters = dict(zip(local["province"], local["cluster_p_lt_0_05_nominal"]))
    color_map = {"HH": "#D55E00", "LL": "#0072B2", "HL": "#E69F00", "LH": "#56B4E9", "NS": "#D1D5DB"}
    code_map = {"NS": 0, "LL": 1, "LH": 2, "HL": 3, "HH": 4}

    fig = plt.figure(figsize=(8.27, 6.05))
    ax = fig.add_axes([0.075, 0.200, 0.840, 0.660])
    lisa_values = {k: code_map.get(v, 0) for k, v in clusters.items()}
    lisa_cmap = mcolors.ListedColormap(["#D1D5DB", "#0072B2", "#56B4E9", "#E69F00", "#D55E00"])
    lisa_norm = mcolors.Normalize(vmin=0, vmax=4)
    draw_base_map(ax, provinces, lisa_values, cmap=lisa_cmap, norm=lisa_norm,
                  edge="#334155", lw=0.40, missing="#FFFFFF")
    add_map_labels(ax, provinces, fontsize=5.8)
    draw_nine_dash(ax, lw=0.65)
    add_north_arrow(ax)
    add_scale_bar(ax)
    add_south_china_inset(fig, ax, provinces, lisa_values, cmap=lisa_cmap, norm=lisa_norm)
    add_map_frame(ax)
    handles = [
        mpatches.Patch(facecolor=color_map[k], edgecolor="none", label=lab)
        for k, lab in [("HH", "High-High"), ("LL", "Low-Low"), ("HL", "High-Low"),
                       ("LH", "Low-High"), ("NS", "Not significant")]
    ]
    handles.append(mpatches.Patch(facecolor="#FFFFFF", edgecolor="#64748B",
                                  linewidth=0.45, label="Not analyzed"))
    fig.text(0.5, 0.125, "Exploratory LISA classes (nominal p < .05; unadjusted)",
             ha="center", va="center", fontsize=7.2)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=6, frameon=False, fontsize=6.8, columnspacing=1.05,
               handlelength=1.0, handletextpad=0.35)
    save_figure(fig, "supp_fig_s1_exploratory_lisa_map", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()


