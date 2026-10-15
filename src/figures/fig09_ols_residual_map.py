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
import matplotlib.pyplot as plt
from remote_style_common import (
    CENTROID_OVERRIDE, PROVINCE_EN, add_horizontal_colorbar, add_map_frame, add_map_labels,
    add_north_arrow, add_scale_bar, add_south_china_inset, decode_topojson,
    draw_base_map, draw_nine_dash, load_analysis_panel, ols_fit, save_figure, set_style,
)


def main():
    set_style()
    panel = load_analysis_panel()
    provinces = decode_topojson()
    x = panel["minority_share"].astype(float).to_numpy()
    y = panel["festival_per_million"].astype(float).to_numpy()
    _, _, _, residual, _ = ols_fit(x, y)
    panel["residual_std"] = (residual - residual.mean()) / residual.std(ddof=1)
    values = dict(zip(panel["province"], panel["residual_std"].astype(float)))
    vals = np.array(list(values.values()), dtype=float)
    lim = max(abs(vals.min()), abs(vals.max()))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    fig = plt.figure(figsize=(8.27, 6.05))
    ax = fig.add_axes([0.075, 0.140, 0.840, 0.720])
    draw_base_map(ax, provinces, values, cmap=cmap, norm=norm, edge="#334155", lw=0.40)
    add_map_labels(ax, provinces, fontsize=5.8)
    draw_nine_dash(ax, lw=0.65)
    add_north_arrow(ax)
    add_scale_bar(ax)
    add_south_china_inset(fig, ax, provinces, values, cmap=cmap, norm=norm)
    add_map_frame(ax)
    add_horizontal_colorbar(fig, ax, cmap, norm, "Standardized OLS residual")
    save_figure(fig, "fig09_ols_residual_map", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

