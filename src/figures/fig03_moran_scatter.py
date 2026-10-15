# -*- coding: utf-8 -*-
"""Plot the manuscript Moran scatterplot from audited Queen row-standardized LISA inputs."""
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
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from remote_style_common import PROVINCE_EN, PALETTE, save_figure, set_style


def main():
    set_style()
    path = PACKAGE_DIR / "results" / "spatial" / "06_local_moran_province_results.csv"
    data = pd.read_csv(path, encoding="utf-8-sig")
    data = data[(data["indicator"] == "crude rate") & (data["weights"] == "Queen row-standardized")].copy()
    if len(data) != 31 or data["province"].nunique() != 31:
        raise ValueError("Expected 31 unique provinces for Queen row-standardized crude-rate results")
    data["spatial_lag_z"] = data["spatial_lag_z"].fillna(0.0)  # Hainan is a Queen isolate (zero row).
    slope = float(np.dot(data["z_value"], data["spatial_lag_z"]) / np.dot(data["z_value"], data["z_value"]))
    moran_path = PACKAGE_DIR / "results" / "spatial" / "05_global_moran_sensitivity.csv"
    moran_table = pd.read_csv(moran_path, encoding="utf-8-sig")
    match = moran_table[(moran_table["indicator"] == "crude rate") & (moran_table["weights"] == "Queen row-standardized")]
    if len(match) != 1:
        raise ValueError("Could not identify the single crude-rate Queen row-standardized global Moran result")
    moran_i = float(match.iloc[0]["moran_I"])
    expected_global_i = slope * len(data) / 30.0  # S0=30 because Hainan is the single zero-row isolate.
    if abs(expected_global_i - moran_i) > 0.002:
        raise ValueError(f"Moran scatter slope {slope:.6f} implies global Moran's I {expected_global_i:.6f}, not {moran_i:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    colors = []
    for _, row in data.iterrows():
        x, y = float(row["z_value"]), float(row["spatial_lag_z"])
        if row["province"] == "海南":
            colors.append("#8A8F98")
        elif x >= 0 and y >= 0:
            colors.append("#D55E00")
        elif x < 0 and y < 0:
            colors.append("#0072B2")
        elif x >= 0 and y < 0:
            colors.append("#E69F00")
        else:
            colors.append("#56B4E9")
    ax.scatter(data["z_value"], data["spatial_lag_z"], s=55, c=colors, edgecolor="white", linewidth=0.65, zorder=4)
    xline = np.linspace(data["z_value"].min() - 0.2, data["z_value"].max() + 0.2, 100)
    ax.plot(xline, slope * xline, color="#111827", lw=1.5)
    ax.axhline(0, color="#6B7280", lw=0.85, ls="--")
    ax.axvline(0, color="#6B7280", lw=0.85, ls="--")
    label_offsets = {"西藏": (8, 0), "青海": (8, 3), "新疆": (10, -18), "福建": (8, -12)}
    for province, offset in label_offsets.items():
        row = data[data["province"] == province]
        if not row.empty:
            ax.annotate(PROVINCE_EN[province], (float(row["z_value"].iloc[0]), float(row["spatial_lag_z"].iloc[0])),
                        xytext=offset, textcoords="offset points", fontsize=8.4,
                        arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.45, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Standardized folklore ICH density")
    ax.set_ylabel("Spatial lag")
    ax.grid(True, color=PALETTE["grid"], lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.495, 0.965, f"Moran's I = {moran_i:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10.2)
    handles = [mpatches.Patch(color=color, label=label) for color, label in [
        ("#D55E00", "High-High"), ("#0072B2", "Low-Low"), ("#E69F00", "High-Low"),
        ("#56B4E9", "Low-High"), ("#8A8F98", "Isolate")]]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save_figure(fig, "fig03_moran_scatter", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()


