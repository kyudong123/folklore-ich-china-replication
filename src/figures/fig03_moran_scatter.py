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
    permutation_p = float(match.iloc[0]["permutation_p_two_sided"])
    permutations = int(match.iloc[0]["permutations"])
    expected_global_i = slope * len(data) / 30.0  # S0=30 because Hainan is the single zero-row isolate.
    if abs(expected_global_i - moran_i) > 0.002:
        raise ValueError(f"Moran scatter slope {slope:.6f} implies global Moran's I {expected_global_i:.6f}, not {moran_i:.6f}")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(data["z_value"], data["spatial_lag_z"], s=16, color="#2878B5",
               edgecolor="white", linewidth=0.55, zorder=4)
    xline = np.linspace(data["z_value"].min() - 0.2, data["z_value"].max() + 0.2, 100)
    ax.plot(xline, slope * xline, color="#111827", lw=1.5)
    ax.axhline(0, color="#6B7280", lw=0.85, ls="--")
    ax.axvline(0, color="#6B7280", lw=0.85, ls="--")
    label_offsets = {
        "西藏": (8, 0), "青海": (8, 3), "新疆": (10, -18),
        "四川": (-24, -11), "云南": (-27, -11), "甘肃": (-24, -8),
    }
    for province, offset in label_offsets.items():
        row = data[data["province"] == province]
        if not row.empty:
            ax.annotate(PROVINCE_EN[province], (float(row["z_value"].iloc[0]), float(row["spatial_lag_z"].iloc[0])),
                        xytext=offset, textcoords="offset points", fontsize=8.4,
                        arrowprops=dict(arrowstyle="-", color="#6B7280", lw=0.45, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Standardized provincial listing density")
    ax.set_ylabel("Spatial lag of standardized listing density")
    ax.grid(True, color=PALETTE["grid"], lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.305, 0.965,
            f"Moran's I = {moran_i:.3f}; {permutations:,}-permutation p = {permutation_p:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2)
    save_figure(fig, "fig03_moran_scatter", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()


