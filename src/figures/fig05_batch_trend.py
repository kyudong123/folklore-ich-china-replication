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
from remote_style_common import BATCH_LABELS, BATCH_ORDER, PALETTE, REGION_ORDER, load_items, save_figure, set_style


def main():
    set_style()
    items = load_items()
    source_counts = items.drop_duplicates(["batch", "source_id"]).groupby("batch").size().reindex(BATCH_ORDER).fillna(0).astype(int)
    x = np.arange(len(BATCH_ORDER))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.55))
    bars = ax1.bar(x, source_counts.values, width=0.56,
                   color="#4DBBD5", edgecolor="white", linewidth=0.8)

    ymax = source_counts.max()
    for bar in bars:
        val = int(bar.get_height())
        ax1.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.018,
                 str(val), ha="center", va="bottom", fontsize=9.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([BATCH_LABELS[b].replace(" ", "\n") for b in BATCH_ORDER])
    ax1.set_ylabel("Folklore ICH records")
    ax1.set_ylim(0, ymax * 1.22)
    ax1.grid(axis="y", color=PALETTE["grid"], lw=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    save_figure(fig, "fig05_batch_trend", FIGURE_OUTPUT)


if __name__ == "__main__":
    main()


