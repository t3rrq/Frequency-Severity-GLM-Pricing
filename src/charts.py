"""Lift / Lorenz plots for the technical premium vs a flat rate."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lift_and_lorenz(lift: pd.DataFrame, gini: float, naive_gini: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lift_gini.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(lift["decile"], lift["lift"], color="#1f4e79", width=0.7)
    axes[0].axhline(1.0, color="#888888", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Decile (1 = highest predicted premium)")
    axes[0].set_ylabel("Actual loss / portfolio average")
    axes[0].set_title("Lift chart — model vs flat rate")
    axes[0].set_xticks(lift["decile"])

    # Lorenz: cumulative loss vs cumulative exposure, ordered by predicted PP
    x = np.concatenate([[0.0], lift["cum_exposure_share"].to_numpy()])
    y = np.concatenate([[0.0], lift["cum_loss_share"].to_numpy()])
    axes[1].plot(x, y, color="#1f4e79", label=f"GLM rank (Gini {gini:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#888888", linestyle="--", label=f"Random / flat (Gini {naive_gini:.3f})")
    axes[1].set_xlabel("Cumulative exposure share")
    axes[1].set_ylabel("Cumulative actual loss share")
    axes[1].set_title("Lorenz curve of risk ranking")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
