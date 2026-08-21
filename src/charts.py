"""Lift / Lorenz plots and holdout diagnostic charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
_NAVY = "#1f4e79"
_TEAL = "#2a9d8f"
_GRAY = "#888888"
_FACTOR_ORDER = {
    "VehPowerBand": ["4-", "5", "6", "7", "8-9", "10-15", "16+"],
    "VehAgeBand": ["0", "1", "2-4", "5-10", "11+"],
    "DrivAgeBand": ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    "BonusMalusBand": ["50", "51-60", "61-80", "81-100", "101-150"],
}


def plot_lift_and_lorenz(
    lift: pd.DataFrame,
    gini: float,
    naive_gini: float,
    out_dir: Path,
    *,
    filename: str = "lift_gini.png",
    lift_title: str = "Lift chart — model vs flat rate",
    score_label: str = "Decile (1 = highest predicted premium)",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(lift["decile"], lift["lift"], color=_NAVY, width=0.7)
    axes[0].axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    axes[0].set_xlabel(score_label)
    axes[0].set_ylabel("Actual loss / portfolio average")
    axes[0].set_title(lift_title)
    axes[0].set_xticks(lift["decile"])

    x = np.concatenate([[0.0], lift["cum_exposure_share"].to_numpy()])
    y = np.concatenate([[0.0], lift["cum_loss_share"].to_numpy()])
    axes[1].plot(x, y, color=_NAVY, label=f"GLM rank (Gini {gini:.3f})")
    axes[1].plot(
        [0, 1],
        [0, 1],
        color=_GRAY,
        linestyle="--",
        label=f"Equality (Gini {naive_gini:.3f} on exposure rank)",
    )
    axes[1].set_xlabel("Cumulative exposure share")
    axes[1].set_ylabel("Cumulative actual loss share")
    axes[1].set_title("Lorenz curve of risk ranking")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lift_comparison(policy_lift: pd.DataFrame, annual_lift: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "lift_ranking_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, 11)
    w = 0.38
    ax.bar(x - w / 2, policy_lift["lift"], width=w, color=_NAVY, label="Ranked by policy expected loss")
    ax.bar(x + w / 2, annual_lift["lift"], width=w, color=_TEAL, label="Ranked by annual pure premium")
    ax.axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Decile (1 = highest predicted)")
    ax.set_ylabel("Actual loss / portfolio average")
    ax.set_title("Lift by ranking score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_relativities(rel: pd.DataFrame, out_dir: Path) -> Path:
    factors = ["BonusMalusBand", "DrivAgeBand", "VehAgeBand"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, factor in zip(axes, factors):
        sub = rel.loc[rel["factor"] == factor].copy()
        order = _FACTOR_ORDER.get(factor)
        if order:
            sub["_ord"] = sub["level"].map({lv: i for i, lv in enumerate(order)})
            sub = sub.sort_values("_ord")
        freq = sub.loc[sub["model"] == "frequency"]
        sev = sub.loc[sub["model"] == "severity"]
        x = np.arange(len(freq))
        labels = freq["level"].astype(str).tolist()
        w = 0.38
        ax.bar(x - w / 2, freq["relativity"], width=w, color=_NAVY, label="Frequency")
        ax.bar(x + w / 2, sev["relativity"], width=w, color=_TEAL, label="Severity")
        ax.axhline(1.0, color=_GRAY, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(factor)
        ax.set_ylabel("Relativity vs base")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("GLM relativities (exp of log-link coefficients)", y=1.02)
    fig.tight_layout()
    path = out_dir / "relativities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_one_ways(ave: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for factor in ["BonusMalusBand", "DrivAgeBand"]:
        sub = ave.loc[ave["factor"] == factor].copy()
        order = _FACTOR_ORDER.get(factor)
        if order:
            sub["_ord"] = sub["level"].map({lv: i for i, lv in enumerate(order)})
            sub = sub.sort_values("_ord")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(sub))
        labels = sub["level"].astype(str).tolist()
        w = 0.38
        axes[0].bar(x - w / 2, sub["obs_frequency"], width=w, color=_NAVY, label="Observed")
        axes[0].bar(x + w / 2, sub["pred_frequency"], width=w, color=_TEAL, label="GLM predicted")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Claims / exposure year")
        axes[0].set_title(f"Frequency — {factor}")
        axes[0].legend(frameon=False)

        axes[1].bar(x - w / 2, sub["actual_loss"] / 1e6, width=w, color=_NAVY, label="Actual loss")
        axes[1].bar(x + w / 2, sub["model_premium"] / 1e6, width=w, color=_TEAL, label="Model premium")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("EUR millions")
        axes[1].set_title(f"Loss vs premium — {factor}")
        axes[1].legend(frameon=False)

        fig.tight_layout()
        path = out_dir / f"oneway_{factor.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_calibration(cal: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "calibration_annual_pp.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        cal["model_avg_pp"],
        cal["actual_loss_ratio"],
        "o-",
        color=_NAVY,
        label="Holdout actual loss / exposure",
    )
    lo = min(cal["model_avg_pp"].min(), cal["actual_loss_ratio"].min())
    hi = max(cal["model_avg_pp"].max(), cal["actual_loss_ratio"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=_GRAY, label="Perfect calibration")
    ax.set_xlabel("Model pure premium / exposure (EUR)")
    ax.set_ylabel("Actual loss / exposure (EUR)")
    ax.set_title("Holdout calibration by annual pure-premium band")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
