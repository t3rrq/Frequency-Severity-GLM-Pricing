"""Combine frequency and severity into a technical pure premium and rank risks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_pure_premium(
    policies: pd.DataFrame,
    freq_hat: np.ndarray,
    sev_hat: np.ndarray,
) -> pd.DataFrame:
    out = policies.copy()
    out["freq_hat"] = freq_hat
    out["sev_hat"] = sev_hat
    out["pure_premium"] = out["freq_hat"] * out["sev_hat"]
    out["annual_pure_premium"] = out["pure_premium"] / out["Exposure"]
    return out


def actual_loss_by_policy(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.Series:
    loss = sev.groupby("IDpol", as_index=True)["ClaimAmount"].sum()
    return loss.reindex(freq["IDpol"]).fillna(0.0)


def gini_coefficient(loss: np.ndarray, score: np.ndarray) -> float:
    """Gini of actual loss when policies are ordered by predicted score (higher = riskier)."""
    order = np.argsort(-np.asarray(score))
    y = np.asarray(loss, dtype=float)[order]
    if y.sum() <= 0:
        return 0.0
    n = y.size
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * y) / np.sum(y) - (n + 1)) / n)


def lift_table(frame: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    work = frame.sort_values("pure_premium", ascending=False).copy()
    work["decile"] = pd.qcut(np.arange(len(work)), n_deciles, labels=False) + 1
    naive_rate = work["actual_loss"].sum() / work["Exposure"].sum()
    work["naive_premium"] = naive_rate * work["Exposure"]

    grouped = work.groupby("decile", as_index=False).agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        actual_loss=("actual_loss", "sum"),
        model_premium=("pure_premium", "sum"),
        naive_premium=("naive_premium", "sum"),
    )
    grouped["actual_loss_ratio"] = grouped["actual_loss"] / grouped["exposure"]
    grouped["model_avg_pp"] = grouped["model_premium"] / grouped["exposure"]
    overall = work["actual_loss"].sum() / work["Exposure"].sum()
    grouped["lift"] = grouped["actual_loss_ratio"] / overall
    grouped["cum_exposure_share"] = grouped["exposure"].cumsum() / grouped["exposure"].sum()
    grouped["cum_loss_share"] = grouped["actual_loss"].cumsum() / grouped["actual_loss"].sum()
    return grouped
