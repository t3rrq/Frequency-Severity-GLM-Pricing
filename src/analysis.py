"""Holdout diagnostics on the fitted frequency–severity GLMs (no new model families)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ONE_WAY_FACTORS = [
    "Area",
    "VehPowerBand",
    "VehAgeBand",
    "DrivAgeBand",
    "BonusMalusBand",
    "VehBrand",
    "VehGas",
    "Region",
]

_TERM_RE = re.compile(r"^C\((?P<factor>[^)]+)\)\[T\.(?P<level>.+)\]$")


def relativity_table(result, model_name: str) -> pd.DataFrame:
    """Log-link GLM coefficients as multiplicative relativities vs the treatment reference."""
    rows: list[dict[str, Any]] = []

    for name, coef in result.params.items():
        se = float(result.bse[name])
        z = float(result.tvalues[name])
        p = float(result.pvalues[name])
        degenerate = abs(float(coef)) < 1e-10 or se < 1e-12
        m = _TERM_RE.match(str(name))
        if name == "Intercept":
            rows.append(
                {
                    "model": model_name,
                    "factor": "Intercept",
                    "level": "(base mean, log scale)",
                    "coef": float(coef),
                    "relativity": float(np.exp(coef)),
                    "std_err": se,
                    "z": z,
                    "p_value": p,
                    "is_base": False,
                    "degenerate": degenerate,
                }
            )
            continue
        if name == "LogDensity":
            rows.append(
                {
                    "model": model_name,
                    "factor": "LogDensity",
                    "level": "+1 log(Density)",
                    "coef": float(coef),
                    "relativity": float(np.exp(coef)),
                    "std_err": se,
                    "z": z,
                    "p_value": p,
                    "is_base": False,
                    "degenerate": degenerate,
                }
            )
            continue
        if m:
            factor, level = m.group("factor"), m.group("level")
            rows.append(
                {
                    "model": model_name,
                    "factor": factor,
                    "level": level,
                    "coef": float(coef),
                    "relativity": float(np.exp(coef)),
                    "std_err": se,
                    "z": z,
                    "p_value": p,
                    "is_base": False,
                    "degenerate": degenerate,
                }
            )

    frame = pd.DataFrame(rows)
    return frame


def add_base_levels(rel: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Insert relativity 1.0 rows for each factor's treatment-contrast reference level."""
    extra = []
    for factor in ONE_WAY_FACTORS:
        if factor not in sample.columns:
            continue
        observed = [str(x) for x in sample[factor].dropna().unique()]
        fitted = set(rel.loc[rel["factor"] == factor, "level"].astype(str))
        bases = [lv for lv in observed if lv not in fitted]
        if not bases:
            continue
        # statsmodels Treatment: first category of a pandas Categorical, else sorted unique.
        col = sample[factor]
        if isinstance(col.dtype, pd.CategoricalDtype):
            base = str(col.cat.categories[0])
        else:
            base = sorted(bases)[0]
        extra.append(
            {
                "model": rel["model"].iloc[0] if len(rel) else "",
                "factor": factor,
                "level": base,
                "coef": 0.0,
                "relativity": 1.0,
                "std_err": np.nan,
                "z": np.nan,
                "p_value": np.nan,
                "is_base": True,
                "degenerate": False,
            }
        )
    if not extra:
        return rel
    out = pd.concat([rel, pd.DataFrame(extra)], ignore_index=True)
    return out


def data_quality_snapshot(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    *,
    claim_nb_cap: int = 4,
    exposure_cap: float = 1.0,
    severity_quantile: float = 0.995,
) -> dict[str, Any]:
    n_sev = sev.groupby("IDpol").size()
    n_aligned = n_sev.reindex(freq["IDpol"]).fillna(0).astype(int).to_numpy()
    claimnb = freq["ClaimNb"].to_numpy()
    mismatch = int((claimnb != n_aligned).sum())
    both_positive_mismatch = int(((claimnb > 0) & (n_aligned > 0) & (claimnb != n_aligned)).sum())
    cap = float(sev["ClaimAmount"].quantile(severity_quantile))
    n_capped_sev = int((sev["ClaimAmount"] > cap).sum())
    return {
        "policies": int(len(freq)),
        "exposure_years": float(freq["Exposure"].sum()),
        "claimnb_sum": int(freq["ClaimNb"].sum()),
        "severity_rows": int(len(sev)),
        "policies_claimnb_ne_sev_rows": mismatch,
        "policies_both_positive_count_mismatch": both_positive_mismatch,
        "claim_nb_cap": claim_nb_cap,
        "policies_at_claimnb_cap": int((freq["ClaimNb"] >= claim_nb_cap).sum()),
        "exposure_cap": exposure_cap,
        "severity_cap_quantile": severity_quantile,
        "severity_cap_amount": cap,
        "severity_rows_above_cap": n_capped_sev,
    }


def one_way_actual_vs_expected(test: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Holdout one-ways: observed vs GLM-predicted frequency, severity, and loss."""
    test_ids = set(test["IDpol"])
    claims_test = claims.loc[claims["IDpol"].isin(test_ids)].copy()
    parts = []
    for factor in ONE_WAY_FACTORS:
        grp = test.groupby(factor, observed=True)
        agg = grp.agg(
            policies=("IDpol", "count"),
            exposure=("Exposure", "sum"),
            claimnb=("ClaimNb", "sum"),
            freq_hat=("freq_hat", "sum"),
            actual_loss=("actual_loss", "sum"),
            model_premium=("pure_premium", "sum"),
        ).reset_index()
        agg = agg.rename(columns={factor: "level"})
        agg.insert(0, "factor", factor)

        if len(claims_test) and factor in claims_test.columns:
            sev_obs = (
                claims_test.groupby(factor, observed=True)["ClaimAmount"]
                .agg(n_claims="size", claim_amount="sum")
                .reset_index()
                .rename(columns={factor: "level"})
            )
            agg = agg.merge(sev_obs, on="level", how="left")
        else:
            agg["n_claims"] = np.nan
            agg["claim_amount"] = np.nan

        agg["obs_frequency"] = agg["claimnb"] / agg["exposure"]
        agg["pred_frequency"] = agg["freq_hat"] / agg["exposure"]
        agg["obs_severity"] = agg["claim_amount"] / agg["n_claims"]
        agg["pred_severity"] = agg["model_premium"] / agg["freq_hat"]
        agg["loss_ratio"] = agg["actual_loss"] / agg["model_premium"]
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def calibration_by_annual_pp(test: pd.DataFrame, n_bands: int = 10) -> pd.DataFrame:
    work = test.copy()
    work["band"] = pd.qcut(work["annual_pure_premium"], n_bands, labels=False, duplicates="drop") + 1
    grouped = work.groupby("band", as_index=False).agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        actual_loss=("actual_loss", "sum"),
        model_premium=("pure_premium", "sum"),
        mean_annual_pp=("annual_pure_premium", "mean"),
    )
    grouped["actual_loss_ratio"] = grouped["actual_loss"] / grouped["exposure"]
    grouped["model_avg_pp"] = grouped["model_premium"] / grouped["exposure"]
    grouped["loss_ratio"] = grouped["actual_loss"] / grouped["model_premium"]
    return grouped


def write_analysis_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
