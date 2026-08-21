"""Gamma GLM for positive claim size given a claim occurred."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.features import glm_formula


def claim_level_severity(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """One row per claim, with the policy's rating factors attached."""
    claims = sev.merge(freq.drop(columns=["ClaimNb", "Exposure"]), on="IDpol", how="inner")
    claims = claims.loc[claims["ClaimAmount"] > 0].copy()
    # Cap the most extreme tails so a handful of large losses do not dominate Gamma MLE.
    cap = claims["ClaimAmount"].quantile(0.995)
    claims["ClaimAmount"] = claims["ClaimAmount"].clip(upper=cap)
    return claims


def fit_severity(train_claims: pd.DataFrame):
    formula = glm_formula("ClaimAmount")
    model = smf.glm(
        formula=formula,
        data=train_claims,
        family=sm.families.Gamma(sm.families.links.Log()),
    )
    return model.fit(maxiter=100)


def predict_severity(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame))
