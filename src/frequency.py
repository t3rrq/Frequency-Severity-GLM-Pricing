"""Poisson GLM for claim counts with exposure offset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.features import glm_formula


def fit_frequency(train: pd.DataFrame):
    formula = glm_formula("ClaimNb")
    model = smf.glm(
        formula=formula,
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"].to_numpy()),
    )
    return model.fit(maxiter=100)


def predict_frequency(model, frame: pd.DataFrame) -> np.ndarray:
    """Expected claim count over the observed exposure (not annualized)."""
    mu = model.predict(frame, offset=np.log(frame["Exposure"].to_numpy()))
    return np.asarray(mu)
