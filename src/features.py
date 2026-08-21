"""Rating-factor engineering used by both GLMs."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORMULA_TERMS = [
    "C(Area)",
    "C(VehPowerBand)",
    "C(VehAgeBand)",
    "C(DrivAgeBand)",
    "C(BonusMalusBand)",
    "C(VehBrand)",
    "C(VehGas)",
    "C(Region)",
    "LogDensity",
]


def add_rating_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["VehPowerBand"] = pd.cut(
        out["VehPower"],
        bins=[0, 4, 5, 6, 7, 9, 15, np.inf],
        labels=["4-", "5", "6", "7", "8-9", "10-15", "16+"],
        right=True,
    )
    out["VehAgeBand"] = pd.cut(
        out["VehAge"],
        bins=[-np.inf, 0, 1, 4, 10, np.inf],
        labels=["0", "1", "2-4", "5-10", "11+"],
    )
    out["DrivAgeBand"] = pd.cut(
        out["DrivAge"],
        bins=[-np.inf, 20, 25, 30, 40, 50, 60, 70, np.inf],
        labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    )
    out["BonusMalusBand"] = pd.cut(
        out["BonusMalus"].clip(upper=150),
        bins=[49, 50, 60, 80, 100, 150],
        labels=["50", "51-60", "61-80", "81-100", "101-150"],
        include_lowest=True,
    )
    out["LogDensity"] = np.log(out["Density"].clip(lower=1))
    return out


def glm_formula(response: str) -> str:
    return f"{response} ~ " + " + ".join(FORMULA_TERMS)
