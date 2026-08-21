"""Run the three scoped deliverables: frequency GLM, severity GLM, pure premium + lift/Gini."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.charts import plot_lift_and_lorenz
from src.data import ensure_raw_files, load_frequency, load_severity
from src.features import add_rating_factors
from src.frequency import fit_frequency, predict_frequency
from src.premium import actual_loss_by_policy, attach_pure_premium, gini_coefficient, lift_table
from src.severity import claim_level_severity, fit_severity, predict_severity


def _write_coefs(result, path: Path) -> None:
    params = result.params.rename("coef").to_frame()
    params["std_err"] = result.bse
    params["z"] = result.tvalues
    params["p_value"] = result.pvalues
    params.to_csv(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = ensure_raw_files(root)

    freq = add_rating_factors(load_frequency(paths["freq"]))
    sev = load_severity(paths["sev"])
    freq["actual_loss"] = actual_loss_by_policy(freq, sev).to_numpy()

    train_ids, test_ids = train_test_split(
        freq["IDpol"],
        test_size=0.2,
        random_state=42,
        stratify=(freq["ClaimNb"] > 0).astype(int),
    )
    train = freq[freq["IDpol"].isin(train_ids)].copy()
    test = freq[freq["IDpol"].isin(test_ids)].copy()

    freq_model = fit_frequency(train)
    _write_coefs(freq_model, paths["outputs"] / "frequency_coefs.csv")

    claims = add_rating_factors(claim_level_severity(freq, sev))
    train_claims = claims[claims["IDpol"].isin(train_ids)]
    sev_model = fit_severity(train_claims)
    _write_coefs(sev_model, paths["outputs"] / "severity_coefs.csv")

    test = attach_pure_premium(
        test,
        predict_frequency(freq_model, test),
        predict_severity(sev_model, test),
    )

    gini_model = gini_coefficient(test["actual_loss"].to_numpy(), test["pure_premium"].to_numpy())
    gini_naive = gini_coefficient(test["actual_loss"].to_numpy(), test["Exposure"].to_numpy())
    lift = lift_table(test)
    lift.to_csv(paths["outputs"] / "lift_table.csv", index=False)
    chart_path = plot_lift_and_lorenz(lift, gini_model, gini_naive, paths["outputs"])

    portfolio_loss = test["actual_loss"].sum()
    model_prem = test["pure_premium"].sum()
    naive_prem = (portfolio_loss / test["Exposure"].sum()) * test["Exposure"].sum()

    metrics = [
        f"train_policies={len(train):,}",
        f"test_policies={len(test):,}",
        f"train_claims={len(train_claims):,}",
        f"frequency_deviance={freq_model.deviance:.1f}",
        f"severity_deviance={sev_model.deviance:.1f}",
        f"test_actual_loss={portfolio_loss:,.0f}",
        f"test_model_premium={model_prem:,.0f}",
        f"test_naive_premium={naive_prem:,.0f}",
        f"gini_glm={gini_model:.4f}",
        f"gini_naive_exposure={gini_naive:.4f}",
        f"chart={chart_path}",
    ]
    (paths["outputs"] / "metrics.txt").write_text("\n".join(metrics) + "\n", encoding="utf-8")
    print("\n".join(metrics))


if __name__ == "__main__":
    main()
