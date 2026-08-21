"""Run frequency GLM, severity GLM, pure premium, and holdout analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis import (
    add_base_levels,
    calibration_by_annual_pp,
    data_quality_snapshot,
    one_way_actual_vs_expected,
    relativity_table,
    write_analysis_json,
)
from src.charts import (
    plot_calibration,
    plot_lift_and_lorenz,
    plot_lift_comparison,
    plot_one_ways,
    plot_relativities,
)
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
    out = paths["outputs"]

    freq = add_rating_factors(load_frequency(paths["freq"]))
    sev = load_severity(paths["sev"])
    freq["actual_loss"] = actual_loss_by_policy(freq, sev).to_numpy()
    quality = data_quality_snapshot(freq, sev)

    train_ids, test_ids = train_test_split(
        freq["IDpol"],
        test_size=0.2,
        random_state=42,
        stratify=(freq["ClaimNb"] > 0).astype(int),
    )
    train = freq[freq["IDpol"].isin(train_ids)].copy()
    test = freq[freq["IDpol"].isin(test_ids)].copy()

    freq_model = fit_frequency(train)
    _write_coefs(freq_model, out / "frequency_coefs.csv")

    claims_fit = add_rating_factors(claim_level_severity(freq, sev))
    claims_raw = add_rating_factors(claim_level_severity(freq, sev, cap_quantile=None))
    train_claims = claims_fit[claims_fit["IDpol"].isin(train_ids)]
    sev_model = fit_severity(train_claims)
    _write_coefs(sev_model, out / "severity_coefs.csv")

    test = attach_pure_premium(
        test,
        predict_frequency(freq_model, test),
        predict_severity(sev_model, test),
    )

    gini_policy = gini_coefficient(test["actual_loss"].to_numpy(), test["pure_premium"].to_numpy())
    gini_annual = gini_coefficient(test["actual_loss"].to_numpy(), test["annual_pure_premium"].to_numpy())
    gini_naive = gini_coefficient(test["actual_loss"].to_numpy(), test["Exposure"].to_numpy())

    lift_policy = lift_table(test, score_col="pure_premium")
    lift_annual = lift_table(test, score_col="annual_pure_premium")
    lift_policy.to_csv(out / "lift_table.csv", index=False)
    lift_annual.to_csv(out / "lift_table_annual.csv", index=False)

    rel_freq = add_base_levels(relativity_table(freq_model, "frequency"), train)
    rel_sev = add_base_levels(relativity_table(sev_model, "severity"), train_claims)
    relativities = pd.concat([rel_freq, rel_sev], ignore_index=True)
    relativities.to_csv(out / "relativities.csv", index=False)

    ave = one_way_actual_vs_expected(test, claims_raw)
    ave.to_csv(out / "oneway_ave.csv", index=False)

    cal = calibration_by_annual_pp(test)
    cal.to_csv(out / "calibration_annual_pp.csv", index=False)

    chart_policy = plot_lift_and_lorenz(
        lift_policy,
        gini_policy,
        gini_naive,
        out,
        filename="lift_gini.png",
        lift_title="Lift — ranked by policy expected loss",
        score_label="Decile (1 = highest policy pure premium)",
    )
    chart_annual = plot_lift_and_lorenz(
        lift_annual,
        gini_annual,
        gini_naive,
        out,
        filename="lift_gini_annual.png",
        lift_title="Lift — ranked by annual pure premium",
        score_label="Decile (1 = highest annual pure premium)",
    )
    chart_cmp = plot_lift_comparison(lift_policy, lift_annual, out)
    chart_rel = plot_relativities(relativities, out)
    chart_ow = plot_one_ways(ave, out)
    chart_cal = plot_calibration(cal, out)

    portfolio_loss = float(test["actual_loss"].sum())
    model_prem = float(test["pure_premium"].sum())
    naive_prem = portfolio_loss
    loss_ratio = portfolio_loss / model_prem if model_prem else float("nan")

    metrics = [
        f"train_policies={len(train):,}",
        f"test_policies={len(test):,}",
        f"train_claims={len(train_claims):,}",
        f"frequency_deviance={freq_model.deviance:.1f}",
        f"severity_deviance={sev_model.deviance:.1f}",
        f"test_actual_loss={portfolio_loss:,.0f}",
        f"test_model_premium={model_prem:,.0f}",
        f"test_naive_premium={naive_prem:,.0f}",
        f"test_loss_ratio={loss_ratio:.4f}",
        f"gini_glm_policy_pp={gini_policy:.4f}",
        f"gini_glm_annual_pp={gini_annual:.4f}",
        f"gini_naive_exposure={gini_naive:.4f}",
        f"claimnb_sev_mismatch_policies={quality['policies_claimnb_ne_sev_rows']:,}",
        f"severity_cap_amount={quality['severity_cap_amount']:,.0f}",
        f"chart_policy={chart_policy.relative_to(root).as_posix()}",
        f"chart_annual={chart_annual.relative_to(root).as_posix()}",
        f"chart_comparison={chart_cmp.relative_to(root).as_posix()}",
        f"chart_relativities={chart_rel.relative_to(root).as_posix()}",
        f"chart_calibration={chart_cal.relative_to(root).as_posix()}",
        f"charts_oneway={','.join(p.relative_to(root).as_posix() for p in chart_ow)}",
    ]
    (out / "metrics.txt").write_text("\n".join(metrics) + "\n", encoding="utf-8")

    summary = {
        "quality": quality,
        "holdout": {
            "train_policies": int(len(train)),
            "test_policies": int(len(test)),
            "train_claims": int(len(train_claims)),
            "test_exposure": float(test["Exposure"].sum()),
            "actual_loss": portfolio_loss,
            "model_premium": model_prem,
            "loss_ratio": loss_ratio,
            "gini_policy_pp": gini_policy,
            "gini_annual_pp": gini_annual,
            "gini_naive_exposure": gini_naive,
            "frequency_deviance": float(freq_model.deviance),
            "severity_deviance": float(sev_model.deviance),
        },
        "lift_policy": lift_policy.to_dict(orient="records"),
        "lift_annual": lift_annual.to_dict(orient="records"),
        "calibration": cal.to_dict(orient="records"),
        "relativities": relativities.to_dict(orient="records"),
        "oneway": ave.to_dict(orient="records"),
    }
    write_analysis_json(out / "analysis_summary.json", summary)
    print("\n".join(metrics))


if __name__ == "__main__":
    main()
