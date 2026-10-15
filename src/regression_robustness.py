# -*- coding: utf-8 -*-
"""Regression robustness analysis using the harmonized 31-province folklore panel."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA = PACKAGE_DIR / "data" / "processed" / "provincial_analysis_panel_2020.csv"
TABLE_DIR = PACKAGE_DIR / "results" / "tables"
FIGURE_DIR = PACKAGE_DIR / "results" / "figures"


def save_csv(df, name):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    needed = {"province", "pop_2020", "minority_share_2020", "folklore_count", "folklore_per_million"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["log_pop_offset"] = np.log(df["pop_2020"])
    df["minority_share_10pp"] = df["minority_share_2020"] * 10.0
    df["minority_share"] = df["minority_share_2020"]
    if len(df) != 31 or df["province"].nunique() != 31:
        raise ValueError(f"Expected 31 unique mainland provinces; got n={len(df)}, unique={df['province'].nunique()}")
    if (df["folklore_count"] < 0).any() or (df["pop_2020"] <= 0).any():
        raise ValueError("Counts must be nonnegative and population positive")
    save_csv(df, "01_analysis_panel_used.csv")

    # Core models: rate OLS with HC3 inference; count Poisson with log(population) offset;
    # NB2 count model with the same offset as a dispersion-sensitive comparison.
    ols = smf.ols("folklore_per_million ~ minority_share", data=df).fit(cov_type="HC3", use_t=True)
    rlm = smf.rlm("folklore_per_million ~ minority_share", data=df,
                  M=sm.robust.norms.HuberT()).fit(cov="H1")
    pois = smf.glm(
        "folklore_count ~ minority_share_10pp", data=df,
        family=sm.families.Poisson(), offset=df["log_pop_offset"]
    ).fit(cov_type="HC3", use_t=True)
    dispersion = float(pois.pearson_chi2 / pois.df_resid)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nb = smf.negativebinomial(
            "folklore_count ~ minority_share_10pp", data=df,
            offset=df["log_pop_offset"]
        ).fit(disp=False, cov_type="HC0", use_t=True, maxiter=500)
    # An additional quasi-Poisson-style scale correction is not used for selection; report dispersion explicitly.
    ols_inf = ols.get_influence()
    infl = pd.DataFrame({
        "province": df["province"],
        "fitted_density": ols.fittedvalues,
        "residual": ols.resid,
        "studentized_external_resid": ols_inf.resid_studentized_external,
        "leverage": ols_inf.hat_matrix_diag,
        "cooks_distance": ols_inf.cooks_distance[0],
        "dffits": ols_inf.dffits[0],
    })
    p = int(ols.df_model + 1)
    infl["high_leverage_2p_over_n"] = infl["leverage"] > (2 * p / len(df))
    infl["cooks_gt_4_over_n"] = infl["cooks_distance"] > (4 / len(df))
    save_csv(infl.sort_values("cooks_distance", ascending=False), "02_ols_influence_diagnostics.csv")

    # Pre-specified influence sensitivity: all 31; omit Tibet; omit Qinghai; omit both.
    loo_rows = []
    scenarios = [("All 31 provinces", []) , ("Exclude Tibet", ["西藏"]),
                 ("Exclude Qinghai", ["青海"]), ("Exclude Tibet and Qinghai", ["西藏", "青海"])]
    for label, dropped in scenarios:
        sub = df.loc[~df.province.isin(dropped)].copy()
        fit = smf.ols("folklore_per_million ~ minority_share", data=sub).fit(cov_type="HC3", use_t=True)
        loo_rows.append({"scenario": label, "n": len(sub), "dropped": ";".join(dropped),
                         "slope_per_1_share_unit": fit.params["minority_share"],
                         "density_change_per_10pp": fit.params["minority_share"] * .1,
                         "HC3_se_slope": fit.bse["minority_share"],
                         "p_value_HC3": fit.pvalues["minority_share"],
                         "ci95_low_slope": fit.conf_int().loc["minority_share", 0],
                         "ci95_high_slope": fit.conf_int().loc["minority_share", 1],
                         "R2": fit.rsquared})
        pfit = smf.glm(
            "folklore_count ~ minority_share_10pp", data=sub,
            family=sm.families.Poisson(), offset=sub["log_pop_offset"]
        ).fit(cov_type="HC3", use_t=True)
        nbfit = smf.negativebinomial(
            "folklore_count ~ minority_share_10pp", data=sub,
            offset=sub["log_pop_offset"]
        ).fit(disp=False, cov_type="HC0", use_t=True, maxiter=500)
        rfit = smf.rlm("folklore_per_million ~ minority_share", data=sub,
                       M=sm.robust.norms.HuberT()).fit(cov="H1")
        loo_rows[-1].update({
            "Huber_RLM_density_change_per_10pp": float(rfit.params["minority_share"] * .1),
            "Huber_RLM_p": float(rfit.pvalues["minority_share"]),
            "poisson_IRR_per_10pp": float(np.exp(pfit.params["minority_share_10pp"])),
            "poisson_IRR_ci95_low": float(np.exp(pfit.conf_int().loc["minority_share_10pp", 0])),
            "poisson_IRR_ci95_high": float(np.exp(pfit.conf_int().loc["minority_share_10pp", 1])),
            "poisson_HC3_p": float(pfit.pvalues["minority_share_10pp"]),
            "NB2_IRR_per_10pp": float(np.exp(nbfit.params["minority_share_10pp"])),
            "NB2_IRR_ci95_low": float(np.exp(nbfit.conf_int().loc["minority_share_10pp", 0])),
            "NB2_IRR_ci95_high": float(np.exp(nbfit.conf_int().loc["minority_share_10pp", 1])),
            "NB2_HC0_p": float(nbfit.pvalues["minority_share_10pp"]),
            "NB2_alpha": float(nbfit.params.get("alpha", np.nan)),
            "NB2_converged": bool(nbfit.mle_retvals.get("converged", False)),
        })
    loo = pd.DataFrame(loo_rows)
    save_csv(loo, "03_ols_tibet_qinghai_sensitivity.csv")

    # Full model comparison table with interpretable effect scales and uncertainty.
    ols_b = float(ols.params["minority_share"] * .1)
    ols_ci = ols.conf_int().loc["minority_share"].to_numpy() * .1
    rlm_b = float(rlm.params["minority_share"] * .1)
    rlm_ci = rlm.conf_int().loc["minority_share"].to_numpy() * .1
    p_b = float(pois.params["minority_share_10pp"])
    p_ci = pois.conf_int().loc["minority_share_10pp"].to_numpy()
    nb_b = float(nb.params["minority_share_10pp"])
    nb_ci = nb.conf_int().loc["minority_share_10pp"].to_numpy()
    rows = [
        {"model": "OLS rate, HC3", "effect_scale": "Density difference per +10 percentage points", "estimate": ols_b,
         "std_error": float(ols.bse["minority_share"] * .1), "ci95_low": ols_ci[0], "ci95_high": ols_ci[1],
         "p_value": float(ols.pvalues["minority_share"]), "n": int(ols.nobs), "R2_or_pseudoR2": ols.rsquared,
         "AIC": np.nan, "dispersion": np.nan},
        {"model": "Huber M-estimator rate (RLM)", "effect_scale": "Density difference per +10 percentage points", "estimate": rlm_b,
         "std_error": float(rlm.bse["minority_share"] * .1), "ci95_low": rlm_ci[0], "ci95_high": rlm_ci[1],
         "p_value": float(rlm.pvalues["minority_share"]), "n": int(rlm.nobs), "R2_or_pseudoR2": np.nan,
         "AIC": np.nan, "dispersion": np.nan},
        {"model": "Poisson count, population offset, HC3", "effect_scale": "Incidence-rate ratio per +10 percentage points", "estimate": np.exp(p_b),
         "std_error": float(pois.bse["minority_share_10pp"]), "ci95_low": np.exp(p_ci[0]), "ci95_high": np.exp(p_ci[1]),
         "p_value": float(pois.pvalues["minority_share_10pp"]), "n": int(pois.nobs), "R2_or_pseudoR2": np.nan,
         "AIC": pois.aic, "dispersion": dispersion},
        {"model": "Negative binomial NB2 count, population offset, HC0", "effect_scale": "Incidence-rate ratio per +10 percentage points", "estimate": np.exp(nb_b),
         "std_error": float(nb.bse["minority_share_10pp"]), "ci95_low": np.exp(nb_ci[0]), "ci95_high": np.exp(nb_ci[1]),
         "p_value": float(nb.pvalues["minority_share_10pp"]), "n": int(nb.nobs), "R2_or_pseudoR2": np.nan,
         "AIC": nb.aic, "dispersion": np.nan},
    ]
    models = pd.DataFrame(rows)
    save_csv(models, "04_model_comparison.csv")

    # Machine-readable full coefficient summaries for auditing/reproduction.
    coef_rows = []
    for mname, fit, var in [("OLS_rate_HC3", ols, "minority_share"),
                            ("Huber_RLM_rate", rlm, "minority_share"),
                            ("Poisson_offset_HC3", pois, "minority_share_10pp"),
                            ("NB2_offset_HC0", nb, "minority_share_10pp")]:
        ci = fit.conf_int().loc[var]
        coef_rows.append({"model": mname, "coefficient": fit.params[var], "std_error": fit.bse[var],
                          "statistic": fit.tvalues[var] if hasattr(fit, "tvalues") else fit.tvalues[var],
                          "p_value": fit.pvalues[var], "ci95_low": ci.iloc[0], "ci95_high": ci.iloc[1],
                          "n": int(fit.nobs), "AIC": getattr(fit, "aic", np.nan)})
    save_csv(pd.DataFrame(coef_rows), "05_coefficient_details.csv")

    # Figure A: effect estimates on their distinct, correctly interpretable scales.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.errorbar([ols_b], [1], xerr=[[ols_b-ols_ci[0]], [ols_ci[1]-ols_b]], fmt="o", capsize=4, color="#2166AC")
    ax.errorbar([rlm_b], [0], xerr=[[rlm_b-rlm_ci[0]], [rlm_ci[1]-rlm_b]], fmt="o", capsize=4, color="#7B3294")
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks([0, 1], ["Huber RLM", "OLS (HC3)"])
    ax.set_xlabel("Change in listings per million residents\nper +10 percentage points minority share")
    ax.set_title("Rate OLS")
    ax.grid(axis="x", alpha=.2)
    ax = axes[1]
    vals = [np.exp(p_b), np.exp(nb_b)]
    cis = [np.exp(p_ci), np.exp(nb_ci)]
    labels = ["Poisson (HC3)", "Negative binomial (HC0)"]
    for i, (v, ci) in enumerate(zip(vals, cis)):
        ax.errorbar([v], [i], xerr=[[v-ci[0]], [ci[1]-v]], fmt="o", capsize=4,
                    color=["#D95F02", "#1B9E77"][i])
    ax.axvline(1, color="#777777", lw=1, ls="--")
    ax.set_yticks(range(2), labels)
    ax.set_xlabel("Incidence-rate ratio (95% CI)\nper +10 percentage points minority share")
    ax.set_title("Count models, log(population) offset")
    ax.grid(axis="x", alpha=.2)
    fig.suptitle("Association estimates across rate and count specifications", fontsize=12)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "06_model_effect_comparison.png", dpi=300, bbox_inches="tight")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "06_model_effect_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    # Figure B: leverage and studentized residuals; label provinces that appear in the checks.
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.scatter(infl["leverage"], infl["studentized_external_resid"], s=40, color="#2878B5", edgecolor="white")
    ax.axhline(0, color="#666666", lw=.8)
    ax.axhline(2, color="#B2182B", lw=.8, ls="--")
    ax.axhline(-2, color="#B2182B", lw=.8, ls="--")
    ax.axvline(2*p/len(df), color="#B2182B", lw=.8, ls=":", label="2p/n leverage reference")
    province_en = {"西藏": "Tibet", "青海": "Qinghai", "新疆": "Xinjiang"}
    for _, r in infl.iterrows():
        if r["province"] in province_en or r["cooks_gt_4_over_n"]:
            ax.annotate(province_en.get(r["province"], r["province"]), (r["leverage"], r["studentized_external_resid"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("OLS leverage (hat value)")
    ax.set_ylabel("Externally studentized residual")
    ax.set_title("OLS influence diagnostics (31 mainland provinces)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.18)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "07_ols_influence_diagnostics.png", dpi=300, bbox_inches="tight")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "07_ols_influence_diagnostics.svg", bbox_inches="tight")
    plt.close(fig)

    # Plain-text report, including exact data provenance and convergence flags.
    with (TABLE_DIR / "08_run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Regression robustness analysis: harmonized 31-province folklore panel\n")
        f.write(f"Input: {DATA.relative_to(ROOT)}\nN={len(df)}; sum(count)={int(df.folklore_count.sum())}; population={int(df.pop_2020.sum())}\n")
        f.write(f"Poisson Pearson dispersion={dispersion:.4f}\n")
        f.write(f"OLS R2={ols.rsquared:.4f}; slope per +10pp={ols_b:.6f}; HC3 p={ols.pvalues['minority_share']:.6g}\n")
        f.write(f"Huber RLM slope per +10pp={rlm_b:.6f}; 95% CI=({rlm_ci[0]:.6f}, {rlm_ci[1]:.6f}); p={rlm.pvalues['minority_share']:.6g}\n")
        f.write(f"Poisson IRR per +10pp={np.exp(p_b):.4f}; 95% CI=({np.exp(p_ci[0]):.4f}, {np.exp(p_ci[1]):.4f}); HC3 p={pois.pvalues['minority_share_10pp']:.6g}\n")
        f.write(f"NB2 alpha={nb.params.get('alpha', np.nan):.6f}; IRR per +10pp={np.exp(nb_b):.4f}; 95% CI=({np.exp(nb_ci[0]):.4f}, {np.exp(nb_ci[1]):.4f}); HC0 p={nb.pvalues['minority_share_10pp']:.6g}; converged={nb.mle_retvals.get('converged', 'unknown')}\n")
        f.write("\nInfluence flags (descriptive thresholds, not automatic exclusion rules):\n")
        f.write(infl.loc[infl.high_leverage_2p_over_n | infl.cooks_gt_4_over_n,
                         ["province", "leverage", "studentized_external_resid", "cooks_distance", "high_leverage_2p_over_n", "cooks_gt_4_over_n"]].to_string(index=False))
        f.write("\n\nTibet/Qinghai sensitivity:\n")
        f.write(loo.to_string(index=False))
        if caught:
            f.write("\n\nNB warnings:\n" + "\n".join(str(x.message) for x in caught))
    summary_path = TABLE_DIR / "08_run_summary.txt"
    summary = summary_path.read_text(encoding="utf-8")
    province_names = {"北京":"Beijing", "天津":"Tianjin", "河北":"Hebei", "山西":"Shanxi", "内蒙古":"Inner Mongolia", "辽宁":"Liaoning", "吉林":"Jilin", "黑龙江":"Heilongjiang", "上海":"Shanghai", "江苏":"Jiangsu", "浙江":"Zhejiang", "安徽":"Anhui", "福建":"Fujian", "江西":"Jiangxi", "山东":"Shandong", "河南":"Henan", "湖北":"Hubei", "湖南":"Hunan", "广东":"Guangdong", "广西":"Guangxi", "海南":"Hainan", "重庆":"Chongqing", "四川":"Sichuan", "贵州":"Guizhou", "云南":"Yunnan", "西藏":"Tibet", "陕西":"Shaanxi", "甘肃":"Gansu", "青海":"Qinghai", "宁夏":"Ningxia", "新疆":"Xinjiang"}
    for name_zh, name_en in province_names.items():
        summary = summary.replace(name_zh, name_en)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()
