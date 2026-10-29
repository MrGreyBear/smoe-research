"""LaTeX tables for the extension sections: realised variance (sec:rv), K and gate (sec:kgate),
foundation-model release-date test (sec:lockbox). Safe to re-run; skips missing inputs."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smoe import TABLES  # noqa: E402
from smoe.data import UNIVERSE  # noqa: E402
from smoe.report import to_tex  # noqa: E402

LAB = {k: v.label for k, v in UNIVERSE.items()}


def have(*names):
    return all((TABLES / n).exists() for n in names)


def star(v, p, nd=2, scale=100.0):
    """Signed difference x scale with a star when the one-sided p-value is below 5%."""
    s = f"{v * scale:.{nd}f}"
    return f"${s}^{{*}}$" if p < 0.05 else f"${s}$"


# ------------------------------------------------------------------ realised variance
RV_ORDER = ["SMoE-PK", "SMoE-RV", "SMoE-TV", "SMoE-CJ", "SMoE-SV", "SMoE-PKRV", "Gauss-HAR-PK", "Gauss-HAR-RV",
            "GARCH-t", "RealGARCH-t"]

if have("rv_measurement.csv"):
    m = pd.read_csv(TABLES / "rv_measurement.csv")
    est = m[m["omega2"].notna()][["estimator", "omega2", "attenuation"]].copy()
    est.columns = ["Daily estimator", "$\\omega^2=\\Var\\eta_t$", "$\\nu_1/(\\nu_1+\\omega^2)$"]
    nu1, b5, b22 = m["nu1"].dropna().iloc[0], m["b5"].iloc[0], m["b22"].iloc[0]
    pr = m.set_index("estimator")["ratio"]
    to_tex(est, "rv_measure", "Measurement error of daily variance estimators and the implied attenuation of the "
           "daily HAR loading (CSI 300, development sample)", "tab:rv-measure", nd=3,
           notes=f"$\\omega^2$ for squared returns and RV from the $\\chi^2$ laws in the text; Parkinson (BM) by "
                 f"simulation of 20,000 Brownian paths with 2,400 steps; Parkinson (implied) from the residual "
                 f"variance of the daily range on its weekly and monthly averages minus $\\nu_1$. "
                 f"$\\nu_1={nu1:.3f}$ is the residual variance of daily log RV on its weekly and monthly averages "
                 f"minus $\\omega^2_{{\\mathrm{{RV}}}}$; projection coefficients $b_5={b5:.3f}$, $b_{{22}}={b22:.3f}$. "
                 f"Predicted $\\delta_1^{{\\mathrm{{PK}}}}/\\delta_1^{{\\mathrm{{RV}}}}$ from \\eqref{{eq:eiv-ratio}}: "
                 f"{pr['predicted ratio (BM)']:.2f} (BM) and {pr['predicted ratio (implied)']:.2f} (implied); "
                 f"estimated in the $K=1$ model: {pr['observed Gauss[single]']:.2f}.")

if have("rv_losses.csv", "rv_dm.csv"):
    L = pd.read_csv(TABLES / "rv_losses.csv").set_index(["asset", "model"])
    D = pd.read_csv(TABLES / "rv_dm.csv").set_index(["asset", "model", "loss"])
    rows = []
    for mdl in RV_ORDER:
        r = {"Model": mdl}
        for a, loss, lab in [("csi300", "NLL", "CSI NLL"), ("csi300", "QLIKE", "CSI QLIKE"), ("csi300", "FZ5", "CSI FZ5"),
                             ("if_main", "NLL", "IF NLL"), ("etf510300", "NLL", "ETF NLL")]:
            if mdl == "SMoE-PK":
                r[lab] = f"{L.loc[(a, mdl), loss]:.4f}"
            else:
                d = D.loc[(a, mdl, loss)]
                r[lab] = star(-d["mean_diff"], d["p_model_better"])
        r["QLIKE$_{\\mathrm{RV}}$"] = L.loc[("csi300", mdl), "QLIKE_RV"]
        r["MZ $R^2_{\\mathrm{RV}}$"] = L.loc[("csi300", mdl), "MZ_R2_RV"]
        r["Kupiec 1\\%"] = int(sum(L.loc[(a, mdl), "kupiec1_p"] >= 0.05 for a in ("csi300", "if_main", "etf510300")))
        rows.append(r)
    n = {a: int(L.loc[(a, "SMoE-PK"), "n"]) for a in ("csi300", "if_main", "etf510300")}
    to_tex(pd.DataFrame(rows), "rv_oos", "Realised-variance covariates out of sample (development period)",
           "tab:rv-oos", nd=3, size="\\footnotesize",
           notes=f"First row: SMoE-PK losses. Other rows in loss columns: loss of the model minus loss of SMoE-PK, "
                 f"$\\times100$ (negative = model better); $^*$: one-sided DM-HLN $p<0.05$ in favour of the model. "
                 f"QLIKE$_{{\\mathrm{{RV}}}}$ and MZ $R^2_{{\\mathrm{{RV}}}}$: CSI 300 variance forecasts against "
                 f"$\\mathrm{{RV}}_{{t+1}}+r^2_{{\\text{{overnight}},t+1}}$. Kupiec 1\\%: series (of 3) on which the 1\\% "
                 f"VaR passes at 5\\%. Common samples: CSI 300 $n={n['csi300']}$, IF $n={n['if_main']}$, "
                 f"ETF $n={n['etf510300']}$.")

if have("rv_insample_lik.csv"):
    k = pd.read_csv(TABLES / "rv_insample_lik.csv")
    k = k[k["model"].str.startswith(("SMoE", "Gauss"))].copy()
    t = pd.DataFrame({"Model": k["model"], "Log-lik.": k["loglik"], "Params": k["params"].astype(int),
                      "BIC": k["BIC"],
                      "$W$ vol. in gate": [f"{w:.2f} ({p:.3f})" if np.isfinite(w) else "--"
                                           for w, p in zip(k["W_gate_vol"], k["p_gate_vol"])],
                      "$W$ mom.+dd in gate": [f"{w:.2f} ({p:.3f})" if np.isfinite(w) else "--"
                                              for w, p in zip(k["W_gate_momdd"], k["p_gate_momdd"])]})
    sv = k.set_index("model").loc["SMoE-SV"]
    to_tex(t, "rv_insample", "In-sample fits with range-based and realised variance covariates, CSI 300 "
           "(June 2009--June 2024, days with 5-minute data)", "tab:rv-insample", nd=1,
           notes=f"Robust Wald statistics ($p$-values) that the volatility state (2 df) and the momentum and "
                 f"drawdown states (4 df) are absent from the gate. Equality of the negative and positive "
                 f"semivariance loadings in all three experts of SMoE-SV: $W={sv['W_semi_equal']:.2f}$ "
                 f"(3 df, $p={sv['p_semi_equal']:.2f}$).")

if have("rv_lemma.csv"):
    e = pd.read_csv(TABLES / "rv_lemma.csv")
    e = e.set_index("model").loc[RV_ORDER].reset_index()
    t = pd.DataFrame({"Model": e["model"], "$\\Var(e_t)$": e["var_e"],
                      "Predicted $\\Delta$NLL": e["pred_dNLL_var"] * 100, "Observed $\\Delta$NLL": e["obs_dNLL"] * 100})
    to_tex(t, "rv_lemma", "Log-score gains predicted by \\cref{lem:logvar-cost} from the accuracy of the variance "
           "forecast, CSI 300", "tab:rv-lemma", nd=3,
           notes="$e_t=\\log(\\mathrm{RV}_{t+1}+r^2_{\\text{overnight},t+1})-\\log\\hat V_t$, with $\\hat V_t$ the "
                 "forecast variance. Predicted $\\Delta$NLL $=\\{\\Var(e_t)-\\Var(e_t^{\\text{SMoE-PK}})\\}/4$; observed "
                 "$\\Delta$NLL is the out-of-sample log-score difference from SMoE-PK. Both $\\times100$, nats per day; "
                 "negative = better than SMoE-PK. The error of the proxy is common to all models and cancels.")

# ------------------------------------------------------------------ K and gate
KG_ORDER = ["Gauss-HAR", "SMoE-K2tc", "SMoE-K2rc", "SMoE-K2tr", "LSTAR", "SMoE", "MAR", "SMoE-K4c", "SMoE-K4p",
            "SMoE-K5", "SMoE-Gvol", "SMoE-Gvix", "SMoE-Gasym", "SMoE-Grich"]

if have("kg_losses.csv", "kg_dm.csv", "kg_mcs.csv"):
    L = pd.read_csv(TABLES / "kg_losses.csv")
    D = pd.read_csv(TABLES / "kg_dm.csv")
    M = pd.read_csv(TABLES / "kg_mcs.csv")
    ranks = {loss: L.pivot(index="asset", columns="model", values=loss).rank(axis=1).mean() for loss in
             ("NLL", "CRPS", "QLIKE", "FZ5")}
    rows = []
    for mdl in KG_ORDER:
        if mdl not in ranks["NLL"].index:
            continue
        r = {"Model": mdl, "$K$": int(L.loc[L.model == mdl, "K"].iloc[0])}
        for loss in ("NLL", "CRPS", "QLIKE", "FZ5"):
            r[f"Rank {loss}"] = ranks[loss][mdl]
        if mdl != "SMoE":
            d = D[(D.model == mdl) & (D.loss == "NLL")]
            r["NLL better"] = int(((d.p_model_better < 0.05)).sum())
            r["NLL worse"] = int(((d.p_two < 0.05) & (d.mean_diff < 0)).sum())
        else:
            r["NLL better"], r["NLL worse"] = "--", "--"
        for loss in ("NLL", "QLIKE"):
            r[f"MCS {loss}"] = int(M[(M.model == mdl) & (M.loss == loss)]["in_mcs"].sum())
        r["Kupiec 1\\%"] = int((L[L.model == mdl]["kupiec1_p"] >= 0.05).sum())
        rows.append(r)
    to_tex(pd.DataFrame(rows), "kg_oos", "Number of experts and gate covariates: out-of-sample comparison on 19 "
           "series (development period)", "tab:kg-oos", nd=2, size="\\footnotesize",
           notes="Average ranks among the 14 specifications (1 = best). NLL better/worse: series on which the "
                 "specification is significantly better (one-sided DM-HLN, 5\\%) or worse (two-sided, 5\\%) than "
                 "the SMoE in log score. MCS: series (of 19) on which the specification is in the 10\\% model "
                 "confidence set of the 14. Kupiec 1\\%: series on which the 1\\% VaR passes at 5\\%.")

if have("kg_insample.csv"):
    I = pd.read_csv(TABLES / "kg_insample.csv")
    best = I.loc[I.groupby("asset")["BIC"].idxmin()]
    bestK = I[I.model.isin(["Gauss-HAR", "SMoE-K2tc", "SMoE-K2rc", "SMoE-K2tr", "SMoE", "SMoE-K4c", "SMoE-K4p",
                            "SMoE-K5"])]
    bestK = bestK.loc[bestK.groupby("asset")["BIC"].idxmin()]
    t = pd.DataFrame({"Asset": [LAB.get(a, a) for a in best.asset], "BIC-best (all 14)": best.model.values,
                      "BIC-best $K$-spec.": bestK.set_index("asset").loc[best.asset, "model"].values,
                      "$\\Delta$BIC SMoE": (I[I.model == "SMoE"].set_index("asset").loc[best.asset, "BIC"].values
                                            - best.BIC.values)})
    to_tex(t, "kg_bic", "In-sample model selection by BIC (development sample)", "tab:kg-bic", nd=1,
           size="\\footnotesize", notes="$\\Delta$BIC SMoE: BIC of the baseline SMoE minus the smallest BIC "
                                        "(0 when the SMoE is selected).")

if have("kg_lm_gate.csv", "kg_insample_extra.csv"):
    lm = pd.read_csv(TABLES / "kg_lm_gate.csv")
    ex = pd.read_csv(TABLES / "kg_insample_extra.csv").set_index("asset")
    piv = lm.pivot(index="asset", columns="candidate", values="p")
    cols = ["VIX", "last return", "1-day variance", "VIX + return + 1d var"]
    t = piv[cols].copy()
    t["Wald VIX $p$"] = ex.loc[t.index, "vix_p"]
    t["premium: $\\beta_{\\mathrm{lv22}}$ ($z$)"] = [f"{b:.3f} ({z:.1f})" for b, z in
                                                    zip(ex.loc[t.index, "prem_lv22"], ex.loc[t.index, "prem_lv22_z"])]
    t = t.reindex([a for a in LAB if a in t.index])
    t.index = [LAB.get(a, a) for a in t.index]
    t = t.reset_index().rename(columns={"index": "Asset"})
    t.columns = ["Asset", "LM: VIX", "LM: return", "LM: 1d var.", "LM: all three", "Wald VIX $p$",
                 "premium: $\\beta_{\\mathrm{lv22}}$ ($z$)"]
    to_tex(t, "kg_lm", "Screening gate variables: robust LM $p$-values from the baseline fit "
           "(\\cref{prop:lm-gate}), development sample", "tab:kg-lm", nd=3, size="\\footnotesize",
           notes="LM: $p$-value of the score test that the candidate is absent from the gate, computed from the "
                 "baseline SMoE alone ($2d$ df). Wald VIX $p$: robust Wald test of the same hypothesis for VIX from "
                 "the augmented fit (SMoE-Gvix). Premium: standardised loading of the premium expert's mean on "
                 "log 22-day variance in SMoE-K4p, with robust $z$-statistic.")

if have("kg_lr_bootstrap.csv"):
    b = pd.read_csv(TABLES / "kg_lr_bootstrap.csv")
    t = pd.DataFrame({"Series": [LAB.get(a, a) for a in b.asset], "Test": b.test, "LR": b.LR, "$B$": b.B,
                      "Bootstrap 95\\% crit.": b.boot_q95, "Bootstrap mean": b.boot_mean, "$p$": b.boot_p})
    to_tex(t, "kg_boot", "Parametric bootstrap of the likelihood-ratio test for the number of experts",
           "tab:kg-boot", nd=3,
           notes="Fixed-design parametric bootstrap \\citep{mclachlan1987bootstrapping}: $y^\\ast$ drawn from the "
                 "fitted null model at the observed covariates; both models re-estimated on every draw (best of a "
                 "warm start and two random starts). $p=(1+\\#\\{LR^\\ast\\ge LR\\})/(1+B)$.")

# ------------------------------------------------------------------ foundation-model release-date test
if have("fm_release_test.csv"):
    f = pd.read_csv(TABLES / "fm_release_test.csv")
    t = pd.DataFrame({"Model": f.model, "Comparison": f.series, "Release": f.release, "$n_{\\text{pre}}$": f.n_pre,
                      "$n_{\\text{post}}$": f.n_post, "Mean pre": f.mean_pre, "Mean post": f.mean_post,
                      "$\\hat b$": f.b, "$z$": f.z, "$p$ ($b>0$)": f.p_one_sided})
    to_tex(t, "fm_release", "Release-date test for pretraining overlap (log score relative to the SMoE, "
           "averaged across series)", "tab:fm-release", nd=3, size="\\footnotesize",
           notes="$d_t$: cross-series average of NLL(model) $-$ NLL(SMoE) on date $t$, development and lockbox "
                 "periods. $\\hat b$: post-release minus pre-release mean, with Newey--West standard errors. "
                 "Memorisation of the evaluation period predicts $b>0$ for the foundation model. Placebo: GARCH-$t$, "
                 "estimated in real time. Difference: foundation model minus placebo (difference-in-differences).")
print("extension tables written")
