"""Convert result CSVs into the LaTeX tables included by the paper. Safe to re-run; skips missing inputs."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smoe import TABLES  # noqa: E402
from smoe.data import UNIVERSE  # noqa: E402
from smoe.report import to_tex  # noqa: E402

LAB = {k: v.label for k, v in UNIVERSE.items()}


def have(name):
    return (TABLES / name).exists()


if have("descriptives.csv"):
    d = pd.read_csv(TABLES / "descriptives.csv")
    t = d[["asset", "start", "N", "mean", "sd", "skew", "exkurt", "rho1", "rho1_abs", "LB10_r_p", "ARCH5_p"]].copy()
    t.columns = ["Asset", "Start", "N", "Mean", "SD", "Skew", "ExKurt", "$\\rho_1(r)$", "$\\rho_1(|r|)$",
                 "LB(10) $p$", "ARCH(5) $p$"]
    to_tex(t, "desc", "Descriptive statistics of daily log returns (\\%), January 2005 (or inception) to June 2024",
           "tab:desc", nd=3, size="\\scriptsize",
           notes="LB(10): Ljung--Box test of no autocorrelation in returns; ARCH(5): Engle LM test. "
                 "Every series rejects normality (Jarque--Bera $p<10^{-10}$) and every series has ADF $p<0.001$ for returns.")

if have("lagging_paradox.csv"):
    d = pd.read_csv(TABLES / "lagging_paradox.csv")
    t = d[["asset", "N", "RMSE_RW", "RMSE_AR60", "ratio", "sum_coef", "corr_pred_lag", "diffDA_RW", "sign_persist",
           "diffDA_AR60", "properDA_AR60"]].copy()
    t.columns = ["Asset", "N", "RMSE RW", "RMSE AR60", "Ratio", "$\\sum\\hat\\phi_j$", "corr$(\\hat P_{t+1},P_t)$",
                 "DA$^{\\Delta}$ RW", "Sign persist.", "DA$^{\\Delta}$ AR60", "DA AR60"]
    to_tex(t, "lagging", "The lagging paradox: price-level forecasts out of sample (2012--June 2024)", "tab:lagging",
           nd=4, size="\\scriptsize",
           notes="RW: random walk $\\hat P_{t+1}=P_t$. AR60: OLS of $P_{t+1}$ on 60 lagged price levels, re-estimated "
                 "yearly. Ratio $>1$ means AR60 is worse than the random walk. DA$^{\\Delta}$ compares the change in the "
                 "forecast path with the change in the price path; for the RW it equals sign persistence exactly "
                 "(\\cref{prop:da}). DA compares $\\hat P_{t+1}-P_t$ with $P_{t+1}-P_t$.")

if have("eval_losses.csv"):
    L = pd.read_csv(TABLES / "eval_losses.csv")
    order = ["HistGauss", "GARCH-t", "AR-GJR-t", "Gauss-HAR", "MAR", "LSTAR", "MS-AR", "XGB-Gauss", "Chronos-Bolt-small",
             "TimesFM-2.5", "Deep-SMoE", "SMoE"]
    for loss, cap in [("NLL", "negative log score"), ("CRPS", "CRPS"), ("QLIKE", "QLIKE (proxy $y^2$)"),
                      ("FZ5", "FZ0 loss for (VaR, ES) at 5\\%")]:
        piv = L.pivot(index="asset", columns="model", values=loss)
        piv = piv[[c for c in order if c in piv.columns]]
        piv = piv.reindex([a for a in LAB if a in piv.index])
        rel = piv.div(piv["GARCH-t"], axis=0) if loss != "NLL" else piv.sub(piv["GARCH-t"], axis=0)
        rel.index = [LAB.get(a, a) for a in rel.index]
        rel = rel.reset_index().rename(columns={"index": "Asset"})
        how = "difference from" if loss == "NLL" else "ratio to"
        to_tex(rel, f"loss_{loss}", f"Out-of-sample {cap}: {how} GARCH-$t$ (lower is better)", f"tab:loss_{loss}",
               nd=4 if loss == "NLL" else 3, size="\\scriptsize", col_fmt="l" + "r" * (rel.shape[1] - 1))

if have("insample_csi300_params.csv"):
    p = pd.read_csv(TABLES / "insample_csi300_params.csv", index_col=0)
    t = p.reset_index().rename(columns={"index": "Parameter"})
    t["Parameter"] = "\\texttt{" + t["Parameter"].str.replace("_", "\\_") + "}"
    t = t[["Parameter", "estimate", "se_sandwich", "se_mle", "z", "p"]]
    t.columns = ["Parameter", "Estimate", "SE (sandwich)", "SE (model)", "$z$", "$p$"]
    to_tex(t, "csi300_params", "SMoE estimates for CSI 300, January 2005--June 2024", "tab:csi300_params", nd=3,
           size="\\scriptsize", notes="$z$ and $p$ use the sandwich standard errors with Newey--West meat.")

if have("insample_csi300_wald.csv"):
    w = pd.read_csv(TABLES / "insample_csi300_wald.csv")
    w.columns = ["Null hypothesis", "Wald", "df", "$p$"]
    to_tex(w, "csi300_wald", "Robust Wald tests of semantic restrictions, CSI 300", "tab:csi300_wald", nd=3)

if have("insample_csi300_lik.csv"):
    w = pd.read_csv(TABLES / "insample_csi300_lik.csv")
    to_tex(w, "csi300_lik", "In-sample likelihood comparison, CSI 300", "tab:csi300_lik", nd=1,
           bold_min_cols=("AIC", "BIC"))

print("tables written to", TABLES / "tex")

if have("eval_csi300_rv.csv"):
    r = pd.read_csv(TABLES / "eval_csi300_rv.csv")
    r = r[["model", "n", "QLIKE_RV", "MSE_RV", "MZ_b", "MZ_R2"]]
    r.columns = ["Model", "N", "QLIKE", "MSE", "MZ slope", "MZ $R^2$"]
    to_tex(r, "rv", "CSI 300 variance forecasts against realised variance from 5-minute data (plus squared overnight return)",
           "tab:rv", nd=3, bold_min_cols=("QLIKE", "MSE"), bold_max_cols=("MZ $R^2$",))

if have("insample_fingerprints.csv"):
    f = pd.read_csv(TABLES / "insample_fingerprints.csv")
    t = pd.DataFrame({"Asset": f["label"], "LR vs $K{=}1$": f["LR_vs_K1"], "Gate Wald": f["W_gate"],
                      "$\\bar\\pi_{\\text{trend}}$": f["pi_trend"], "$\\bar\\pi_{\\text{crisis}}$": f["pi_crisis"],
                      "$t$(mom$_{120}$)": f["beta[trend]:mom120_t"], "$t$(mom$_{20}$)": f["beta[trend]:mom20_t"],
                      "$t$(rev$_1$)": f["beta[reversal]:rev1_t"], "$t$(rev$_5$)": f["beta[reversal]:rev5_t"],
                      "$t(\\gamma_{\\text{trend,vol}})$": f["gamma[trend]:g_lvol_t"]})
    to_tex(t, "fingerprints", "White-box fingerprints: SMoE estimated on each series, 2005 (or inception) to June 2024",
           "tab:fingerprints", nd=2, size="\\scriptsize",
           notes="Robust (sandwich, Newey--West) $t$-statistics. Gate Wald: test that all gate slopes are zero (6 df).")

if have("insample_csi300_events.csv"):
    e = pd.read_csv(TABLES / "insample_csi300_events.csv")
    e.columns = ["Date", "Event", "trend (pre)", "reversal (pre)", "crisis (pre)", "trend (post)", "reversal (post)",
                 "crisis (post)"]
    to_tex(e, "events", "CSI 300 gate weights in the 20 trading days before and after market and policy events",
           "tab:events", nd=3, size="\\scriptsize", col_fmt="llrrrrrr")

if have("econ_main.csv"):
    e = pd.read_csv(TABLES / "econ_main.csv")
    keep = ["SMoE", "Deep-SMoE", "GARCH-t", "MAR", "Gauss-HAR", "HistGauss", "Chronos-Bolt-small"]
    for tr, cap in [("etf510300", "CSI 300 ETF (510300), long-only, 3bp"), ("if_main", "IF futures, long/short, 2bp")]:
        x = e[(e.traded == tr) & e.model.isin(keep)].copy()
        x = x[["model", "rule", "ann_ret", "ann_vol", "sharpe", "sharpe_se", "max_dd", "avg_w", "turnover_ann",
               "excess_ci_lo", "excess_ci_hi"]]
        x.columns = ["Model", "Rule", "Return", "Vol", "Sharpe", "SE", "MaxDD", "Avg $w$", "Turnover",
                     "Excess CI lo", "Excess CI hi"]
        b = e[e.traded == tr].iloc[0]
        to_tex(x.sort_values(["Rule", "Sharpe"], ascending=[True, False]), f"econ_{tr}",
               f"Trading results: {cap}, out of sample", f"tab:econ_{tr}", nd=3, size="\\scriptsize",
               col_fmt="llrrrrrrrrr",
               notes=f"Buy and hold: Sharpe {b['bench_sharpe']:.3f}, maximum drawdown {b['bench_max_dd']:.3f}. "
                     "Annualised; decisions at the close, execution at the next open. Excess CI: 95\\% stationary-bootstrap "
                     "interval of the annualised mean return in excess of buy and hold.")

if have("econ_overfitting.csv"):
    o = pd.read_csv(TABLES / "econ_overfitting.csv")
    o = o[["n_trials", "best_config", "best_sharpe", "DSR_all", "best_config_3bp", "best_sharpe_3bp", "DSR", "PBO_3bp"]]
    for c in ("best_config", "best_config_3bp"):
        o[c] = o[c].str.replace("|", ", ", regex=False)
    o.columns = ["Trials", "Best (all costs)", "Sharpe (all)", "DSR (all)", "Best at 3bp", "Sharpe (3bp)",
                 "DSR (3bp)", "PBO (3bp)"]
    to_tex(o, "overfit", "Backtest overfitting diagnostics", "tab:overfit", nd=3, col_fmt="rp{3.5cm}rrp{3.5cm}rrr",
           notes="DSR: deflated Sharpe ratio, the probability that the true Sharpe ratio exceeds the expected maximum of "
                 "the trials under the null; values below 0.95 are not significant. PBO: probability of backtest "
                 "overfitting by combinatorially symmetric cross-validation with 16 blocks.")

if have("conformal_summary.csv"):
    c = pd.read_csv(TABLES / "conformal_summary.csv")
    c = c[c.model.isin(["SMoE", "GARCH-t", "Gauss-HAR", "Deep-SMoE", "HistGauss"])]
    c.columns = ["Model", "$a$", "Calibration", "Mean rate", "Max $|$rate$-a|$", "Kupiec reject", "Christoffersen reject"]
    to_tex(c, "conformal", "VaR violations before (raw) and after adaptive conformal calibration (ACI, $\\eta=0.005$)",
           "tab:conformal", nd=4, size="\\scriptsize",
           notes="Rates averaged over 19 series; reject columns: share of series rejecting at 5\\%.")

if have("mc_summary.csv") and have("mc_tests.csv"):
    s = pd.read_csv(TABLES / "mc_summary.csv")
    s.columns = ["$n$", "Mean $|$bias$|$", "Mean RMSE", "Coverage (sandwich)", "Coverage (model)"]
    to_tex(s, "mc_summary", "Monte Carlo: estimator accuracy and 95\\% interval coverage (averaged over 27 parameters)",
           "tab:mc_summary", nd=3)
    t = pd.read_csv(TABLES / "mc_tests.csv")
    t.columns = ["$n$", "DGP", "Reject 5\\%", "Reject 10\\%", "$R$"]
    to_tex(t, "mc_tests", "Monte Carlo: size and power of the robust Wald test of gate relevance", "tab:mc_tests", nd=3,
           col_fmt="rlrrr")

if have("yardstick.csv"):
    y = pd.read_csv(TABLES / "yardstick.csv")
    g = y.groupby("teacher").agg(n_assets=("asset", "size"), gain=("LS_teacher", lambda s: 0),
                                 Phi_median=("Phi", "median"), R2_mean=("R2_mean", "median"),
                                 R2_logsd=("R2_logsd", "median"), R2_probup=("R2_probup", "median"),
                                 KL=("KL", "median"), AG_sig=("AG_p", lambda p: int((p < 0.05).sum())))
    g["gain"] = (y.assign(g=y.LS_teacher - y.LS_reference).groupby("teacher")["g"].median())
    g = g.reset_index()
    g.columns = ["Teacher", "Series", "Median LS gain", "Median $\\Phi$", "$R^2$ mean", "$R^2$ log var",
                 "$R^2$ P(up)", "Median KL", "AG reject"]
    to_tex(g, "yardstick", "The white-box yardstick: fidelity of SMoE shadows of black-box forecasters (out of sample)",
           "tab:yardstick", nd=3, size="\\scriptsize",
           notes="LS gain: teacher log score minus historical-Gaussian log score. $\\Phi$: share of that gain reproduced by "
                 "the SMoE shadow. AG reject: number of series on which the Amisano--Giacomini test rejects equal "
                 "log scores of teacher and shadow at 5\\%.")
print("extra tables written")

if have("ablation_summary.csv"):
    a = pd.read_csv(TABLES / "ablation_summary.csv")
    a.columns = ["Variant", "Loss", "Median loss difference", "Full SMoE significantly better", "Full SMoE significantly worse"]
    to_tex(a, "ablation", "Out-of-sample ablations of the SMoE on eight series (variant minus full model; positive = variant worse)",
           "tab:ablation", nd=4, col_fmt="llrrr",
           notes="Series: CSI 300, CSI 500, SSE 50, ChiNext, S\\&P 500, Hang Seng, COMEX gold, Bitcoin. Counts of series with "
                 "one-sided DM-HLN $p<0.05$ in favour of the full model and two-sided $p<0.05$ against it.")

if have("leak_anatomy.csv"):
    k = pd.read_csv(TABLES / "leak_anatomy.csv")
    k = k[["mechanism", "target", "n", "hit_rate", "PT_p", "R2_oos", "longonly_sharpe", "RMSE", "diffDA"]]
    k.columns = ["Mechanism", "Target", "N", "Hit rate", "PT $p$", "OOS $R^2$", "Long-only Sharpe", "RMSE",
                 "DA$^{\\Delta}$"]
    to_tex(k, "leak", "Anatomy of a leak: the two look-ahead mechanisms of the undergraduate thesis reproduced on CSI 300 "
                      "(out of sample, 2012--June 2024)", "tab:leak", nd=4, size="\\scriptsize", col_fmt="llrrrrrrr",
           notes="Upper block: gradient boosting on 10-day windows of the causal features, with and without the 3-day "
                 "forward mean return (lagged one day, as in the original code) among the inputs. Lower block: OLS on "
                 "60 lagged price levels, with and without the residual correction $0.6\\cdot\\mathrm{MA}_3(P_t-\\hat P_t)$. "
                 "Long-only Sharpe ignores costs.")

# ---- development vs lockbox summary
if have("eval_losses.csv") and have("eval_losses_lockbox.csv"):
    rows = []
    for per, sfx in [("Development 2012--2024/06", ""), ("Lockbox 2024/07--2026/10", "_lockbox")]:
        Lp = pd.read_csv(TABLES / f"eval_losses{sfx}.csv")
        Mp = pd.read_csv(TABLES / f"eval_mcs{sfx}.csv")
        Tp = pd.read_csv(TABLES / f"eval_tests{sfx}.csv")
        ranks = {loss: Lp.pivot(index="asset", columns="model", values=loss).rank(axis=1).mean() for loss in
                 ("NLL", "CRPS", "QLIKE", "FZ5")}
        mcs = Mp.groupby(["model", "loss"])["in_mcs"].sum().unstack()
        kup = Tp.groupby("model")["kupiec1_p"].apply(lambda p: int((p > 0.05).sum()))
        for mname in ["SMoE", "Deep-SMoE", "MAR", "LSTAR", "GARCH-t", "AR-GJR-t", "Gauss-HAR", "MS-AR", "XGB-Gauss",
                      "Chronos-Bolt-base", "TimesFM-2.5", "HistGauss"]:
            rows.append({"Period": per, "Model": mname, **{f"Rank {k}": ranks[k][mname] for k in ranks},
                         "MCS NLL": int(mcs.loc[mname, "NLL"]), "MCS QLIKE": int(mcs.loc[mname, "QLIKE"]),
                         "MCS FZ5": int(mcs.loc[mname, "FZ5"]), "Kupiec 1\\% pass": kup[mname]})
    s = pd.DataFrame(rows)
    to_tex(s, "dev_vs_lockbox", "Development period versus lockbox: average ranks across 19 series (1 = best), "
                                "model-confidence-set membership (10\\%) and Kupiec passes", "tab:lockbox", nd=2,
           size="\\scriptsize", col_fmt="llrrrrrrrr",
           notes="Ranks among the thirteen forecasters (Chronos-Bolt-small not shown). MCS and Kupiec columns count series "
                 "out of 19. TimesFM-2.5 was added after the lockbox had been opened (\\cref{sec:lockbox}).")
    if have("econ_overfitting_lockbox.csv"):
        o = pd.read_csv(TABLES / "econ_overfitting_lockbox.csv")
        o = o[["best_config_3bp", "best_sharpe_3bp", "DSR", "PBO_3bp"]]
        o["best_config_3bp"] = o["best_config_3bp"].str.replace("|", ", ", regex=False)
        o.columns = ["Best at 3bp", "Sharpe", "DSR", "PBO"]
        to_tex(o, "overfit_lockbox", "Lockbox: backtest overfitting diagnostics (CSI 300 ETF, 360 configurations at 3bp)",
               "tab:overfit_lockbox", nd=3, col_fmt="p{6cm}rrr")

# ---- per-asset parameter tables
if (TABLES / "params").exists():
    pblocks = []
    for a in LAB:
        f = TABLES / "params" / f"{a}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f, index_col=0).reset_index().rename(columns={"index": "Parameter"})
        p["Parameter"] = "\\texttt{" + p["Parameter"].str.replace("_", "\\_") + "}"
        p = p[["Parameter", "estimate", "se_sandwich", "se_mle", "z", "p"]]
        p.columns = ["Parameter", "Estimate", "SE (sandwich)", "SE (model)", "$z$", "$p$"]
        to_tex(p, f"params_{a}", f"SMoE estimates: {LAB[a]} (development sample)", f"tab:params_{a}", nd=3,
               size="\\scriptsize")
        pblocks.append(f"\\input{{../results/tables/tex/params_{a}}}")
    (TABLES / "tex" / "params_inputs.tex").write_text("\n".join(pblocks), encoding="utf-8")

# ---- per-asset appendix tables
if have("eval_losses.csv") and have("eval_tests.csv") and have("eval_mcs.csv"):
    Lx = pd.read_csv(TABLES / "eval_losses.csv")
    Tx = pd.read_csv(TABLES / "eval_tests.csv")
    Mx = pd.read_csv(TABLES / "eval_mcs.csv")
    blocks = []
    for a in LAB:
        if a not in set(Lx.asset):
            continue
        l = Lx[Lx.asset == a].set_index("model")
        t = Tx[Tx.asset == a].set_index("model")
        m = Mx[(Mx.asset == a) & (Mx.loss == "NLL")].set_index("model")["mcs_p"]
        tab = pd.DataFrame({"NLL": l["NLL"], "CRPS": l["CRPS"], "QLIKE": l["QLIKE"], "FZ5": l["FZ5"], "Brier": l["Brier"],
                            "MCS $p$ (NLL)": m, "Hit": t["hit_rate"], "Viol 1\\%": t["viol1"], "Viol 5\\%": t["viol5"],
                            "Kupiec 1\\% $p$": t["kupiec1_p"], "Berkowitz $p$": t["berk_p"]})
        tab = tab.reindex([c for c in ["HistGauss", "GARCH-t", "AR-GJR-t", "Gauss-HAR", "MAR", "LSTAR", "MS-AR", "XGB-Gauss",
                                       "Chronos-Bolt-small", "Chronos-Bolt-base", "TimesFM-2.5", "Deep-SMoE", "SMoE"]
                           if c in tab.index])

        tab = tab.reset_index().rename(columns={"model": "Model"})
        n = int(l["n"].iloc[0])
        to_tex(tab, f"asset_{a}", f"Out-of-sample results: {LAB[a]} ($n={n}$)", f"tab:asset_{a}", nd=4,
               size="\\scriptsize", bold_min_cols=("NLL", "CRPS", "QLIKE", "FZ5"))
        blocks.append(f"\\input{{../results/tables/tex/asset_{a}}}")
    (TABLES / "tex" / "per_asset_inputs.tex").write_text("\n\\clearpage\n".join(blocks), encoding="utf-8")
