"""Robustness: variance features from 5-minute realised measures instead of daily ranges.

Assets: CSI 300 and the two instruments written on it (IF futures, 510300 ETF). All three use
the index's 5-minute realised measures, which cover 2009-05 .. 2025-07. Development period only.
Every model is re-estimated on the same rows (days with 5-minute data), so differences are due
to the variance covariates alone.

  SMoE-PK        baseline SMoE (Parkinson HAR variance, Parkinson gate volatility)
  SMoE-RV        realised-variance HAR; gate volatility = log RV22
  SMoE-TV        HAR of RV + overnight^2 (same coverage as the close-to-close target)
  SMoE-CJ        continuous (bipower) HAR + jump term
  SMoE-SV        negative and positive semivariance of day t + weekly/monthly RV
  SMoE-PKRV      Parkinson and RV HAR together (encompassing)
  Gauss-HAR-PK / Gauss-HAR-RV, GARCH-t, RealGARCH-t

Writes results/tables/rv_*.csv
usage: python experiments/14_rv_robustness.py [n_jobs]
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.special import polygamma  # noqa: E402

from common import FORECASTS, OOS_START, DEV_END, features  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data.realized import build_csi300_realized  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402
from smoe.evaluation.losses import all_losses, qlike  # noqa: E402
from smoe.features.extra import REALIZED_GROUPS as RG, add_realized  # noqa: E402
from smoe.features.semantic import SEMANTIC_GROUPS as G  # noqa: E402
from smoe.models import SMoE, single_gaussian  # noqa: E402
from smoe.models.benchmarks import GARCH, RealizedGARCH  # noqa: E402
from smoe.walkforward import ForecastPanel, walk_forward  # noqa: E402

ASSETS = ["csi300", "if_main", "etf510300"]
GATE_RV = ["g_lrv", "g_mom60", "g_dd"]
VARIANTS = {
    "SMoE-PK": lambda: SMoE(name="SMoE-PK"),
    "SMoE-RV": lambda: SMoE(var_cols=RG["rv"], gate_cols=GATE_RV, name="SMoE-RV"),
    "SMoE-TV": lambda: SMoE(var_cols=RG["tv"], gate_cols=GATE_RV, name="SMoE-TV"),
    "SMoE-CJ": lambda: SMoE(var_cols=RG["bv"] + RG["jump"], gate_cols=GATE_RV, name="SMoE-CJ"),
    "SMoE-SV": lambda: SMoE(var_cols=RG["semi"] + ["lrv5", "lrv22"], gate_cols=GATE_RV, name="SMoE-SV"),
    "SMoE-PKRV": lambda: SMoE(var_cols=G["var"] + RG["rv"], name="SMoE-PKRV"),
    "Gauss-HAR-PK": lambda: single_gaussian(name="Gauss-HAR-PK"),
    "Gauss-HAR-RV": lambda: single_gaussian(var_cols=RG["rv"], name="Gauss-HAR-RV"),
    "GARCH-t": lambda: GARCH(name="GARCH-t"),
    "RealGARCH-t": RealizedGARCH,
}
FOCAL = "SMoE-PK"
OUT = FORECASTS / "dev_rv"


def rv_features(asset, rm):
    return add_realized(features(asset), rm)


def run(asset, name, rm):
    path = OUT / f"{asset}__{name}.npz"
    if not path.exists():
        walk_forward(VARIANTS[name], rv_features(asset, rm), start=OOS_START, end=DEV_END, freq="QS",
                     asset=asset).to_npz(path)
    return asset, name


def evaluate(rm):
    proxy = rm["rv"] + rm["ret_overnight"].fillna(0) ** 2
    nxt = pd.Series(proxy.index, index=proxy.index).shift(-1)
    loss_rows, dm_rows, mcs_rows = [], [], []
    for a in ASSETS:
        P = {v: ForecastPanel.from_npz(OUT / f"{a}__{v}.npz") for v in VARIANTS}
        idx = sorted(set.intersection(*[set(p.index) for p in P.values()]))
        L, F = {}, {}
        for m, p in P.items():
            pos = pd.Index(p.index).get_indexer(idx)
            fc, y = p.fc.subset(pos), p.y[pos]
            L[m], F[m] = all_losses(fc, y), (fc, y)
            hits = y < fc.quantile(0.01)
            row = {"asset": a, "model": m, "n": len(y), **L[m][["NLL", "CRPS", "QLIKE", "FZ5", "FZ1"]].mean(),
                   "viol1": hits.mean(), "kupiec1_p": T.kupiec(hits, 0.01)["p"],
                   "chris1_p": T.christoffersen(hits, 0.01)["p_cc"], "pit_sd": T.berkowitz(fc.pit(y))["sigma"]}
            if a == "csi300":
                target = pd.DatetimeIndex(idx).map(lambda d: nxt.get(d, pd.NaT))
                ok = ~pd.isna(target)
                pv = proxy.reindex(target[ok]).to_numpy()
                hv = fc.var()[np.asarray(ok)]
                good = np.isfinite(pv)
                mz = T.mincer_zarnowitz(pv[good], hv[good])
                row.update({"QLIKE_RV": qlike(hv[good], pv[good]).mean(), "MZ_R2_RV": mz["R2"], "MZ_b_RV": mz["b"]})
            loss_rows.append(row)
        for m in L:
            if m == FOCAL:
                continue
            for loss in ("NLL", "CRPS", "QLIKE", "FZ5"):
                d = T.dm_test(L[FOCAL][loss].to_numpy(), L[m][loss].to_numpy())
                dm_rows.append({"asset": a, "model": m, "loss": loss, "mean_diff": d["mean_diff"],
                                "dm_stat": d["stat"], "p_model_better": d["p_b_better"], "p_two": d["p_two"]})
        for loss in ("NLL", "QLIKE", "FZ5"):
            mat = pd.DataFrame({m: L[m][loss].to_numpy() for m in L})
            mc = T.model_confidence_set(mat, alpha=0.10, B=1000)
            for m, r in mc.iterrows():
                mcs_rows.append({"asset": a, "loss": loss, "model": m, "mcs_p": r["mcs_p"], "in_mcs": r["in_mcs"]})
        print("evaluated", a, len(idx), flush=True)
    pd.DataFrame(loss_rows).to_csv(TABLES / "rv_losses.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(TABLES / "rv_dm.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(TABLES / "rv_mcs.csv", index=False)


def measurement_error(rm, n_paths=20000, n_steps=2400, seed=0):
    """Variance of the log of each daily variance estimator around log sigma^2 (Brownian motion,
    constant intraday volatility), and the implied reliability of the daily HAR regressor."""
    rng = np.random.default_rng(seed)
    lr = []
    for _ in range(n_paths // 1000):
        path = np.cumsum(rng.standard_normal((1000, n_steps)) / np.sqrt(n_steps), axis=1)
        path = np.hstack([np.zeros((1000, 1)), path])
        rng_ = path.max(1) - path.min(1)
        lr.append(np.log(rng_**2 / (4 * np.log(2))))
    lpk = np.concatenate(lr)
    fs = rv_features("csi300", rm).slice(None, DEV_END)
    X = fs.X

    def resid_var(c1, c5, c22):
        A = np.c_[np.ones(len(X)), X[c5], X[c22]]
        coef = np.linalg.lstsq(A, X[c1], rcond=None)[0]
        return float(np.var(X[c1] - A @ coef)), coef[1:]

    om_rv, om_pk_bm = float(polygamma(1, 24)), float(lpk.var())
    res_rv, b_rv = resid_var("lrv1", "lrv5", "lrv22")
    res_pk, _ = resid_var("lv1", "lv5", "lv22")
    nu1 = res_rv - om_rv  # Var(log sigma_t^2 | weekly, monthly), errors in weekly/monthly terms neglected
    om_pk_emp = res_pk - nu1
    rows = [
        {"estimator": "squared return", "omega2": float(polygamma(1, 0.5)),
         "kappa": float(polygamma(0, 0.5) + np.log(2))},
        {"estimator": "Parkinson range (BM)", "omega2": om_pk_bm, "kappa": float(lpk.mean())},
        {"estimator": "Parkinson range (implied)", "omega2": om_pk_emp, "kappa": np.nan},
        {"estimator": "5-min RV (M=48)", "omega2": om_rv, "kappa": float(polygamma(0, 24) - np.log(24))},
    ]
    df = pd.DataFrame(rows)
    df["nu1"] = nu1
    df["attenuation"] = nu1 / (nu1 + df["omega2"])
    # observed daily HAR loadings (raw units) in the K=1 model and in every SMoE expert
    ok = fs.y.notna()
    Xo, yo = X[ok], fs.y[ok]
    obs = {}
    for pk, rv_, var_pk, var_rv in [("Gauss-HAR-PK", "Gauss-HAR-RV", "lv1", "lrv1"), ("SMoE-PK", "SMoE-RV", "lv1", "lrv1")]:
        mp, mr = VARIANTS[pk]().fit(Xo, yo), VARIANTS[rv_]().fit(Xo, yo)
        for k, e in enumerate(mp.expert_names):
            dp = mp.params_[1][k][1 + mp.var_cols.index(var_pk)] / mp.scale_[var_pk]
            dr = mr.params_[1][k][1 + mr.var_cols.index(var_rv)] / mr.scale_[var_rv]
            obs[f"{pk.split('-')[0]}[{e}]"] = (dp, dr, dp / dr)
    extra = pd.DataFrame([{"estimator": f"observed {k}", "delta1_pk": v[0], "delta1_rv": v[1], "ratio": v[2]}
                          for k, v in obs.items()])
    pred = pd.DataFrame([{"estimator": f"predicted ratio ({lab})", "ratio": (nu1 + om_rv) / (nu1 + om)}
                         for lab, om in [("BM", om_pk_bm), ("implied", om_pk_emp)]])
    df = pd.concat([df, pred, extra], ignore_index=True)
    df["b5"], df["b22"] = b_rv[0], b_rv[1]
    df.to_csv(TABLES / "rv_measurement.csv", index=False)
    print(df.round(4).to_string())


def lemma_check(rm):
    """Lemma (cost of a log-variance error): predicted log-score difference = Delta E[e^2] / 4, where
    e = log(proxy_{t+1}) - log(forecast variance); the proxy's own error is common to all models."""
    proxy = rm["rv"] + rm["ret_overnight"].fillna(0) ** 2
    nxt = pd.Series(proxy.index, index=proxy.index).shift(-1)
    P = {v: ForecastPanel.from_npz(OUT / f"csi300__{v}.npz") for v in VARIANTS}
    idx = sorted(set.intersection(*[set(p.index) for p in P.values()]))
    target = pd.DatetimeIndex(idx).map(lambda d: nxt.get(d, pd.NaT))
    ok = np.asarray(~pd.isna(target))
    lp = np.log(proxy.reindex(target[ok]).to_numpy())
    rows = []
    for m, p in P.items():
        pos = pd.Index(p.index).get_indexer(idx)
        fc, y = p.fc.subset(pos), p.y[pos]
        e = lp - np.log(fc.var()[ok])
        g = np.isfinite(e)
        rows.append({"model": m, "mean_e": e[g].mean(), "var_e": e[g].var(), "msq_e": np.mean(e[g] ** 2),
                     "NLL": -fc.logpdf(y)[ok][g].mean()})
    df = pd.DataFrame(rows).set_index("model")
    base = df.loc["SMoE-PK"]
    df["pred_dNLL_var"] = (df["var_e"] - base["var_e"]) / 4
    df["obs_dNLL"] = df["NLL"] - base["NLL"]
    df.to_csv(TABLES / "rv_lemma.csv")
    print(df.round(4).to_string())


def insample(rm):
    """CSI 300, development sample: HAR loadings and gate tests under each variance specification."""
    fs = rv_features("csi300", rm).slice(None, DEV_END)
    ok = fs.y.notna()
    X, y = fs.X[ok].assign(r=fs.r[ok]), fs.y[ok]
    par, lik = [], []
    for name in ["SMoE-PK", "SMoE-RV", "SMoE-TV", "SMoE-CJ", "SMoE-SV", "SMoE-PKRV", "Gauss-HAR-PK",
                 "Gauss-HAR-RV"]:
        m = VARIANTS[name]().fit(X, y)
        inf = m.inference(X, y)
        lik.append({"model": name, "loglik": m.loglik_, "params": m.n_params_, "AIC": m.aic, "BIC": m.bic})
        th, V = m.sandwich_cov(X, y)
        for nm, r in inf.iterrows():
            if nm.startswith("delta") or nm.startswith("gamma"):
                par.append({"model": name, "param": nm, "estimate": r["estimate"], "z": r["z"]})
        if m.K > 1:
            w = m.wald(th, V, lambda n: n.startswith("gamma") and ("g_lvol" in n or "g_lrv" in n))
            lik[-1].update({"W_gate_vol": w["W"], "p_gate_vol": w["p"]})
            w = m.wald(th, V, lambda n: n.startswith("gamma") and ("g_mom60" in n or "g_dd" in n))
            lik[-1].update({"W_gate_momdd": w["W"], "p_gate_momdd": w["p"]})
        if name == "SMoE-SV":  # bad vs good volatility: delta_neg = delta_pos in every expert
            names = m.param_names()
            R = np.zeros((m.K, len(th)))
            for k, e in enumerate(m.expert_names):
                R[k, names.index(f"delta[{e}]:lrsn1")], R[k, names.index(f"delta[{e}]:lrsp1")] = 1, -1
            d = R @ th
            W = float(d @ np.linalg.solve(R @ V @ R.T, d))
            from scipy.stats import chi2
            lik[-1].update({"W_semi_equal": W, "p_semi_equal": float(chi2.sf(W, m.K))})
    for nm in ["RealGARCH-t", "GARCH-t"]:
        m = VARIANTS[nm]().fit(X, y)
        lik.append({"model": nm, "loglik": float(m.predict(X).logpdf(y.to_numpy()).sum()), "params": np.nan})
    pd.DataFrame(par).to_csv(TABLES / "rv_insample_params.csv", index=False)
    pd.DataFrame(lik).to_csv(TABLES / "rv_insample_lik.csv", index=False)
    print(pd.DataFrame(lik).round(2).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rm = build_csi300_realized()
    if sys.argv[1:2] == ["measure"]:
        measurement_error(rm)
        lemma_check(rm)
        sys.exit()
    Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 10)(
        delayed(run)(a, v, rm) for a in ASSETS for v in VARIANTS)
    evaluate(rm)
    measurement_error(rm)
    lemma_check(rm)
    insample(rm)
