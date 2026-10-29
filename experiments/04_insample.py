"""In-sample white-box analysis on the development sample (through DEV_END).

CSI 300: parameter table with sandwich SEs, Wald tests of semantic restrictions, likelihood
comparison with nested models, gate-probability and risk-decomposition figures, policy events.
All assets: white-box "fingerprints" (key coefficients and robust t-statistics).
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

from common import ASSETS, DEV_END, features  # noqa: E402
from smoe import FIGURES, TABLES  # noqa: E402
from smoe.data import UNIVERSE, load_daily  # noqa: E402
from smoe.models import SMoE, constant_gate, lstar, single_gaussian  # noqa: E402
from smoe import plotstyle as ps  # noqa: E402

POLICY_EVENTS = {  # date: label (China-specific market/policy events)
    "2008-09-19": "stamp-duty cut, Huijin buying",
    "2015-06-12": "2015 crash begins",
    "2015-07-08": "rescue package",
    "2016-01-04": "circuit breaker",
    "2018-03-22": "trade war tariffs",
    "2020-01-23": "Wuhan lockdown",
    "2024-02-05": "national team buying",
    "2024-09-24": "PBoC/CSRC package",
}


def wald(inf_cov, theta, names, select):
    idx = [i for i, n in enumerate(names) if select(n)]
    g = theta[idx]
    W = g @ np.linalg.solve(inf_cov[np.ix_(idx, idx)], g)
    return W, len(idx), stats.chi2.sf(W, len(idx))


def full_cov(m, X, y):
    th = m._theta(m.params_)
    D = m._design(X)
    yv = y.to_numpy(float)
    S = m._scores(th, D, yv)
    n, P = S.shape
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    B = S.T @ S
    for l in range(1, L + 1):
        G = S[l:].T @ S[:-l]
        B += (1 - l / (L + 1)) * (G + G.T)
    A = np.zeros((P, P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = 1e-5
        A[:, j] = (m._scores(th + e, D, yv).sum(0) - m._scores(th - e, D, yv).sum(0)) / 2e-5
    Ai = np.linalg.pinv(0.5 * (A + A.T))
    return th, Ai @ B @ Ai


def fingerprint(asset):
    fs = features(asset).slice(None, DEV_END)
    ok = fs.y.notna()
    X, y = fs.X[ok], fs.y[ok]
    m = SMoE().fit(X, y)
    g = single_gaussian().fit(X, y)
    inf = m.inference(X, y)
    (TABLES / "params").mkdir(parents=True, exist_ok=True)
    inf.to_csv(TABLES / "params" / f"{asset}.csv")
    th, V = full_cov(m, X, y)
    names = m.param_names()
    Wg = wald(V, th, names, lambda n: n.startswith("gamma") and not n.endswith(":const"))
    pi = m.gate_probs(X).mean()
    row = {"asset": asset, "label": UNIVERSE[asset].label, "n": len(y),
           "LR_vs_K1": 2 * (m.loglik_ - g.loglik_), "dBIC_vs_K1": g.bic - m.bic,
           "W_gate": Wg[0], "W_gate_p": Wg[2], **{f"pi_{k}": v for k, v in pi.items()}}
    for nm in ["beta[trend]:mom120", "beta[trend]:mom20", "beta[reversal]:rev1", "beta[reversal]:rev5",
               "beta[crisis]:const", "gamma[trend]:g_lvol", "gamma[trend]:g_dd", "gamma[reversal]:g_lvol"]:
        row[nm] = inf.loc[nm, "estimate"]
        row[nm + "_t"] = inf.loc[nm, "z"]
    return row, m, X, y, inf, (th, V, names)


if __name__ == "__main__":
    ps.use()
    TABLES.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=10)(delayed(fingerprint)(a) for a in ASSETS)
    fp = pd.DataFrame([r[0] for r in res])
    fp.to_csv(TABLES / "insample_fingerprints.csv", index=False)
    print(fp.round(3).to_string())

    # ------------------------------------------------------------- CSI 300 detail
    row, m, X, y, inf, (th, V, names) = res[ASSETS.index("csi300")]
    inf.to_csv(TABLES / "insample_csi300_params.csv")
    tests = []
    for label, sel in [
        ("all gate slopes = 0 (SMoE vs MAR)", lambda n: n.startswith("gamma") and not n.endswith(":const")),
        ("volatility does not enter the gate", lambda n: n.startswith("gamma") and n.endswith("g_lvol")),
        ("momentum does not enter the gate", lambda n: n.startswith("gamma") and n.endswith("g_mom60")),
        ("drawdown does not enter the gate", lambda n: n.startswith("gamma") and n.endswith("g_dd")),
        ("trend expert slopes = 0", lambda n: n.startswith("beta[trend]") and not n.endswith(":const")),
        ("reversal expert slopes = 0", lambda n: n.startswith("beta[reversal]") and not n.endswith(":const")),
        ("all mean slopes = 0", lambda n: n.startswith("beta") and not n.endswith(":const")),
    ]:
        W, df, p = wald(V, th, names, sel)
        tests.append({"hypothesis": label, "Wald": W, "df": df, "p": p})
    # variance HAR loadings equal across experts: delta_k slopes equal (linear restriction R theta = 0)
    vs = [[i for i, n in enumerate(names) if n.startswith(f"delta[{k}]") and not n.endswith(":const")]
          for k in m.expert_names]
    R = np.zeros((2 * len(vs[0]), len(th)))
    for j in range(len(vs[0])):
        R[j, vs[0][j]], R[j, vs[1][j]] = 1, -1
        R[len(vs[0]) + j, vs[0][j]], R[len(vs[0]) + j, vs[2][j]] = 1, -1
    r = R @ th
    W = r @ np.linalg.solve(R @ V @ R.T, r)
    tests.append({"hypothesis": "equal variance dynamics across experts", "Wald": W, "df": R.shape[0],
                  "p": stats.chi2.sf(W, R.shape[0])})
    pd.DataFrame(tests).to_csv(TABLES / "insample_csi300_wald.csv", index=False)

    comp = []
    for name, mod in [("Gauss-HAR (K=1)", single_gaussian()), ("MAR", constant_gate()), ("LSTAR", lstar()),
                      ("SMoE", m)]:
        if mod is not m:
            mod.fit(X, y)
        comp.append({"model": name, "loglik": mod.loglik_, "params": mod.n_params_, "AIC": mod.aic, "BIC": mod.bic})
    pd.DataFrame(comp).to_csv(TABLES / "insample_csi300_lik.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string(), "\n", pd.DataFrame(comp).round(1).to_string())

    # ------------------------------------------------------------- figures
    import matplotlib.pyplot as plt

    gp = m.gate_probs(X)
    fc = m.predict(X)
    price = load_daily("csi300")["close"].reindex(X.index)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]})
    axes[0].plot(price.index, price, color=ps.INK, lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("CSI 300 (log scale)")
    for d, lab in POLICY_EVENTS.items():
        d = pd.Timestamp(d)
        if X.index[0] <= d <= X.index[-1]:
            for ax in axes:
                ax.axvline(d, color=ps.MUTED, lw=0.5, ls=":")
    sm = gp.rolling(10, min_periods=1).mean()
    axes[1].stackplot(sm.index, *[sm[k] for k in m.expert_names],
                      colors=[ps.EXPERT_COLORS[k] for k in m.expert_names], labels=m.expert_names, lw=0)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("gate weight $\\pi_k$\n(10-day mean)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    wv = pd.Series(fc.within_var(), index=X.index).rolling(10, min_periods=1).mean()
    bv = pd.Series(fc.between_var(), index=X.index).rolling(10, min_periods=1).mean()
    axes[2].stackplot(wv.index, wv, bv, colors=[ps.SERIES[0], ps.SERIES[3]], lw=0,
                      labels=["within-regime risk", "regime-disagreement risk"])
    axes[2].set_ylabel("predictive variance\n(%$^2$, 10-day mean)")
    axes[2].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ps.save(fig, FIGURES / "csi300_gates_risk.pdf")

    # gate response curves: pi_k as a function of each state variable, others at their mean
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    base = X.mean()
    for ax, c in zip(axes, m.gate_cols):
        grid = np.linspace(X[c].quantile(0.01), X[c].quantile(0.99), 100)
        Xg = pd.DataFrame([base] * len(grid))
        Xg[c] = grid
        P = m.gate_probs(Xg)
        for k in m.expert_names:
            ax.plot(grid, P[k], color=ps.EXPERT_COLORS[k], label=k)
        ax.set_xlabel({"g_lvol": "log 22-day variance", "g_mom60": "60-day momentum (scaled)",
                       "g_dd": "log drawdown from 1y high"}[c])
    axes[0].set_ylabel("gate weight")
    axes[0].legend(loc="best")
    ps.save(fig, FIGURES / "csi300_gate_curves.pdf")

    # policy events: gate weights before/after
    ev = []
    for d, lab in POLICY_EVENTS.items():
        d = pd.Timestamp(d)
        if not (X.index[0] + pd.Timedelta(days=60) <= d <= X.index[-1] - pd.Timedelta(days=30)):
            continue
        pos = X.index.searchsorted(d)
        pre, post = gp.iloc[pos - 20:pos].mean(), gp.iloc[pos:pos + 20].mean()
        ev.append({"date": d.date(), "event": lab, **{f"pre_{k}": pre[k] for k in m.expert_names},
                   **{f"post_{k}": post[k] for k in m.expert_names}})
    pd.DataFrame(ev).to_csv(TABLES / "insample_csi300_events.csv", index=False)
    print(pd.DataFrame(ev).round(3).to_string())
