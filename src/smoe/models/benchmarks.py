"""Benchmark density forecasters, all returning `MixtureForecast`.

Interface shared with SMoE:
    fit(X, y)      X: features incl. column "r" (= r_t), y: r_{t+1}
    predict(X)     X: rows to forecast, preceded by any history the model needs
                   (recursive models read only X["r"], i.e. returns up to t).
predict returns one forecast per row of X.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import roots_genlaguerre, gammaln

from smoe.distributions import MixtureForecast
from smoe.features.semantic import SEMANTIC_GROUPS


def student_t_mixture(mu, scale, nu, n_nodes: int = 64) -> MixtureForecast:
    """Student-t(mu, scale, nu) as a Gaussian scale mixture via generalised Gauss-Laguerre quadrature.

    t = N(mu, scale^2 / lam), lam ~ Gamma(nu/2, rate nu/2). With u = (nu/2) lam,
    E[f(lam)] = Gamma(nu/2)^{-1} int f(2u/nu) u^{nu/2-1} e^{-u} du.
    """
    mu, scale = np.asarray(mu, float), np.asarray(scale, float)
    a = nu / 2.0
    u, wq = roots_genlaguerre(n_nodes, a - 1.0)
    wq = np.exp(np.log(wq) - gammaln(a))
    wq = wq / wq.sum()
    lam = u / a
    T = mu.size
    w = np.tile(wq, (T, 1))
    sd = scale[:, None] / np.sqrt(lam)[None, :]
    return MixtureForecast(w, np.tile(mu[:, None], (1, n_nodes)), sd)


@dataclass
class HistGauss:
    """N(rolling mean, rolling variance) of the last `window` returns."""
    window: int = 250
    name: str = "HistGauss"

    def fit(self, X, y):
        return self

    def predict(self, X):
        r = X["r"]
        m = r.rolling(self.window, min_periods=60).mean().bfill()
        s = r.rolling(self.window, min_periods=60).std().bfill()
        return MixtureForecast.gaussian(m.to_numpy(), s.to_numpy())


@dataclass
class GARCH:
    """AR(p)-(GJR-)GARCH(1,1) with Student-t or Gaussian errors (arch package, fixed-param recursion)."""
    ar: int = 0
    gjr: bool = False
    dist: str = "t"
    zero_mean: bool = False
    name: str = "GARCH-t"

    def fit(self, X, y):
        from arch import arch_model

        r = X["r"].to_numpy(float)
        mean = "Zero" if self.zero_mean else ("AR" if self.ar else "Constant")
        am = arch_model(r, mean=mean, lags=self.ar or None, vol="GARCH", p=1, o=int(self.gjr), q=1,
                        dist="t" if self.dist == "t" else "normal", rescale=False)
        res = am.fit(disp="off", options={"maxiter": 500})
        self.params_ = res.params
        self.r_last_ = r
        return self

    def _recursion(self, r):
        p = self.params_
        n = len(r)
        mu_c = 0.0 if self.zero_mean else p.get("Const", p.get("mu", 0.0))
        phi = p.get("y[1]", 0.0) if self.ar else 0.0
        mean_next = mu_c + phi * r  # forecast of r_{t+1} made at t
        eps = np.r_[np.nan, r[1:] - (mu_c + phi * r[:-1])]
        eps[0] = r[0] - mu_c
        om, al, be = p["omega"], p["alpha[1]"], p["beta[1]"]
        ga = p.get("gamma[1]", 0.0)
        s2 = np.empty(n)
        prev = np.var(r)
        for t in range(n):  # s2[t] = variance of r_{t+1} given info at t
            prev = om + (al + ga * (eps[t] < 0)) * eps[t] ** 2 + be * prev
            s2[t] = prev
        return mean_next, s2

    def predict(self, X):
        mean_next, s2 = self._recursion(X["r"].to_numpy(float))
        if self.dist == "t":
            nu = float(self.params_["nu"])
            scale = np.sqrt(s2 * (nu - 2) / nu)
            return student_t_mixture(mean_next, scale, nu)
        return MixtureForecast.gaussian(mean_next, np.sqrt(s2))


@dataclass
class MarkovSwitching:
    """Hamilton-type 2-regime switching regression r_{t+1} = c_s + b_s r_t + sigma_s e."""
    k_regimes: int = 2
    name: str = "MS-AR"

    def fit(self, X, y):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        ok = y.notna()
        endog, exog = y[ok].to_numpy(float), X.loc[ok, "r"].to_numpy(float)
        best = None
        for seed in range(3):
            np.random.seed(seed)  # statsmodels search_reps draws from the global RNG
            mod = MarkovRegression(endog, k_regimes=self.k_regimes, exog=exog, switching_variance=True)
            try:
                res = mod.fit(disp=False, search_reps=10 if seed else 0, maxiter=200)
            except Exception:
                continue
            if best is None or res.llf > best.llf:
                best = res
        self.res_ = best
        self.params_ = best.params
        self.y_train_, self.r_train_ = endog, exog
        self.train_last_ = X.index[ok][-1]
        return self

    def predict(self, X):
        """Online filtering with fixed parameters; regime probs at row t use y up to row t-1 (= r up to t)."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        r = X["r"].to_numpy(float)
        y_lag = np.r_[r[1:], np.nan]  # y_t = r_{t+1}; only y_{t-1} = r_t enters the filter for row t
        endog = y_lag[:-1]
        exog = r[:-1]
        mod = MarkovRegression(endog, k_regimes=self.k_regimes, exog=exog, switching_variance=True)
        fres = mod.filter(self.params_)
        filt = np.asarray(fres.filtered_marginal_probabilities)  # P(S_t | y_0..y_t), rows 0..n-2
        P = np.asarray(fres.regime_transition)[:, :, 0]  # P[i, j] = P(S_t = i | S_{t-1} = j)
        pred = np.empty((len(r), self.k_regimes))
        pred[0] = np.asarray(fres.predicted_marginal_probabilities)[0]
        pred[1:] = filt @ P.T  # P(S_{t+1} | y_0..y_t)
        p = self.params_
        k = self.k_regimes
        names = list(self.res_.model.param_names)
        c = np.array([p[names.index(f"const[{i}]")] for i in range(k)])
        b = np.array([p[names.index(f"x1[{i}]")] for i in range(k)])
        s2 = np.array([p[names.index(f"sigma2[{i}]")] for i in range(k)])
        mu = c[None, :] + b[None, :] * r[:, None]
        sd = np.tile(np.sqrt(s2), (len(r), 1))
        return MixtureForecast(pred, mu, sd)


@dataclass
class XGBGauss:
    """Gradient boosting for the mean + log-linear HAR variance on boosted residuals."""
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.03
    name: str = "XGB-Gauss"
    mean_cols: list = field(default_factory=lambda: SEMANTIC_GROUPS["trend"] + SEMANTIC_GROUPS["reversal"]
                            + SEMANTIC_GROUPS["var"] + SEMANTIC_GROUPS["gate"])

    def fit(self, X, y):
        from xgboost import XGBRegressor

        ok = y.notna()
        Xm, yv = X.loc[ok, self.mean_cols].to_numpy(), y[ok].to_numpy()
        n_val = max(int(0.15 * len(yv)), 100)
        self.mod_ = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                 learning_rate=self.learning_rate, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=20, reg_lambda=5.0, early_stopping_rounds=30,
                                 random_state=0, n_jobs=4)
        self.mod_.fit(Xm[:-n_val], yv[:-n_val], eval_set=[(Xm[-n_val:], yv[-n_val:])], verbose=False)
        res2 = np.log((yv - self.mod_.predict(Xm)) ** 2 + 1e-6)
        V = np.c_[np.ones(len(yv)), X.loc[ok, SEMANTIC_GROUPS["var"]].to_numpy()]
        self.vcoef_ = np.linalg.lstsq(V, res2, rcond=None)[0]
        # E[log chi2_1] = -1.27036: bias correction for log squared residuals
        self.vcoef_[0] += 1.27036
        return self

    def predict(self, X):
        m = self.mod_.predict(X[self.mean_cols].to_numpy())
        V = np.c_[np.ones(len(X)), X[SEMANTIC_GROUPS["var"]].to_numpy()]
        return MixtureForecast.gaussian(m, np.exp(0.5 * V @ self.vcoef_))


@dataclass
class RealizedGARCH:
    """Log-linear Realized GARCH(1,1) of Hansen, Huang & Shek (2012) with standardised Student-t errors.

        r_t        = mu + sqrt(h_t) z_t
        log h_t    = omega + beta log h_{t-1} + gamma log x_{t-1}
        log x_t    = xi + phi log h_t + tau1 z_t + tau2 (z_t^2 - 1) + u_t,   u_t ~ N(0, s_u^2)

    x_t is the 5-minute realised variance of day t (column `x_col`, already in logs), so the
    forecast of r_{t+1} made at row t uses h_{t+1}, a function of returns and RV up to t.
    The joint likelihood of (r_t, log x_t) is maximised; the recursion is a linear filter in log h.
    """
    x_col: str = "lrv1"
    name: str = "RealGARCH-t"

    def _filter(self, th, r, lx, lh0):
        from scipy.signal import lfilter

        mu, om, be, ga = th[:4]
        drive = np.r_[0.0, om + ga * lx[:-1]]  # input at t: omega + gamma log x_{t-1}
        lh, _ = lfilter([1.0], [1.0, -be], drive[1:], zi=[be * lh0])
        return np.r_[lh0, lh]  # log h_t for rows 0..n-1

    def _nll(self, th, r, lx, lh0):
        mu, om, be, ga, xi, phi, t1, t2, lsu, lnu = th
        nu = 2.0 + np.exp(lnu)
        lh = self._filter(th, r, lx, lh0)
        z = (r - mu) / np.exp(0.5 * lh)
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        ll_r = c - 0.5 * lh - (nu + 1) / 2 * np.log1p(z**2 / (nu - 2))
        u = lx - xi - phi * lh - t1 * z - t2 * (z**2 - 1)
        ll_x = -0.5 * (np.log(2 * np.pi) + 2 * lsu + u**2 * np.exp(-2 * lsu))
        v = -(ll_r[1:].sum() + ll_x[1:].sum())
        return v if np.isfinite(v) else 1e12

    def fit(self, X, y, warm_start: "RealizedGARCH | None" = None):
        from scipy.optimize import minimize

        r, lx = X["r"].to_numpy(float), X[self.x_col].to_numpy(float)
        self.lh0_ = float(np.log(np.var(r)))
        if warm_start is not None and hasattr(warm_start, "theta_"):
            th0 = warm_start.theta_
        else:
            th0 = np.array([r.mean(), 0.0, 0.6, 0.35, np.mean(lx) - self.lh0_, 1.0, -0.05, 0.05, np.log(0.5), np.log(6.0)])
            th0[1] = (1 - 0.6 - 0.35) * self.lh0_
        bounds = [(None, None), (None, None), (-0.5, 0.9999), (-1.0, 2.0), (None, None), (-2.0, 4.0),
                  (None, None), (None, None), (np.log(1e-3), None), (np.log(0.05), np.log(200.0))]
        res = minimize(self._nll, th0, args=(r, lx, self.lh0_), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000})
        self.theta_ = res.x
        self.loglik_ = -res.fun
        return self

    def predict(self, X):
        r, lx = X["r"].to_numpy(float), X[self.x_col].to_numpy(float)
        th = self.theta_
        mu, om, be, ga = th[:4]
        nu = 2.0 + np.exp(th[9])
        lh = self._filter(th, r, lx, self.lh0_)
        lh_next = om + be * lh + ga * lx  # log h_{t+1} given information at t
        scale = np.sqrt(np.exp(lh_next) * (nu - 2) / nu)
        return student_t_mixture(np.full(len(r), mu), scale, nu)
