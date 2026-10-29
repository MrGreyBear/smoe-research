"""White-box Semantic Mixture of Experts (SMoE) estimated by ECM.

Model
-----
    y_t | x_t ~ sum_k pi_k(z_t) N(mu_k(t), sigma_k^2(t))
    pi_k(z)        = softmax_k(gamma_k' [1, z]),   gamma_K = 0 (reference expert)
    mu_k(t)        = beta_k' [1, x_t^{(k)}]        (expert k sees only its semantic group)
    log sigma_k^2  = delta_k' [1, v_t]

Estimation maximises the penalised log-likelihood
    l(theta) - (lam/2) * ||slopes||^2 - (lam_v/2) * ||delta - delta_0||^2
by ECM (Meng & Rubin, 1993): E-step responsibilities, then conditional
maximisation over beta (weighted ridge LS, closed form), delta (Newton on a
Gamma-type GLM with exact Hessian) and gamma (Newton on a weighted multinomial
logit). Every CM step increases the expected complete-data objective, so the
penalised likelihood is monotone non-decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import norm

from smoe.distributions import MixtureForecast
from smoe.features.semantic import SEMANTIC_GROUPS

LOG2PI = np.log(2 * np.pi)

DEFAULT_EXPERTS = {
    "trend": SEMANTIC_GROUPS["trend"],
    "reversal": SEMANTIC_GROUPS["reversal"],
    "crisis": [],
}


@dataclass
class _Design:
    Xk: list  # per expert (n, 1+p_k)
    V: np.ndarray  # (n, 1+q)
    Z: np.ndarray  # (n, 1+m)


@dataclass
class SMoE:
    experts: dict = field(default_factory=lambda: dict(DEFAULT_EXPERTS))
    var_cols: list = field(default_factory=lambda: list(SEMANTIC_GROUPS["var"]))
    gate_cols: list = field(default_factory=lambda: list(SEMANTIC_GROUPS["gate"]))
    lam: float = 1.0
    lam_v: float = 1.0
    max_iter: int = 150
    tol: float = 1e-7
    n_init: int = 4
    seed: int = 0
    name: str = "SMoE"

    # ------------------------------------------------------------------ setup
    @property
    def K(self) -> int:
        return len(self.experts)

    @property
    def expert_names(self) -> list:
        return list(self.experts)

    def _all_cols(self):
        cols = []
        for c in [*sum(self.experts.values(), []), *self.var_cols, *self.gate_cols]:
            if c not in cols:
                cols.append(c)
        return cols

    def _design(self, X: pd.DataFrame) -> _Design:
        Xs = (X[self._cols] - self.loc_) / self.scale_
        one = np.ones((len(X), 1))
        Xk = [np.hstack([one, Xs[c].to_numpy()]) for c in self.experts.values()]
        V = np.hstack([one, Xs[self.var_cols].to_numpy()])
        Z = np.hstack([one, Xs[self.gate_cols].to_numpy()])
        return _Design(Xk, V, Z)

    # ------------------------------------------------------- core quantities
    def _gate(self, Z, gamma):
        logits = np.hstack([Z @ gamma.T, np.zeros((Z.shape[0], 1))])
        return logits - logsumexp(logits, axis=1, keepdims=True)

    def _components(self, D: _Design, beta, delta, gamma):
        mu = np.column_stack([D.Xk[k] @ beta[k] for k in range(self.K)])
        lv = np.column_stack([D.V @ delta[k] for k in range(self.K)])
        lpi = self._gate(D.Z, gamma) if self.K > 1 else np.zeros((D.V.shape[0], 1))
        return mu, lv, lpi

    def _penalty(self, beta, delta, gamma):
        p = 0.5 * self.lam * sum(np.sum(b[1:] ** 2) for b in beta)
        p += 0.5 * self.lam * np.sum(gamma[:, 1:] ** 2) if self.K > 1 else 0.0
        p += 0.5 * self.lam_v * sum(np.sum((d - self.delta0_) ** 2) for d in delta)
        return p

    def _loglik_terms(self, D, y, beta, delta, gamma):
        mu, lv, lpi = self._components(D, beta, delta, gamma)
        lc = lpi - 0.5 * (LOG2PI + lv + (y[:, None] - mu) ** 2 / np.exp(lv))
        return lc, logsumexp(lc, axis=1)

    # ---------------------------------------------------------------- M steps
    def _m_beta(self, D, y, tau, lv, beta):
        out = []
        for k in range(self.K):
            X = D.Xk[k]
            a = tau[:, k] / np.exp(lv[:, k])
            R = self.lam * np.eye(X.shape[1])
            R[0, 0] = 0.0
            A = X.T @ (X * a[:, None]) + R
            out.append(np.linalg.solve(A + 1e-10 * np.eye(len(A)), X.T @ (a * y)))
        return out

    def _m_delta(self, D, y, tau, mu, delta, steps=5):
        V = D.V
        out = []
        for k in range(self.K):
            d = delta[k].copy()
            e2 = (y - mu[:, k]) ** 2
            t = tau[:, k]

            def q(dd):
                eta = V @ dd
                return np.sum(t * (-0.5 * eta - 0.5 * e2 * np.exp(-eta))) - 0.5 * self.lam_v * np.sum(
                    (dd - self.delta0_) ** 2)

            for _ in range(steps):
                eta = V @ d
                u = e2 * np.exp(-eta)
                g = V.T @ (t * 0.5 * (u - 1)) - self.lam_v * (d - self.delta0_)
                H = -(V.T @ (V * (t * 0.5 * u)[:, None])) - self.lam_v * np.eye(len(d))
                step = np.linalg.solve(H, g)
                q0, s = q(d), 1.0
                while s > 1e-6 and q(d - s * step) < q0:
                    s *= 0.5
                if s <= 1e-6:
                    break
                d = d - s * step
            out.append(d)
        return out

    def _m_gamma(self, D, tau, gamma, steps=5):
        if self.K == 1:
            return gamma
        Z = D.Z
        m, Km1 = Z.shape[1], self.K - 1
        R = self.lam * np.eye(m)
        R[0, 0] = 1e-6

        def q(g):
            lpi = self._gate(Z, g)
            return np.sum(tau * lpi) - 0.5 * self.lam * np.sum(g[:, 1:] ** 2)

        g = gamma.copy()
        for _ in range(steps):
            pi = np.exp(self._gate(Z, g))
            grad = ((tau - pi)[:, :Km1].T @ Z) - g @ R
            H = np.zeros((Km1 * m, Km1 * m))
            for a in range(Km1):
                for b in range(Km1):
                    wab = pi[:, a] * ((a == b) - pi[:, b])
                    H[a * m:(a + 1) * m, b * m:(b + 1) * m] = -(Z.T @ (Z * wab[:, None])) - (a == b) * R
            step = np.linalg.solve(H, grad.ravel()).reshape(Km1, m)
            q0, s = q(g), 1.0
            while s > 1e-6 and q(g - s * step) < q0:
                s *= 0.5
            if s <= 1e-6:
                break
            g = g - s * step
        return g

    # -------------------------------------------------------------------- fit
    def _init_tau(self, D, y, rng):
        n = len(y)
        tau = rng.dirichlet(np.ones(self.K), size=n)
        if "crisis" in self.experts and self.K > 1:
            k = self.expert_names.index("crisis")
            hi = D.V[:, -1] > np.quantile(D.V[:, -1], 0.8)
            tau[hi, k] += 2.0
            tau /= tau.sum(axis=1, keepdims=True)
        return tau

    def _run(self, D, y, tau=None, params=None):
        n = len(y)
        if params is None:
            beta = [np.zeros(X.shape[1]) for X in D.Xk]
            delta = [self.delta0_.copy() for _ in range(self.K)]
            gamma = np.zeros((self.K - 1, D.Z.shape[1]))
            mu = np.zeros((n, self.K))
            lv = np.tile(D.V @ self.delta0_, (self.K, 1)).T
            for _ in range(3):  # M-step from initial responsibilities
                beta = self._m_beta(D, y, tau, lv, beta)
                mu = np.column_stack([D.Xk[k] @ beta[k] for k in range(self.K)])
                delta = self._m_delta(D, y, tau, mu, delta)
                lv = np.column_stack([D.V @ delta[k] for k in range(self.K)])
            gamma = self._m_gamma(D, tau, gamma)
        else:
            beta, delta, gamma = [b.copy() for b in params[0]], [d.copy() for d in params[1]], params[2].copy()

        path = []
        for it in range(self.max_iter):
            lc, ll = self._loglik_terms(D, y, beta, delta, gamma)
            obj = ll.sum() - self._penalty(beta, delta, gamma)
            path.append(obj)
            if it > 0 and abs(path[-1] - path[-2]) < self.tol * (1 + abs(path[-1])):
                break
            tau = np.exp(lc - ll[:, None])
            mu, lv, _ = self._components(D, beta, delta, gamma)
            beta = self._m_beta(D, y, tau, lv, beta)
            mu = np.column_stack([D.Xk[k] @ beta[k] for k in range(self.K)])
            delta = self._m_delta(D, y, tau, mu, delta)
            gamma = self._m_gamma(D, tau, gamma)
        return (beta, delta, gamma), np.array(path)

    def fit(self, X: pd.DataFrame, y: pd.Series, warm_start: "SMoE | None" = None) -> "SMoE":
        ok = y.notna()
        X, y = X[ok], y[ok]
        self._cols = self._all_cols()
        self.loc_ = X[self._cols].mean()
        self.scale_ = X[self._cols].std().replace(0, 1.0)
        yv = y.to_numpy(float)
        self.delta0_ = np.zeros(1 + len(self.var_cols))
        self.delta0_[0] = np.log(np.var(yv))
        D = self._design(X)

        best = None
        if warm_start is not None and hasattr(warm_start, "params_"):
            best = self._run(D, yv, params=warm_start.params_)
        rng = np.random.default_rng(self.seed)
        n_init = self.n_init if best is None else 1
        for _ in range(n_init):
            cand = self._run(D, yv, tau=self._init_tau(D, yv, rng))
            if best is None or cand[1][-1] > best[1][-1]:
                best = cand
        self.params_, self.path_ = best
        self.params_ = self._polish(D, yv, self.params_)
        _, ll = self._loglik_terms(D, yv, *self.params_)
        self.loglik_ = float(ll.sum())
        self.n_obs_ = len(yv)
        self.n_params_ = self._theta(self.params_).size
        self.train_index_ = X.index
        return self

    def _penalty_grad(self, theta):
        beta, delta, gamma = self._unpack(theta)
        g = [self.lam * np.r_[0.0, b[1:]] for b in beta]
        g += [self.lam_v * (d - self.delta0_) for d in delta]
        if self.K > 1:
            gg = self.lam * gamma.copy()
            gg[:, 0] = 0.0
            g.append(gg.ravel())
        return np.concatenate(g)

    def _polish(self, D, y, params):
        """Quasi-Newton refinement of the ECM solution with the exact analytic gradient."""
        from scipy.optimize import minimize

        def f(th):
            b, d, g = self._unpack(th)
            _, ll = self._loglik_terms(D, y, b, d, g)
            obj = ll.sum() - self._penalty(b, d, g)
            grad = self._scores(th, D, y).sum(0) - self._penalty_grad(th)
            return -obj, -grad

        th0 = self._theta(params)
        res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-6})
        if np.isfinite(res.fun) and -res.fun >= -f(th0)[0]:
            self.path_ = np.append(self.path_, -res.fun)
            return self._unpack(res.x)
        return params

    # ---------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> MixtureForecast:
        D = self._design(X)
        mu, lv, lpi = self._components(D, *self.params_)
        return MixtureForecast(np.exp(lpi), mu, np.exp(0.5 * lv))

    def responsibilities(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Posterior regime probabilities P(k | y_t, x_t) (ex-post, for interpretation only)."""
        D = self._design(X)
        lc, ll = self._loglik_terms(D, y.to_numpy(float), *self.params_)
        return pd.DataFrame(np.exp(lc - ll[:, None]), index=X.index, columns=self.expert_names)

    def gate_probs(self, X: pd.DataFrame) -> pd.DataFrame:
        D = self._design(X)
        _, _, lpi = self._components(D, *self.params_)
        return pd.DataFrame(np.exp(lpi), index=X.index, columns=self.expert_names)

    # --------------------------------------------------------- white box
    def attribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact additive decomposition of the predictive mean.

        E[y|x] = sum_k pi_k beta_k0 + sum_k sum_j pi_k beta_kj x_kj (standardised x).
        Columns sum exactly to `predict(X).mean()`.
        """
        D = self._design(X)
        beta, delta, gamma = self.params_
        pi = np.exp(self._components(D, *self.params_)[2])
        out = {}
        for k, (name, cols) in enumerate(self.experts.items()):
            out[f"{name}:const"] = pi[:, k] * beta[k][0]
            for j, c in enumerate(cols):
                out[f"{name}:{c}"] = pi[:, k] * beta[k][j + 1] * D.Xk[k][:, j + 1]
        return pd.DataFrame(out, index=X.index)

    def gate_elasticity(self, X: pd.DataFrame) -> pd.DataFrame:
        """d E[y|x] / d z_j through the gate only (per one standard deviation of z_j)."""
        D = self._design(X)
        beta, delta, gamma = self.params_
        mu, lv, lpi = self._components(D, *self.params_)
        pi = np.exp(lpi)
        G = np.vstack([gamma, np.zeros((1, gamma.shape[1]))])[:, 1:]  # (K, m)
        gbar = pi @ G  # (n, m)
        el = np.einsum("nk,nk,nkm->nm", pi, mu, G[None, :, :] - gbar[:, None, :])
        return pd.DataFrame(el, index=X.index, columns=self.gate_cols)

    # ------------------------------------------------------------ inference
    def _theta(self, params):
        beta, delta, gamma = params
        return np.concatenate([*beta, *delta, gamma.ravel()])

    def _unpack(self, theta):
        beta, i = [], 0
        for c in self.experts.values():
            beta.append(theta[i:i + 1 + len(c)])
            i += 1 + len(c)
        q = 1 + len(self.var_cols)
        delta = [theta[i + k * q:i + (k + 1) * q] for k in range(self.K)]
        i += self.K * q
        gamma = theta[i:].reshape(self.K - 1, 1 + len(self.gate_cols)) if self.K > 1 else np.zeros((0, 1 + len(self.gate_cols)))
        return beta, delta, gamma

    def param_names(self):
        names = []
        for k, cols in self.experts.items():
            names += [f"beta[{k}]:const"] + [f"beta[{k}]:{c}" for c in cols]
        for k in self.experts:
            names += [f"delta[{k}]:const"] + [f"delta[{k}]:{c}" for c in self.var_cols]
        for k in self.expert_names[:-1]:
            names += [f"gamma[{k}]:const"] + [f"gamma[{k}]:{c}" for c in self.gate_cols]
        return names

    def _scores(self, theta, D, y):
        """Per-observation scores of the observed log-likelihood (Fisher's identity)."""
        beta, delta, gamma = self._unpack(theta)
        mu, lv, lpi = self._components(D, beta, delta, gamma)
        lc = lpi - 0.5 * (LOG2PI + lv + (y[:, None] - mu) ** 2 / np.exp(lv))
        tau = np.exp(lc - logsumexp(lc, axis=1, keepdims=True))
        s = []
        for k in range(self.K):
            s.append((tau[:, k] * (y - mu[:, k]) / np.exp(lv[:, k]))[:, None] * D.Xk[k])
        for k in range(self.K):
            u = (y - mu[:, k]) ** 2 / np.exp(lv[:, k])
            s.append((tau[:, k] * 0.5 * (u - 1))[:, None] * D.V)
        pi = np.exp(lpi)
        for k in range(self.K - 1):
            s.append((tau[:, k] - pi[:, k])[:, None] * D.Z)
        return np.hstack(s)

    def sandwich_cov(self, X: pd.DataFrame, y: pd.Series, hac_lags: int | None = None) -> tuple:
        """(theta, V): parameter vector and its QMLE sandwich covariance with Newey-West meat."""
        ok = y.notna()
        D = self._design(X[ok])
        yv = y[ok].to_numpy(float)
        th = self._theta(self.params_)
        S = self._scores(th, D, yv)
        n, P = S.shape
        L = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        B = S.T @ S
        for l in range(1, L + 1):
            G = S[l:].T @ S[:-l]
            B += (1 - l / (L + 1)) * (G + G.T)
        A = np.zeros((P, P))
        for j in range(P):
            e = np.zeros(P)
            e[j] = 1e-5
            A[:, j] = (self._scores(th + e, D, yv).sum(0) - self._scores(th - e, D, yv).sum(0)) / 2e-5
        Ainv = np.linalg.pinv(0.5 * (A + A.T))
        return th, Ainv @ B @ Ainv

    def wald(self, theta, V, select) -> dict:
        """Wald test that the parameters whose names satisfy `select` are jointly zero."""
        idx = [i for i, n in enumerate(self.param_names()) if select(n)]
        g = theta[idx]
        W = float(g @ np.linalg.solve(V[np.ix_(idx, idx)], g))
        from scipy.stats import chi2
        return {"W": W, "df": len(idx), "p": float(chi2.sf(W, len(idx)))}

    def lm_gate_test(self, X: pd.DataFrame, y: pd.Series, cols: list, hac_lags: int | None = None) -> dict:
        """Robust LM (score) test that candidate state variables `cols` are absent from the gate.

        Only the fitted (restricted) model is needed. The new parameters' score is
        sum_t (tau_tk - pi_tk) w_t; the statistic s2' (C B C')^{-1} s2 with C = [-A21 A11^{-1}, I]
        accounts for estimation of the restricted parameters and is chi2 with (K-1)*len(cols) df.
        """
        if self.K < 2:
            raise ValueError("the gate test needs K >= 2")
        ok = y.notna()
        X, yv = X[ok], y[ok].to_numpy(float)
        aug = SMoE(experts=self.experts, var_cols=self.var_cols, gate_cols=self.gate_cols + list(cols),
                   lam=self.lam, lam_v=self.lam_v)
        aug._cols = aug._all_cols()
        fresh = [c for c in cols if c not in self.loc_.index]
        aug.loc_ = pd.concat([self.loc_, X[fresh].mean()])[aug._cols]
        aug.scale_ = pd.concat([self.scale_, X[fresh].std().replace(0, 1.0)])[aug._cols]
        aug.delta0_ = self.delta0_
        beta, delta, gamma = self.params_
        gamma_aug = np.hstack([gamma, np.zeros((self.K - 1, len(cols)))])
        aug.params_ = (beta, delta, gamma_aug)
        D = aug._design(X)
        th = aug._theta(aug.params_)
        names = aug.param_names()
        new = np.array([n.startswith("gamma") and n.split(":", 1)[1] in cols for n in names])
        S = aug._scores(th, D, yv)
        n, P = S.shape
        L = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        B = S.T @ S
        for l in range(1, L + 1):
            G = S[l:].T @ S[:-l]
            B += (1 - l / (L + 1)) * (G + G.T)
        A = np.zeros((P, P))
        for j in range(P):
            e = np.zeros(P)
            e[j] = 1e-5
            A[:, j] = -(aug._scores(th + e, D, yv).sum(0) - aug._scores(th - e, D, yv).sum(0)) / 2e-5
        A = 0.5 * (A + A.T)
        C = np.hstack([-A[np.ix_(new, ~new)] @ np.linalg.pinv(A[np.ix_(~new, ~new)]), np.eye(new.sum())])
        order = np.r_[np.where(~new)[0], np.where(new)[0]]
        Bo = B[np.ix_(order, order)]
        s2 = C @ S[:, order].sum(0)  # efficient score; equals the raw new-parameter score when s1 = 0
        LM = float(s2 @ np.linalg.solve(C @ Bo @ C.T, s2))
        from scipy.stats import chi2
        return {"LM": LM, "df": int(new.sum()), "p": float(chi2.sf(LM, int(new.sum())))}

    def inference(self, X: pd.DataFrame, y: pd.Series, hac_lags: int | None = None) -> pd.DataFrame:
        """QMLE sandwich standard errors with Newey-West HAC meat."""
        ok = y.notna()
        D = self._design(X[ok])
        yv = y[ok].to_numpy(float)
        th = self._theta(self.params_)
        S = self._scores(th, D, yv)
        n, P = S.shape
        L = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        B = S.T @ S
        for l in range(1, L + 1):
            G = S[l:].T @ S[:-l]
            B += (1 - l / (L + 1)) * (G + G.T)
        A = np.zeros((P, P))
        h = 1e-5
        for j in range(P):
            e = np.zeros(P)
            e[j] = h
            A[:, j] = (self._scores(th + e, D, yv).sum(0) - self._scores(th - e, D, yv).sum(0)) / (2 * h)
        A = 0.5 * (A + A.T)
        Ainv = np.linalg.pinv(A)
        Vs = Ainv @ B @ Ainv
        se = np.sqrt(np.clip(np.diag(Vs), 0, None))
        se_mle = np.sqrt(np.clip(np.diag(-Ainv), 0, None))
        z = th / se
        return pd.DataFrame({"estimate": th, "se_sandwich": se, "se_mle": se_mle, "z": z,
                             "p": 2 * norm.sf(np.abs(z))}, index=self.param_names())

    @property
    def aic(self):
        return -2 * self.loglik_ + 2 * self.n_params_

    @property
    def bic(self):
        return -2 * self.loglik_ + np.log(self.n_obs_) * self.n_params_


def single_gaussian(cols=None, var_cols=None, name="Gauss-HAR") -> SMoE:
    """K = 1 nested model: one linear-Gaussian expert with all mean features."""
    cols = cols if cols is not None else SEMANTIC_GROUPS["trend"] + SEMANTIC_GROUPS["reversal"]
    return SMoE(experts={"single": list(cols)}, var_cols=list(var_cols or SEMANTIC_GROUPS["var"]),
                gate_cols=[], n_init=1, name=name)


def constant_gate(**kw) -> SMoE:
    """Mixture autoregression with constant weights (Wong & Li, 2000 type)."""
    return SMoE(gate_cols=[], name="MAR", **kw)


def lstar(**kw) -> SMoE:
    """Two-expert logistic smooth transition between trend and reversal."""
    return SMoE(experts={"trend": SEMANTIC_GROUPS["trend"], "reversal": SEMANTIC_GROUPS["reversal"]},
                gate_cols=["g_lvol"], name="LSTAR", **kw)
