"""Gaussian-mixture predictive distributions with closed-form functionals.

All models in the project return a `MixtureForecast`: for each date t a
K-component mixture  sum_k w_tk N(mu_tk, sd_tk^2). A single Gaussian is K = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass
class MixtureForecast:
    w: np.ndarray  # (T, K), rows sum to 1
    mu: np.ndarray  # (T, K)
    sd: np.ndarray  # (T, K), > 0

    def __post_init__(self):
        self.w = np.atleast_2d(np.asarray(self.w, float))
        self.mu = np.atleast_2d(np.asarray(self.mu, float))
        self.sd = np.atleast_2d(np.asarray(self.sd, float))

    @classmethod
    def gaussian(cls, mu, sd) -> "MixtureForecast":
        mu = np.asarray(mu, float).reshape(-1, 1)
        return cls(np.ones_like(mu), mu, np.asarray(sd, float).reshape(-1, 1))

    def __len__(self):
        return self.w.shape[0]

    def subset(self, idx) -> "MixtureForecast":
        return MixtureForecast(self.w[idx], self.mu[idx], self.sd[idx])

    # ---- moments ---------------------------------------------------------
    def mean(self):
        return np.sum(self.w * self.mu, axis=1)

    def within_var(self):
        """E_k[sigma_k^2]: risk inside the regimes."""
        return np.sum(self.w * self.sd**2, axis=1)

    def between_var(self):
        """Var_k[mu_k]: risk from disagreement between regimes."""
        m = self.mean()[:, None]
        return np.sum(self.w * (self.mu - m) ** 2, axis=1)

    def var(self):
        return self.within_var() + self.between_var()

    def skew(self):
        m = self.mean()[:, None]
        d = self.mu - m
        m3 = np.sum(self.w * (d**3 + 3 * d * self.sd**2), axis=1)
        return m3 / self.var() ** 1.5

    def kurt(self):
        m = self.mean()[:, None]
        d = self.mu - m
        s2 = self.sd**2
        m4 = np.sum(self.w * (d**4 + 6 * d**2 * s2 + 3 * s2**2), axis=1)
        return m4 / self.var() ** 2

    # ---- distribution ----------------------------------------------------
    def cdf(self, y):
        y = np.asarray(y, float).reshape(-1, 1)
        return np.sum(self.w * norm.cdf((y - self.mu) / self.sd), axis=1)

    def prob_up(self):
        return 1.0 - self.cdf(np.zeros(len(self)))

    def logpdf(self, y):
        y = np.asarray(y, float).reshape(-1, 1)
        with np.errstate(divide="ignore"):
            lw = np.log(self.w)
        return logsumexp(lw + norm.logpdf(y, self.mu, self.sd), axis=1)

    def quantile(self, a: float, tol: float = 1e-10, iters: int = 200):
        """Vectorised bisection on the mixture cdf."""
        lo = np.min(self.mu - 12 * self.sd, axis=1)
        hi = np.max(self.mu + 12 * self.sd, axis=1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < a
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            if np.max(hi - lo) < tol:
                break
        return 0.5 * (lo + hi)

    def es(self, a: float):
        """Lower-tail expected shortfall E[Y | Y <= q_a] (a negative number for losses)."""
        q = self.quantile(a)[:, None]
        z = (q - self.mu) / self.sd
        part = self.w * (self.mu * norm.cdf(z) - self.sd * norm.pdf(z))
        return np.sum(part, axis=1) / a

    def crps(self, y):
        """Closed-form CRPS of a Gaussian mixture (Grimit et al., 2006)."""
        y = np.asarray(y, float).reshape(-1, 1)

        def A(m, s):
            return 2 * s * norm.pdf(m / s) + m * (2 * norm.cdf(m / s) - 1)

        t1 = np.sum(self.w * A(y - self.mu, self.sd), axis=1)
        dm = self.mu[:, :, None] - self.mu[:, None, :]
        ds = np.sqrt(self.sd[:, :, None] ** 2 + self.sd[:, None, :] ** 2)
        ww = self.w[:, :, None] * self.w[:, None, :]
        t2 = 0.5 * np.sum(ww * A(dm, ds), axis=(1, 2))
        return t1 - t2

    def pit(self, y):
        return self.cdf(y)

    def sample(self, rng: np.random.Generator, n: int):
        T, K = self.w.shape
        u = rng.random((T, n))
        comp = (u[:, :, None] > np.cumsum(self.w, axis=1)[:, None, :]).sum(axis=2)
        comp = np.minimum(comp, K - 1)
        mu = np.take_along_axis(self.mu, comp, axis=1)
        sd = np.take_along_axis(self.sd, comp, axis=1)
        return mu + sd * rng.standard_normal((T, n))
