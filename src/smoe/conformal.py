"""Adaptive conformal inference (Gibbs & Candes, 2021) for one-sided quantile (VaR) forecasts.

Given a sequence of predictive cdfs F_t and a target violation rate a, ACI issues
q_t = F_t^{-1}(alpha_t) and updates
    alpha_{t+1} = alpha_t + eta * (a - err_t),   err_t = 1{y_t < q_t}.
For any sequence (no distributional assumption, misspecified F_t allowed)
    | (1/T) sum_t err_t - a | <= (max(alpha_1, 1 - alpha_1) + eta) / (eta T).
Levels alpha_t <= 0 give q_t = -inf (no violation possible), alpha_t >= 1 give +inf.
"""

from __future__ import annotations

import numpy as np

from smoe.distributions import MixtureForecast


def aci_quantiles(fc: MixtureForecast, y, a: float, eta: float = 0.005, alpha1: float | None = None):
    y = np.asarray(y, float)
    T = len(y)
    alpha = a if alpha1 is None else alpha1
    q = np.empty(T)
    alphas = np.empty(T)
    err = np.empty(T)
    for t in range(T):
        alphas[t] = alpha
        if alpha <= 0:
            q[t] = -np.inf
        elif alpha >= 1:
            q[t] = np.inf
        else:
            q[t] = fc.subset(slice(t, t + 1)).quantile(alpha)[0]
        err[t] = float(y[t] < q[t])
        alpha = alpha + eta * (a - err[t])
    bound = (max(alphas[0], 1 - alphas[0]) + eta) / (eta * T)
    return {"q": q, "alpha": alphas, "hits": err.astype(bool), "rate": err.mean(), "bound": bound}
