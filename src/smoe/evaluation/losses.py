"""Per-period scoring rules (all negatively oriented: lower is better)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe.distributions import MixtureForecast


def nll(fc: MixtureForecast, y) -> np.ndarray:
    return -fc.logpdf(y)


def crps(fc: MixtureForecast, y) -> np.ndarray:
    return fc.crps(y)


def se_mean(fc: MixtureForecast, y) -> np.ndarray:
    return (np.asarray(y) - fc.mean()) ** 2


def qlike(var_fc, proxy) -> np.ndarray:
    """Patton (2011) robust QLIKE: proxy/h - log(proxy/h) - 1 (zero at h = proxy)."""
    ratio = np.asarray(proxy) / np.asarray(var_fc)
    ratio = np.clip(ratio, 1e-12, None)
    return ratio - np.log(ratio) - 1


def mse_var(var_fc, proxy) -> np.ndarray:
    return (np.asarray(proxy) - np.asarray(var_fc)) ** 2


def brier(fc: MixtureForecast, y) -> np.ndarray:
    return (fc.prob_up() - (np.asarray(y) > 0)) ** 2


def pinball(q, y, a: float) -> np.ndarray:
    y, q = np.asarray(y), np.asarray(q)
    return (a - (y < q)) * (y - q)


def fz0(y, var_q, es, a: float) -> np.ndarray:
    """Fissler-Ziegel FZ0 joint loss for (VaR, ES) (Patton, Ziegel & Chen, 2019); requires es < 0."""
    y, v, e = map(np.asarray, (y, var_q, es))
    e = np.minimum(e, -1e-8)
    return -((y <= v) * (v - y)) / (a * e) + v / e + np.log(-e) - 1


def all_losses(fc: MixtureForecast, y, rv=None, alphas=(0.01, 0.05)) -> pd.DataFrame:
    """Every loss used in the paper, one column per loss. `rv` is a variance proxy (defaults to y^2)."""
    y = np.asarray(y, float)
    proxy = y**2 if rv is None else np.asarray(rv, float)
    out = {
        "NLL": nll(fc, y),
        "CRPS": crps(fc, y),
        "SE": se_mean(fc, y),
        "Brier": brier(fc, y),
        "QLIKE": qlike(fc.var(), proxy),
        "MSE_var": mse_var(fc.var(), proxy),
    }
    for a in alphas:
        q, e = fc.quantile(a), fc.es(a)
        out[f"Pinball{int(a * 100)}"] = pinball(q, y, a)
        out[f"FZ{int(a * 100)}"] = fz0(y, q, e, a)
    return pd.DataFrame(out)
