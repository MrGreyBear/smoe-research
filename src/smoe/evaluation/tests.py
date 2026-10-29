"""Forecast-evaluation tests used throughout the paper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def nw_lrv(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West long-run variance of a mean-zero-ised series."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = x @ x / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / n
    return max(v, 1e-300)


def dm_test(loss_a, loss_b, h: int = 1, lags: int | None = None) -> dict:
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction.

    d = loss_a - loss_b; positive statistic => model b has lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    stat = d.mean() / np.sqrt(nw_lrv(d, lags if lags is not None else max(h - 1, 0) or None) / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"stat": stat, "p_two": 2 * stats.t.sf(abs(stat), n - 1), "p_b_better": stats.t.sf(stat, n - 1),
            "mean_diff": d.mean(), "n": n}


def clark_west(y, f_bench, f_model, lags: int | None = None) -> dict:
    """Clark & West (2007) MSPE-adjusted test for nested models; H1: larger model better."""
    y, fb, fm = map(lambda a: np.asarray(a, float), (y, f_bench, f_model))
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    n = len(adj)
    stat = adj.mean() / np.sqrt(nw_lrv(adj, lags) / n)
    return {"stat": stat, "p": stats.norm.sf(stat), "n": n}


def oos_r2(y, f_model, f_bench) -> float:
    y, fm, fb = map(lambda a: np.asarray(a, float), (y, f_model, f_bench))
    return 1 - np.sum((y - fm) ** 2) / np.sum((y - fb) ** 2)


def pesaran_timmermann(y, yhat) -> dict:
    """Pesaran & Timmermann (1992) test of directional predictability (H1: positive association)."""
    a = (np.asarray(y) > 0).astype(float)
    b = (np.asarray(yhat) > 0).astype(float)
    n = len(a)
    py, px = a.mean(), b.mean()
    p_hat = np.mean(a == b)
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n**2)
    denom = v_hat - v_star
    stat = (p_hat - p_star) / np.sqrt(denom) if denom > 0 else np.nan
    return {"hit_rate": p_hat, "stat": stat, "p": stats.norm.sf(stat) if np.isfinite(stat) else np.nan}


def giacomini_white(loss_a, loss_b, lags: int | None = None) -> dict:
    """Conditional predictive ability test (Giacomini & White, 2006), instruments h_t = (1, d_{t-1})."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    Z = np.c_[np.ones(len(d) - 1), d[:-1]]
    zd = Z * d[1:, None]
    n = len(zd)
    zbar = zd.mean(0)
    zc = zd - zbar
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    O = zc.T @ zc / n
    for l in range(1, L + 1):
        G = zc[l:].T @ zc[:-l] / n
        O += (1 - l / (L + 1)) * (G + G.T)
    stat = n * zbar @ np.linalg.solve(O, zbar)
    sign = np.sign(d.mean())
    return {"stat": stat, "p": stats.chi2.sf(stat, Z.shape[1]), "sign": sign}


def weighted_lr_test(logs_a, logs_b, y, weight: str = "none", lags=None) -> dict:
    """Amisano & Giacomini (2007) weighted likelihood ratio test on log scores.

    logs_* are log predictive densities (higher better). weight in {none, center, tails, left}.
    Positive statistic => model a better.
    """
    y = np.asarray(y, float)
    z = (y - y.mean()) / y.std()
    w = {"none": np.ones_like(z), "center": stats.norm.pdf(z), "tails": 1 - stats.norm.pdf(z) / stats.norm.pdf(0),
         "left": 1 - stats.norm.cdf(z)}[weight]
    d = w * (np.asarray(logs_a) - np.asarray(logs_b))
    n = len(d)
    stat = d.mean() / np.sqrt(nw_lrv(d, lags) / n)
    return {"stat": stat, "p_two": 2 * stats.norm.sf(abs(stat))}


def berkowitz(pit, eps: float = 1e-10) -> dict:
    """Berkowitz (2001) LR test: z = Phi^{-1}(PIT) ~ iid N(0,1) vs Gaussian AR(1)."""
    z = stats.norm.ppf(np.clip(np.asarray(pit, float), eps, 1 - eps))
    x, yv = z[:-1], z[1:]
    X = np.c_[np.ones_like(x), x]
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ b
    s2 = resid @ resid / len(yv)
    ll1 = np.sum(stats.norm.logpdf(yv, X @ b, np.sqrt(s2)))
    ll0 = np.sum(stats.norm.logpdf(yv))
    lr = 2 * (ll1 - ll0)
    return {"LR": lr, "p": stats.chi2.sf(lr, 3), "mu": b[0] / (1 - b[1]), "rho": b[1],
            "sigma": np.sqrt(s2 / (1 - b[1] ** 2))}


def kupiec(hits, a: float) -> dict:
    h = np.asarray(hits, bool)
    n, x = len(h), h.sum()
    pi = x / n
    ll0 = (n - x) * np.log(1 - a) + x * np.log(a)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return {"rate": pi, "LR_uc": lr, "p": stats.chi2.sf(lr, 1)}


def christoffersen(hits, a: float) -> dict:
    h = np.asarray(hits, int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(q, k0, k1):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return k0 * np.log(1 - q) + k1 * np.log(q)

    lr_ind = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    uc = kupiec(h.astype(bool), a)
    lr_cc = uc["LR_uc"] + lr_ind
    return {"LR_ind": lr_ind, "p_ind": stats.chi2.sf(lr_ind, 1), "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, 2)}


def mincer_zarnowitz(proxy, var_fc, lags=None) -> dict:
    """proxy_t = a + b h_t + e; HAC Wald test of (a, b) = (0, 1)."""
    yv, h = np.asarray(proxy, float), np.asarray(var_fc, float)
    X = np.c_[np.ones_like(h), h]
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    n = len(yv)
    S = X * e[:, None]
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    O = S.T @ S / n
    for l in range(1, L + 1):
        G = S[l:].T @ S[:-l] / n
        O += (1 - l / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ O @ Q / n
    r = b - np.array([0.0, 1.0])
    W = r @ np.linalg.solve(V, r)
    r2 = 1 - e @ e / np.sum((yv - yv.mean()) ** 2)
    return {"a": b[0], "b": b[1], "R2": r2, "Wald": W, "p": stats.chi2.sf(W, 2)}


def murphy_brier(p, outcome, bins: int = 10) -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    p, o = np.asarray(p, float), np.asarray(outcome, float)
    uniq = np.unique(p)
    if len(uniq) <= bins:  # exact decomposition when forecasts take few values
        idx = np.searchsorted(uniq, p)
        bins = len(uniq)
    else:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    obar = o.mean()
    rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel += m.sum() * (p[m].mean() - o[m].mean()) ** 2
            res += m.sum() * (o[m].mean() - obar) ** 2
    n = len(p)
    return {"brier": np.mean((p - o) ** 2), "reliability": rel / n, "resolution": res / n,
            "uncertainty": obar * (1 - obar)}


def _block_bootstrap_idx(n, B, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return idx


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, B: int = 2000, block: int | None = None,
                         seed: int = 0) -> pd.DataFrame:
    """Hansen, Lunde & Nason (2011) MCS with the T_max statistic and moving-block bootstrap.

    Returns the MCS p-value of every model; models with p >= alpha are in the set.
    """
    L = losses.dropna().to_numpy(float)
    names = list(losses.columns)
    n, M = L.shape
    block = block or max(int(round(n ** (1 / 3))), 2)
    rng = np.random.default_rng(seed)
    idx = _block_bootstrap_idx(n, B, block, rng)
    Lbar = L.mean(0)
    Lboot = np.stack([L[i].mean(0) for i in idx])  # (B, M)
    alive = list(range(M))
    pvals = {}
    pmax = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = Lbar[a] - Lbar[a].mean()
        dboot = Lboot[:, a] - Lboot[:, a].mean(1, keepdims=True) - dbar
        var = np.mean(dboot**2, axis=0)
        t = dbar / np.sqrt(var)
        tmax = t.max()
        tboot = (dboot / np.sqrt(var)).max(axis=1)
        p = np.mean(tboot >= tmax)
        pmax = max(pmax, p)
        worst = a[np.argmax(t)]
        pvals[names[worst]] = pmax
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    out = pd.DataFrame({"mean_loss": Lbar, "mcs_p": [pvals[nm] for nm in names]}, index=names)
    out["in_mcs"] = out["mcs_p"] >= alpha
    return out
