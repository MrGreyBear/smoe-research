"""Cost-aware daily backtests driven by predictive distributions.

Timing (conservative, no look-ahead): the forecast indexed by date t is made at the close of t.
The target position w_t is executed at the OPEN of t+1. Hence the P&L of day t+1 is
    w_{t-1} * overnight_{t+1} + w_t * intraday_{t+1} - cost(|w_t - w_{t-1}|)
with overnight_{t+1} = log(O_{t+1}/C_t) and intraday_{t+1} = log(C_{t+1}/O_{t+1}).
Returns are simple-approximated by log returns in percent / 100 (daily, small).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from smoe.distributions import MixtureForecast


@dataclass(frozen=True)
class CostModel:
    """One-way proportional cost in basis points (commission + half spread + slippage)."""
    bps: float = 3.0

    def __call__(self, dw):
        return np.abs(dw) * self.bps / 1e4


ETF_COST = CostModel(3.0)  # 510300: ~1bp commission + ~1.25bp half tick + slippage; no stamp duty on ETFs
FUT_COST = CostModel(2.0)  # IF: exchange fee ~0.23bp + ~0.5bp slippage, margin interest ignored


# ---------------------------------------------------------------- position rules
def rule_sign(fc: MixtureForecast, lo=0.0, hi=1.0, **_):
    return np.where(fc.prob_up() > 0.5, hi, lo)


def rule_voltarget(fc: MixtureForecast, target_vol=1.0, lo=0.0, hi=1.0, **_):
    """Moreira-Muir style: w = target daily vol / predicted vol (percent units)."""
    return np.clip(target_vol / np.sqrt(fc.var()), lo, hi)


def rule_meanvar(fc: MixtureForecast, gamma=5.0, cost_bps=3.0, lo=0.0, hi=1.0, **_):
    """Mean-variance with proportional-cost no-trade band (Proposition: no-trade band).

    Units: mean and variance in percent; cost in percent = bps / 100.
    """
    m, V = fc.mean(), fc.var()
    w_star = m / (gamma * V / 100.0)  # mean in %, variance in %^2 -> scale variance to % units
    band = (cost_bps / 100.0) / (gamma * V / 100.0)
    w = np.empty(len(m))
    prev = 0.0
    for t in range(len(m)):
        x = min(max(prev, w_star[t] - band[t]), w_star[t] + band[t])
        prev = float(np.clip(x, lo, hi))
        w[t] = prev
    return w


def rule_vt_tilt(fc: MixtureForecast, target_vol=1.0, lo=0.0, hi=1.0, **_):
    """Volatility target scaled by directional confidence 2 P(up) (equals the vol target at P(up)=0.5)."""
    return np.clip(target_vol / np.sqrt(fc.var()) * 2 * fc.prob_up(), lo, hi)


RULES = {"sign": rule_sign, "voltarget": rule_voltarget, "meanvar": rule_meanvar, "vt_tilt": rule_vt_tilt}


# ---------------------------------------------------------------- engine
def run_backtest(dates: pd.DatetimeIndex, w: np.ndarray, ohlc: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    """w[i] is the position decided at the close of dates[i]; P&L accrues on the next trading day."""
    px = ohlc[["open", "close"]].copy()
    on = np.log(px["open"] / px["close"].shift(1))
    intra = np.log(px["close"] / px["open"])
    nxt = px.index.searchsorted(dates, side="right")
    ok = nxt < len(px)
    dates, w, nxt = dates[ok], np.asarray(w)[ok], nxt[ok]
    d_next = px.index[nxt]
    w_prev = np.r_[0.0, w[:-1]]
    gross = w_prev * on.to_numpy()[nxt] + w * intra.to_numpy()[nxt]
    tc = cost(w - w_prev)
    out = pd.DataFrame({"w": w, "w_prev": w_prev, "gross": gross, "cost": tc, "net": gross - tc,
                        "bench": on.to_numpy()[nxt] + intra.to_numpy()[nxt]}, index=d_next)
    return out.dropna()


def perf(ret: pd.Series, ann: int = 252) -> dict:
    r = ret.to_numpy()
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd * np.sqrt(ann) if sd > 0 else np.nan
    eq = np.exp(np.cumsum(r))
    dd = eq / np.maximum.accumulate(eq) - 1
    # Lo (2002) iid SE of the annualised Sharpe ratio, adjusted for skew/kurtosis (Mertens 2002)
    s_d = mu / sd if sd > 0 else 0.0
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    se = np.sqrt((1 + 0.5 * s_d**2 - g3 * s_d + (g4 - 3) / 4 * s_d**2) / len(r)) * np.sqrt(ann)
    return {"ann_ret": mu * ann, "ann_vol": sd * np.sqrt(ann), "sharpe": sr, "sharpe_se": se,
            "max_dd": dd.min(), "calmar": mu * ann / abs(dd.min()) if dd.min() < 0 else np.nan, "n": len(r)}


def deflated_sharpe(sr_obs: float, sr_trials: np.ndarray, n_obs: int, skew: float, kurt: float, ann: int = 252):
    """Bailey & Lopez de Prado (2014) DSR: P(true SR > SR_0) where SR_0 is the expected max of N trials."""
    sr_d = np.asarray(sr_trials, float)
    sr_d = sr_d[np.isfinite(sr_d)] / np.sqrt(ann)  # zero-exposure configurations have undefined SR
    N = len(sr_d)
    v = np.var(sr_d, ddof=1) if N > 1 else 0.0
    emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    s = sr_obs / np.sqrt(ann)
    z = (s - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * s + (kurt - 1) / 4 * s**2)
    return float(stats.norm.cdf(z))


def pbo_cscv(M: pd.DataFrame, S: int = 16) -> float:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    M: (T, N) matrix of per-period returns of N strategy configurations.
    """
    from itertools import combinations

    X = M.to_numpy()
    T = (len(X) // S) * S
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for comb in combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.concatenate([blocks[i] for i in range(S) if i not in comb])

        def sr(idx):
            sd = X[idx].std(0, ddof=1)
            return np.where(sd > 0, X[idx].mean(0) / sd, -np.inf)

        best = np.argmax(sr(tr))
        rank = stats.rankdata(sr(te))[best] / (X.shape[1] + 1)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))


def stationary_bootstrap_ci(diff: np.ndarray, B: int = 2000, p: float = 1 / 20, seed: int = 0, ann: int = 252):
    """CI for the annualised mean of a return difference via the Politis-Romano stationary bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(B)
    for b in range(B):
        idx = np.empty(n, int)
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            idx[t] = i
            i = (i + 1) % n
        means[b] = diff[idx].mean()
    return np.percentile(means * ann, [2.5, 97.5])
