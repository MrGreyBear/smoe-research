"""Causal semantic feature groups.

Row t of every feature uses only prices up to and including the close of day t.
The target at row t is the next-day return r_{t+1}. A model trained on rows <= s
and asked to forecast row s+1 therefore never sees information after close s.

Groups
------
trend    : volatility-scaled time-series momentum (20d, 120d)
reversal : volatility-scaled 1d and 5d returns
var      : log HAR components of Parkinson range variance (1d, 5d, 22d)
gate     : state variables — log 22d variance, 60d momentum, drawdown from 250d high
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEMANTIC_GROUPS = {
    "trend": ["mom20", "mom120"],
    "reversal": ["rev1", "rev5"],
    "var": ["lv1", "lv5", "lv22"],
    "gate": ["g_lvol", "g_mom60", "g_dd"],
}

PARKINSON = 1.0 / (4.0 * np.log(2.0))


@dataclass
class FeatureSet:
    X: pd.DataFrame  # all feature columns, indexed by date t
    y: pd.Series  # r_{t+1}, indexed by date t (NaN on the last row)
    r: pd.Series  # r_t, same index

    def slice(self, start=None, end=None) -> "FeatureSet":
        s = slice(start, end)
        return FeatureSet(self.X.loc[s], self.y.loc[s], self.r.loc[s])


def range_variance(df: pd.DataFrame) -> pd.Series:
    """Parkinson variance in percent^2, floored by the squared close-to-close return share."""
    hl = 100 * np.log(df["high"] / df["low"])
    pv = PARKINSON * hl**2
    r2 = (100 * np.log(df["close"]).diff()) ** 2
    bad = ~np.isfinite(pv) | (pv <= 0)
    pv = pv.where(~bad, r2)
    return pv.clip(lower=1e-4)


def build_features(df: pd.DataFrame) -> FeatureSet:
    """df: daily OHLC indexed by date. Returns causal features and next-day target."""
    close = df["close"]
    r = 100 * np.log(close).diff()
    v = range_variance(df)
    sig = np.sqrt(v.rolling(22, min_periods=22).mean())  # daily vol estimate in %

    f = pd.DataFrame(index=df.index)
    f["mom20"] = r.rolling(20).sum() / (sig * np.sqrt(20))
    f["mom120"] = r.rolling(120).sum() / (sig * np.sqrt(120))
    f["rev1"] = r / sig
    f["rev5"] = r.rolling(5).sum() / (sig * np.sqrt(5))
    f["lv1"] = np.log(v)
    f["lv5"] = np.log(v.rolling(5).mean())
    f["lv22"] = np.log(v.rolling(22).mean())
    f["g_lvol"] = f["lv22"]
    f["g_mom60"] = r.rolling(60).sum() / (sig * np.sqrt(60))
    f["g_dd"] = np.log(close / close.rolling(250, min_periods=120).max())

    f = f.replace([np.inf, -np.inf], np.nan)
    y = r.shift(-1)
    ok = f.notna().all(axis=1) & r.notna()
    return FeatureSet(f[ok], y[ok].rename("y"), r[ok].rename("r"))
