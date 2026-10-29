"""Optional covariates that need a second data source: 5-minute realised measures and the VIX.

Realised features (CSI 300 and instruments on it). Row t uses 5-minute bars of day t
only, which are complete at the close of t. All are logs of percent^2 quantities.
    lrv1, lrv5, lrv22     HAR components of realised variance RV
    lbv1, lbv5, lbv22     HAR components of bipower variation BV (continuous part)
    ljump                 log(1 + J_t), J_t = max(RV_t - BV_t, 0)
    lrsn1, lrsp1          negative and positive realised semivariance of day t
    ltv1, ltv5, ltv22     HAR components of RV + overnight^2 (close-to-close coverage)
    g_lrv                 gate volatility state: lrv22

VIX feature (any asset). g_vix at local date t is the log VIX close of the last US
trading date strictly before t, so it is known at the close of t in every time zone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe.features.semantic import FeatureSet

REALIZED_GROUPS = {
    "rv": ["lrv1", "lrv5", "lrv22"],
    "bv": ["lbv1", "lbv5", "lbv22"],
    "jump": ["ljump"],
    "semi": ["lrsn1", "lrsp1"],
    "tv": ["ltv1", "ltv5", "ltv22"],
}

FLOOR = 1e-4


def _har(x: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}1": np.log(x.clip(lower=FLOOR)),
                         f"{prefix}5": np.log(x.rolling(5).mean().clip(lower=FLOOR)),
                         f"{prefix}22": np.log(x.rolling(22).mean().clip(lower=FLOOR))})


def realized_features(rm: pd.DataFrame) -> pd.DataFrame:
    """rm: output of `realized_measures`, indexed by trading day."""
    on2 = rm["ret_overnight"].fillna(0.0) ** 2
    f = pd.concat([_har(rm["rv"], "lrv"), _har(rm["bv"], "lbv"), _har(rm["rv"] + on2, "ltv")], axis=1)
    f["ljump"] = np.log1p(rm["jump"])
    f["lrsn1"] = np.log(rm["rs_neg"].clip(lower=FLOOR))
    f["lrsp1"] = np.log(rm["rs_pos"].clip(lower=FLOOR))
    f["g_lrv"] = f["lrv22"]
    return f


def _extend(fs: FeatureSet, extra: pd.DataFrame) -> FeatureSet:
    X = fs.X.join(extra, how="left")
    ok = X.notna().all(axis=1)
    return FeatureSet(X[ok], fs.y[ok], fs.r[ok])


def add_realized(fs: FeatureSet, rm: pd.DataFrame) -> FeatureSet:
    """Append realised features of the same calendar day; rows without 5-minute data are dropped."""
    return _extend(fs, realized_features(rm).reindex(fs.X.index))


def vix_state(index: pd.DatetimeIndex, vix_close: pd.Series) -> pd.Series:
    """log VIX at the last US date strictly before each date in `index`."""
    v = np.log(vix_close.dropna().sort_index())
    left = pd.DataFrame({"date": index})
    right = pd.DataFrame({"vdate": v.index, "g_vix": v.to_numpy()})
    m = pd.merge_asof(left, right, left_on="date", right_on="vdate", allow_exact_matches=False)
    return pd.Series(m["g_vix"].to_numpy(), index=index, name="g_vix")


def add_vix(fs: FeatureSet, vix_close: pd.Series) -> FeatureSet:
    return _extend(fs, vix_state(fs.X.index, vix_close).to_frame())
