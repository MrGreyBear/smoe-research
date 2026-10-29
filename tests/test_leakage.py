"""No feature at date t may change when data after t is altered."""

import numpy as np
import pandas as pd

from smoe.features import build_features
from smoe.data.realized import realized_measures


def synthetic_ohlc(n=600, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.012, n)
    close = 3000 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_features_do_not_use_future_rows():
    df = synthetic_ohlc()
    cut = df.index[450]
    base = build_features(df)
    corrupted = df.copy()
    corrupted.loc[corrupted.index > cut, ["open", "high", "low", "close"]] *= 1.5
    alt = build_features(corrupted)
    cols = base.X.columns
    pd.testing.assert_frame_equal(base.X.loc[:cut, cols], alt.X.loc[:cut, cols])
    pd.testing.assert_series_equal(base.r.loc[:cut], alt.r.loc[:cut])


def test_target_is_next_day_return():
    df = synthetic_ohlc()
    fs = build_features(df)
    r = 100 * np.log(df["close"]).diff()
    t = fs.y.index[100]
    nxt = df.index[df.index.get_loc(t) + 1]
    assert np.isclose(fs.y.loc[t], r.loc[nxt])


def test_realized_measures_identities():
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=5):
        times = pd.date_range(d + pd.Timedelta("9h35min"), periods=48, freq="5min")
        px = 3000 * np.exp(np.cumsum(rng.normal(0, 0.001, 48)))
        rows.append(pd.DataFrame({"trade_time": times, "open": px, "close": px}))
    rm = realized_measures(pd.concat(rows))
    np.testing.assert_allclose(rm["rv"], rm["rs_pos"] + rm["rs_neg"])
    assert (rm["bv"] > 0).all() and (rm["jump"] >= 0).all()
    # duplicated bars and a zero-price bar must not change anything
    dirty = pd.concat(rows + [rows[2]])
    dirty.iloc[5, dirty.columns.get_loc("close")] = 0.0
    rm2 = realized_measures(dirty)
    assert np.isfinite(rm2.to_numpy(float)[1:]).all()
    np.testing.assert_allclose(rm2["bv"].iloc[2], rm["bv"].iloc[2])


def test_realized_features_and_vix_are_causal():
    from smoe.features.extra import add_realized, add_vix
    df = synthetic_ohlc(500)
    fs = build_features(df)
    rng = np.random.default_rng(0)
    rm = pd.DataFrame({"rv": rng.gamma(2, 0.5, len(df)), "bv": rng.gamma(2, 0.4, len(df)),
                       "jump": rng.gamma(1, 0.1, len(df)), "rs_pos": rng.gamma(2, 0.25, len(df)),
                       "rs_neg": rng.gamma(2, 0.25, len(df)), "ret_overnight": rng.normal(0, 0.3, len(df))},
                      index=df.index)
    vix = pd.Series(np.exp(rng.normal(3, 0.2, len(df))), index=df.index)
    cut = df.index[400]
    base = add_vix(add_realized(fs, rm), vix)
    rm2, vix2 = rm.copy(), vix.copy()
    rm2.loc[rm2.index > cut] *= 3.0
    vix2.loc[vix2.index >= cut] *= 3.0  # VIX of date t itself must not be used at t
    alt = add_vix(add_realized(fs, rm2), vix2)
    pd.testing.assert_frame_equal(base.X.loc[:cut], alt.X.loc[:cut])
