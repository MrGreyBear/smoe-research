import numpy as np
import pandas as pd

from smoe.backtest import CostModel, run_backtest, rule_meanvar, perf, pbo_cscv
from smoe.distributions import MixtureForecast


def test_pnl_uses_next_day_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=4)
    ohlc = pd.DataFrame({"open": [100, 101, 103, 102], "close": [100, 102, 103, 101]}, index=idx, dtype=float)
    w = np.array([1.0, 1.0, 0.0])
    bt = run_backtest(idx[:3], w, ohlc, CostModel(10.0))
    # decided at close of day0, executed at open of day1: earns intraday day1 only
    assert np.isclose(bt["gross"].iloc[0], np.log(102 / 101))
    assert np.isclose(bt["cost"].iloc[0], 10 / 1e4)
    # held through night 1->2, then full day 2
    assert np.isclose(bt["gross"].iloc[1], np.log(103 / 102) + np.log(103 / 103))
    # closed at open of day3: only the overnight gap accrues
    assert np.isclose(bt["gross"].iloc[2], np.log(102 / 103))


def test_no_trade_band_reduces_turnover():
    rng = np.random.default_rng(0)
    T = 500
    fc = MixtureForecast.gaussian(rng.normal(0.02, 0.05, T), np.full(T, 1.2))
    w0 = rule_meanvar(fc, cost_bps=0.0, hi=1.0)
    w1 = rule_meanvar(fc, cost_bps=20.0, hi=1.0)
    assert np.abs(np.diff(w1)).sum() < np.abs(np.diff(w0)).sum()


def test_pbo_high_for_pure_noise_strategies():
    # for pure noise the selected configuration is no better than random out of sample: E[PBO] = 1/2
    vals = [pbo_cscv(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (800, 20))), S=8) for s in range(20)]
    assert 0.35 < np.mean(vals) < 0.65


def test_perf_basic():
    r = pd.Series(np.full(252, 0.001))
    r.iloc[::2] = -0.0005
    p = perf(r)
    assert p["sharpe"] > 0 and p["max_dd"] <= 0
