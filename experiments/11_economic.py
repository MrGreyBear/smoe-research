"""Economic value: trading the CSI 300 ETF (long-only) and IF futures (long/short) from
CSI 300 predictive distributions, plus proxy trading on every other asset.

Fixed a priori: target daily vol 1.0%, gamma = 5, ETF cost 3bp, futures cost 2bp, other assets 2bp.
Sensitivity grid (all counted in DSR trials and PBO): cost in {0,3,10,20}bp, gamma in {2,5,10},
target vol in {0.8,1.0,1.2}.
"""

import itertools
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import ASSETS, load_panels  # noqa: E402
from smoe import FIGURES, TABLES  # noqa: E402
from smoe import plotstyle as ps  # noqa: E402
from smoe.backtest import (CostModel, RULES, deflated_sharpe, pbo_cscv, perf, run_backtest,  # noqa: E402
                           stationary_bootstrap_ci)
from smoe.data import load_daily  # noqa: E402

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
start = sys.argv[2] if len(sys.argv) > 2 else None
sfx = "" if period == "dev" and start is None else f"_{period}" + (f"_{start}" if start else "")
MODELS = ["SMoE", "Deep-SMoE", "GARCH-t", "AR-GJR-t", "Gauss-HAR", "MAR", "MS-AR", "XGB-Gauss", "HistGauss",
          "Chronos-Bolt-small"]
BASE = {"target_vol": 1.0, "gamma": 5.0}
GRID = list(itertools.product([0.0, 3.0, 10.0, 20.0], [2.0, 5.0, 10.0], [0.8, 1.0, 1.2]))


def utility_fee(r_s, r_b, gamma=5.0, ann=252):
    """Annualised fee (in %) equating quadratic utility of strategy and benchmark (Fleming et al., 2001)."""
    u = lambda r: np.mean(r) - gamma / 2 * np.var(r)  # noqa: E731
    return (u(r_s) - u(r_b)) * ann * 100


def trade(signal_asset, trade_asset, lo, hi, base_cost):
    P = load_panels(period, signal_asset, MODELS)
    ohlc = load_daily(trade_asset)
    rows, series = [], {}
    for m, p in P.items():
        if start:
            p = p.loc(start, None)
        for rule, f in RULES.items():
            w = f(p.fc, target_vol=BASE["target_vol"], gamma=BASE["gamma"], cost_bps=base_cost, lo=lo, hi=hi)
            bt = run_backtest(p.index, w, ohlc, CostModel(base_cost))
            if bt.empty:
                continue
            pf = perf(bt["net"])
            pb = perf(bt["bench"])
            ci = stationary_bootstrap_ci((bt["net"] - bt["bench"]).to_numpy(), B=500)
            rows.append({"signal": signal_asset, "traded": trade_asset, "model": m, "rule": rule, **pf,
                         "bench_sharpe": pb["sharpe"], "bench_ann_ret": pb["ann_ret"], "bench_max_dd": pb["max_dd"],
                         "turnover_ann": bt["w"].diff().abs().mean() * 252, "avg_w": bt["w"].mean(),
                         "cost_drag": bt["cost"].mean() * 252, "fee_bp": utility_fee(bt["net"], bt["bench"]) * 100,
                         "excess_ci_lo": ci[0], "excess_ci_hi": ci[1]})
            series[(m, rule)] = bt
    return pd.DataFrame(rows), series


if __name__ == "__main__":
    ps.use()
    main_etf, ser_etf = trade("csi300", "etf510300", 0.0, 1.0, 3.0)
    main_if, ser_if = trade("csi300", "if_main", -1.0, 1.0, 2.0)
    main = pd.concat([main_etf, main_if])
    main.to_csv(TABLES / f"econ_main{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(main.sort_values(["traded", "sharpe"], ascending=[True, False]).round(3).to_string())

    # ---- sensitivity grid on the ETF (every configuration is a 'trial')
    P = load_panels(period, "csi300", MODELS)
    ohlc = load_daily("etf510300")
    trials, grid_rows = {}, []
    for m, p in P.items():
        if start:
            p = p.loc(start, None)
        for rule, f in RULES.items():
            for c, g, tv in GRID:
                w = f(p.fc, target_vol=tv, gamma=g, cost_bps=c, lo=0.0, hi=1.0)
                bt = run_backtest(p.index, w, ohlc, CostModel(c))
                key = f"{m}|{rule}|c{c}|g{g}|v{tv}"
                trials[key] = bt["net"]
                grid_rows.append({"model": m, "rule": rule, "cost_bp": c, "gamma": g, "target_vol": tv,
                                  **perf(bt["net"])})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(TABLES / f"econ_grid{sfx}.csv", index=False)
    Mx = pd.DataFrame(trials).dropna()
    sr_all = grid["sharpe"].to_numpy()
    best = grid.loc[grid["sharpe"].idxmax()]
    bkey = f"{best['model']}|{best['rule']}|c{best['cost_bp']}|g{best['gamma']}|v{best['target_vol']}"
    r = Mx[bkey].to_numpy()
    dsr = deflated_sharpe(best["sharpe"], sr_all, len(r), stats.skew(r), stats.kurtosis(r, fisher=False))
    # PBO over configurations at the realistic cost level only (3bp)
    real = [k for k in Mx.columns if "|c3.0|" in k]
    pbo = pbo_cscv(Mx[real], S=16)
    # DSR of the best configuration at the realistic cost, deflated by the 3bp configurations
    g3 = grid[grid.cost_bp == 3.0]
    b3 = g3.loc[g3["sharpe"].idxmax()]
    k3 = f"{b3['model']}|{b3['rule']}|c3.0|g{b3['gamma']}|v{b3['target_vol']}"
    r3 = Mx[k3].to_numpy()
    dsr3 = deflated_sharpe(b3["sharpe"], g3["sharpe"].to_numpy(), len(r3), stats.skew(r3),
                           stats.kurtosis(r3, fisher=False))
    summ = pd.DataFrame([{"n_trials": len(sr_all), "best_config": bkey, "best_sharpe": best["sharpe"],
                          "DSR_all": dsr, "best_config_3bp": k3, "best_sharpe_3bp": b3["sharpe"], "DSR": dsr3,
                          "PBO_3bp": pbo, "n_configs_pbo": len(real)}])
    summ.to_csv(TABLES / f"econ_overfitting{sfx}.csv", index=False)
    print(summ.to_string())

    # ---- equity curves for the headline rule
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    any_bt = next(iter(ser_etf.values()))
    ax.plot(any_bt.index, np.exp(any_bt["bench"].cumsum()), color=ps.MUTED, lw=1.0, label="buy & hold ETF")
    for i, m in enumerate(["SMoE", "GARCH-t", "Deep-SMoE"]):
        if (m, "vt_tilt") in ser_etf:
            bt = ser_etf[(m, "vt_tilt")]
            ax.plot(bt.index, np.exp(bt["net"].cumsum()), color=ps.SERIES[i], label=f"{m} (vol target + tilt)")
    ax.set_yscale("log")
    ax.set_ylabel("wealth (log scale)")
    ax.legend(loc="upper left")
    ps.save(fig, FIGURES / f"econ_equity_etf{sfx}.pdf")

    # ---- proxy trading on every asset (vt_tilt, long-only for equities/ETF, long-short otherwise)
    rows = []
    for a in ASSETS:
        lo = 0.0 if a in ("csi300", "csi500", "sse50", "chinext", "spx", "ndx", "hsi", "n225", "dax", "ftse",
                          "etf510300") else -1.0
        P = load_panels(period, a, ["SMoE", "GARCH-t", "HistGauss", "Deep-SMoE"])
        ohlc = load_daily(a)
        for m, p in P.items():
            if start:
                p = p.loc(start, None)
            for rule in ("voltarget", "vt_tilt", "sign"):
                w = RULES[rule](p.fc, target_vol=BASE["target_vol"], lo=lo, hi=1.0)
                bt = run_backtest(p.index, w, ohlc, CostModel(2.0))
                pf, pb = perf(bt["net"]), perf(bt["bench"])
                rows.append({"asset": a, "model": m, "rule": rule, "sharpe": pf["sharpe"], "sharpe_se": pf["sharpe_se"],
                             "max_dd": pf["max_dd"], "bench_sharpe": pb["sharpe"], "bench_max_dd": pb["max_dd"]})
    pd.DataFrame(rows).to_csv(TABLES / f"econ_assets{sfx}.csv", index=False)
    print("economic analysis complete")
