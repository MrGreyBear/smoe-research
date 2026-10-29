"""Produce tomorrow's position for the CSI 300 ETF (510300) and IF futures from today's close.

Run after the close (15:00 Beijing time). Downloads fresh data, refits the SMoE on all history
(warm start from the cached model if present), and prints the predictive distribution, the
regime weights, the exact mean attribution and the target positions under each rule.
Orders are meant for the next day's opening auction, matching the backtest timing.

    python trading/live_signal.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smoe import RESULTS
from smoe.backtest import ETF_COST, rule_meanvar, rule_vt_tilt, rule_voltarget
from smoe.data.loaders import download_all, load_daily
from smoe.features import build_features
from smoe.models import SMoE

CACHE = RESULTS / "live" / "smoe_csi300.pkl"
CONFIG = {"target_vol": 1.0, "gamma": 5.0, "cost_bps": ETF_COST.bps}


def main(refresh: bool = True):
    if refresh:
        download_all(["csi300", "etf510300", "if_main"])
    fs = build_features(load_daily("csi300"))
    warm = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else None
    train = fs.y.notna()
    model = SMoE().fit(fs.X[train], fs.y[train], warm_start=warm)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(model))

    last = fs.X.iloc[[-1]]
    fc = model.predict(last)
    gates = model.gate_probs(last).iloc[0]
    attr = model.attribution(last).iloc[0]
    out = {
        "as_of_close": str(last.index[0].date()),
        "mean_%": float(fc.mean()[0]), "sd_%": float(np.sqrt(fc.var())[0]),
        "P(up)": float(fc.prob_up()[0]), "VaR5_%": float(fc.quantile(0.05)[0]), "ES5_%": float(fc.es(0.05)[0]),
        "within_var": float(fc.within_var()[0]), "between_var": float(fc.between_var()[0]),
        "w_etf_voltarget": float(rule_voltarget(fc, CONFIG["target_vol"])[0]),
        "w_etf_vt_tilt": float(rule_vt_tilt(fc, CONFIG["target_vol"])[0]),
        "w_etf_meanvar": float(rule_meanvar(fc, CONFIG["gamma"], CONFIG["cost_bps"])[0]),
        "w_if_vt_tilt": float(rule_vt_tilt(fc, CONFIG["target_vol"], lo=-1.0, hi=1.0)[0]),
    }
    print(pd.Series(out).to_string())
    print("\nregime weights:\n" + gates.round(3).to_string())
    print("\nmean attribution (percent):\n" + attr.round(4).to_string())
    log = RESULTS / "live" / "signals.csv"
    pd.DataFrame([out]).to_csv(log, mode="a", header=not log.exists(), index=False)


if __name__ == "__main__":
    main()
