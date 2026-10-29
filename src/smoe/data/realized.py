"""Realised measures from CSI 300 5-minute bars (percent-squared units)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe import DATA_RAW, DATA_PROCESSED

MU1 = np.sqrt(2 / np.pi)  # E|Z| for Z ~ N(0,1)


def realized_measures(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily realised variance, bipower variation, jumps, semivariances, overnight return.

    `bars` has columns trade_time (datetime), open, close. Intraday returns are taken
    from consecutive 5-minute closes within a day; the overnight return is
    log(first open of day t / last close of day t-1).
    """
    # the vendor file repeats every bar of some days and has isolated zero-price bars
    b = bars.drop_duplicates("trade_time").sort_values("trade_time")
    b = b[(b["open"] > 0) & (b["close"] > 0)].copy()
    b["day"] = b["trade_time"].dt.normalize()
    b["r"] = 100 * np.log(b["close"]).groupby(b["day"]).diff()

    def per_day(g: pd.DataFrame) -> pd.Series:
        r = g["r"].dropna().to_numpy()
        absr = np.abs(r)
        rv = np.sum(r**2)
        bv = MU1**-2 * np.sum(absr[1:] * absr[:-1]) * len(r) / max(len(r) - 1, 1)
        return pd.Series({
            "rv": rv,
            "bv": bv,
            "jump": max(rv - bv, 0.0),
            "rs_pos": np.sum(r[r > 0] ** 2),
            "rs_neg": np.sum(r[r < 0] ** 2),
            "n_bars": len(r),
            "open": g["open"].iloc[0],
            "close": g["close"].iloc[-1],
        })

    out = b.groupby("day").apply(per_day)
    out["ret_overnight"] = 100 * np.log(out["open"] / out["close"].shift(1))
    out.index.name = "date"
    return out.drop(columns=["open", "close"])


def build_csi300_realized() -> pd.DataFrame:
    bars = pd.read_csv(DATA_RAW / "000300_SH_5min.csv", parse_dates=["trade_time"])
    rm = realized_measures(bars)
    rm = rm[rm["n_bars"] >= 40]  # drop half days / broken sessions
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rm.to_csv(DATA_PROCESSED / "csi300_realized.csv")
    return rm


def load_csi300_realized() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "csi300_realized.csv", parse_dates=["date"], index_col="date")
