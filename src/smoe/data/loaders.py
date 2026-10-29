"""Download and load daily OHLCV data for the research universe.

Every series is stored as data/raw/<name>.csv with columns
date, open, high, low, close, volume (volume may be NaN), sorted ascending.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smoe import DATA_RAW

START = "2005-01-01"


@dataclass(frozen=True)
class Asset:
    name: str
    source: str  # "ak_index" | "ak_future" | "ak_etf" | "yf"
    ticker: str
    group: str  # "china_equity" | "intl_equity" | "derivative" | "crypto" | "tradable"
    label: str


UNIVERSE: dict[str, Asset] = {a.name: a for a in [
    Asset("csi300", "ak_index", "sh000300", "china_equity", "CSI 300"),
    Asset("csi500", "ak_index", "sh000905", "china_equity", "CSI 500"),
    Asset("sse50", "ak_index", "sh000016", "china_equity", "SSE 50"),
    Asset("chinext", "ak_index", "sz399006", "china_equity", "ChiNext"),
    Asset("spx", "yf", "^GSPC", "intl_equity", "S&P 500"),
    Asset("ndx", "yf", "^NDX", "intl_equity", "Nasdaq 100"),
    Asset("hsi", "yf", "^HSI", "intl_equity", "Hang Seng"),
    Asset("n225", "yf", "^N225", "intl_equity", "Nikkei 225"),
    Asset("dax", "yf", "^GDAXI", "intl_equity", "DAX"),
    Asset("ftse", "yf", "^FTSE", "intl_equity", "FTSE 100"),
    Asset("if_main", "ak_future", "IF0", "derivative", "CSI 300 futures (IF)"),
    Asset("au_main", "ak_future", "AU0", "derivative", "SHFE gold futures"),
    Asset("cu_main", "ak_future", "CU0", "derivative", "SHFE copper futures"),
    Asset("m_main", "ak_future", "M0", "derivative", "DCE soybean meal futures"),
    Asset("cl_fut", "yf", "CL=F", "derivative", "WTI crude futures"),
    Asset("gc_fut", "yf", "GC=F", "derivative", "COMEX gold futures"),
    Asset("zn_fut", "yf", "ZN=F", "derivative", "US 10y T-note futures"),
    Asset("btc", "yf", "BTC-USD", "crypto", "Bitcoin"),
    Asset("etf510300", "ak_etf", "sh510300", "tradable", "CSI 300 ETF (510300)"),
    Asset("vix", "yf", "^VIX", "state", "CBOE VIX"),
]}


def _standardise(df: pd.DataFrame, date_col: str, cols: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]).dt.tz_localize(None)})
    for new, old in cols.items():
        out[new] = pd.to_numeric(df[old], errors="coerce") if old in df else np.nan
    out = out.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    out = out[out["date"] >= START].reset_index(drop=True)
    return out


def _download(asset: Asset) -> pd.DataFrame:
    import akshare as ak

    ohlcv = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    if asset.source == "ak_index":
        return _standardise(ak.stock_zh_index_daily(symbol=asset.ticker), "date", ohlcv)
    if asset.source == "ak_etf":
        return _standardise(ak.fund_etf_hist_sina(symbol=asset.ticker), "date", ohlcv)
    if asset.source == "ak_future":
        df = ak.futures_main_sina(symbol=asset.ticker)
        cols = {"open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价", "volume": "成交量"}
        return _standardise(df, "日期", cols)
    if asset.source == "yf":
        import yfinance as yf

        df = yf.download(asset.ticker, start=START, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        return _standardise(df, "Date", cols)
    raise ValueError(f"unknown source {asset.source}")


def download_all(names: list[str] | None = None) -> pd.DataFrame:
    """Download every asset to data/raw and write a manifest with row counts and hashes."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in names or list(UNIVERSE):
        asset = UNIVERSE[name]
        df = _download(asset)
        path = DATA_RAW / f"{name}.csv"
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        manifest.append({
            "name": name, "label": asset.label, "group": asset.group, "source": asset.source,
            "ticker": asset.ticker, "rows": len(df),
            "first": df["date"].iloc[0].strftime("%Y-%m-%d"), "last": df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
    man = pd.DataFrame(manifest)
    (DATA_RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return man


def load_daily(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / f"{name}.csv", parse_dates=["date"]).set_index("date")
    return df[df["close"] > 0]


def daily_returns(name: str) -> pd.Series:
    """Close-to-close log returns in percent."""
    close = load_daily(name)["close"]
    return (100 * np.log(close).diff()).dropna().rename(name)
