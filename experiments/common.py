"""Shared experiment configuration: model zoo, asset universe, periods, caching."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# torch and Anaconda's MKL each ship libiomp5md.dll; without this a worker that loads both aborts
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pandas as pd

from smoe import RESULTS, LOCKBOX_START
from smoe.data import UNIVERSE, load_daily
from smoe.features import build_features
from smoe.models import SMoE, single_gaussian, constant_gate, lstar
from smoe.models.benchmarks import GARCH, HistGauss, MarkovSwitching, XGBGauss
from smoe.walkforward import walk_forward, ForecastPanel

OOS_START = "2012-01-01"
DEV_END = (pd.Timestamp(LOCKBOX_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
FORECASTS = RESULTS / "forecasts"


def _deep():
    import torch

    torch.set_num_threads(1)
    from smoe.models.deep import DeepSMoE

    return DeepSMoE()


# name -> (factory, refit frequency)
MODELS = {
    "HistGauss": (HistGauss, "QS"),
    "GARCH-t": (lambda: GARCH(name="GARCH-t"), "QS"),
    "AR-GJR-t": (lambda: GARCH(ar=1, gjr=True, name="AR-GJR-t"), "QS"),
    "Gauss-HAR": (single_gaussian, "QS"),
    "MAR": (constant_gate, "QS"),
    "LSTAR": (lstar, "QS"),
    "MS-AR": (MarkovSwitching, "QS"),
    "XGB-Gauss": (XGBGauss, "QS"),
    "SMoE": (SMoE, "QS"),
    "Deep-SMoE": (_deep, "YS"),
}

ASSETS = [a for a, spec in UNIVERSE.items() if spec.group != "state"]


def features(asset: str):
    return build_features(load_daily(asset))


def panel_path(period: str, asset: str, model: str) -> Path:
    return FORECASTS / period / f"{asset}__{model}.npz"


def run_one(asset: str, model: str, period: str = "dev", force: bool = False) -> str:
    path = panel_path(period, asset, model)
    if path.exists() and not force:
        return f"cached {asset} {model}"
    path.parent.mkdir(parents=True, exist_ok=True)
    factory, freq = MODELS[model]
    # dev: 2012 -> DEV_END; lockbox: refit from scratch at LOCKBOX_START and continue to the last date
    start = LOCKBOX_START if period == "lockbox" else OOS_START
    end = DEV_END if period == "dev" else None
    fs = features(asset)
    panel = walk_forward(factory, fs, start=start, end=end, freq=freq, asset=asset)
    panel.to_npz(path)
    return f"done {asset} {model} n={len(panel.index)} {panel.fit_seconds:.0f}s"


def load_panels(period: str, asset: str, models=None) -> dict[str, ForecastPanel]:
    out = {}
    for m in models or MODELS:
        p = panel_path(period, asset, m)
        if p.exists():
            out[m] = ForecastPanel.from_npz(p)
    return out
