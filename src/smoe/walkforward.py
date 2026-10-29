"""Expanding-window walk-forward evaluation.

At each refit date d the model is fitted on rows with index < d (their targets are
realised no later than the first test row's date, i.e. before any test forecast is
made) and then forecasts rows in [d, next refit date). `predict` receives the
features (incl. r_t) of all rows up to the end of the block but never the targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from smoe.distributions import MixtureForecast
from smoe.features.semantic import FeatureSet


@dataclass
class ForecastPanel:
    index: pd.DatetimeIndex
    y: np.ndarray
    fc: MixtureForecast
    model: str
    asset: str = ""
    fit_seconds: float = 0.0

    def to_npz(self, path):
        np.savez_compressed(path, index=self.index.values.astype("datetime64[D]").astype(str), y=self.y,
                            w=self.fc.w, mu=self.fc.mu, sd=self.fc.sd, model=self.model, asset=self.asset,
                            fit_seconds=self.fit_seconds)

    @classmethod
    def from_npz(cls, path) -> "ForecastPanel":
        z = np.load(path, allow_pickle=False)
        return cls(pd.DatetimeIndex(z["index"]), z["y"], MixtureForecast(z["w"], z["mu"], z["sd"]),
                   str(z["model"]), str(z["asset"]), float(z["fit_seconds"]))

    def loc(self, start=None, end=None) -> "ForecastPanel":
        m = np.ones(len(self.index), bool)
        if start is not None:
            m &= self.index >= pd.Timestamp(start)
        if end is not None:
            m &= self.index <= pd.Timestamp(end)
        return ForecastPanel(self.index[m], self.y[m], self.fc.subset(m), self.model, self.asset, self.fit_seconds)


def refit_dates(index: pd.DatetimeIndex, start, end=None, freq: str = "QS") -> list:
    end = end or index[-1]
    cands = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
    if len(cands) == 0 or cands[0] > pd.Timestamp(start):
        cands = pd.DatetimeIndex([pd.Timestamp(start)]).append(cands)
    return [d for d in cands if d <= index[-1]]


def _pad(fcs: list[MixtureForecast]) -> MixtureForecast:
    K = max(f.w.shape[1] for f in fcs)

    def pad(a, fill):
        return np.hstack([a, np.full((a.shape[0], K - a.shape[1]), fill)]) if a.shape[1] < K else a

    return MixtureForecast(np.vstack([pad(f.w, 0.0) for f in fcs]), np.vstack([pad(f.mu, 0.0) for f in fcs]),
                           np.vstack([pad(f.sd, 1.0) for f in fcs]))


def walk_forward(factory: Callable[[], object], fs: FeatureSet, start, end=None, freq: str = "QS",
                 min_train: int = 750, warm_start: bool = True, asset: str = "", verbose: bool = False
                 ) -> ForecastPanel:
    X = fs.X.assign(r=fs.r)
    y = fs.y
    idx = X.index
    end_ts = pd.Timestamp(end) if end is not None else idx[-1]
    dates = refit_dates(idx, start, end_ts, freq)
    blocks, prev, t0 = [], None, time.time()
    for j, d in enumerate(dates):
        nxt = dates[j + 1] if j + 1 < len(dates) else end_ts + pd.Timedelta(days=1)
        train = idx < d
        test = (idx >= d) & (idx < nxt) & (idx <= end_ts) & y.notna().to_numpy()
        if train.sum() < min_train or test.sum() == 0:
            continue
        model = factory()
        kw = {"warm_start": prev} if (warm_start and prev is not None and "warm_start" in model.fit.__code__.co_varnames) else {}
        model.fit(X[train], y[train], **kw)
        prev = model
        hist_end = np.where(test)[0][-1] + 1
        fc_all = model.predict(X.iloc[:hist_end])
        sel = np.where(test[:hist_end])[0]
        blocks.append((idx[sel], y.to_numpy()[sel], fc_all.subset(sel)))
        if verbose:
            print(f"  {asset} {getattr(model, 'name', '')} refit {d.date()} n_train={train.sum()} n_test={len(sel)}")
    index = pd.DatetimeIndex(np.concatenate([b[0] for b in blocks]))
    return ForecastPanel(index, np.concatenate([b[1] for b in blocks]), _pad([b[2] for b in blocks]),
                         getattr(factory(), "name", "model"), asset, time.time() - t0)
