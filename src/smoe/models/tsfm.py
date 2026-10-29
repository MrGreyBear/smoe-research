"""Zero-shot time-series foundation models (Chronos-Bolt, TimesFM 2.5) as density forecasters.

Both models return quantiles at levels 0.1, ..., 0.9. We project them onto a
Gaussian by least squares on the quantile function, q_i = mu + sigma * Phi^{-1}(tau_i),
which has the closed form of an OLS regression of q on (1, z). The projection is
what enters log-score / CRPS comparisons; tails beyond the 10%/90% levels are
therefore extrapolated (stated as a limitation in the paper).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from smoe.distributions import MixtureForecast

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

LEVELS = np.arange(1, 10) / 10.0


def gaussian_projection(q: np.ndarray) -> MixtureForecast:
    """Least-squares Gaussian fit to quantiles q (T, 9) at LEVELS."""
    z = norm.ppf(LEVELS)
    Z = np.c_[np.ones_like(z), z]
    coef = np.linalg.lstsq(Z, q.T, rcond=None)[0]  # (2, T)
    return MixtureForecast.gaussian(coef[0], np.maximum(coef[1], 1e-3))


@dataclass
class ChronosBolt:
    size: str = "small"
    context: int = 512
    batch: int = 256
    name: str = "Chronos-Bolt"

    def __post_init__(self):
        self.name = f"Chronos-Bolt-{self.size}"

    def _pipe(self):
        if not hasattr(self, "_p"):
            import torch
            from chronos import BaseChronosPipeline

            torch.set_num_threads(4)
            self._p = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{self.size}", device_map="cpu",
                                                          torch_dtype=torch.float32)
        return self._p

    def fit(self, X, y):
        return self  # zero-shot

    def predict_quantiles(self, X, rows=None) -> np.ndarray:
        """Quantiles of r_{t+1} for each row position t in `rows` (default: all rows), context r_{<=t}."""
        import torch

        r = X["r"].to_numpy(np.float32)
        rows = np.arange(len(r)) if rows is None else np.asarray(rows)
        pipe = self._pipe()
        out = np.empty((len(rows), len(LEVELS)))
        for i0 in range(0, len(rows), self.batch):
            ts = rows[i0:i0 + self.batch]
            ctx = [torch.from_numpy(r[max(0, t + 1 - self.context):t + 1].copy()) for t in ts]
            q, _ = pipe.predict_quantiles(inputs=ctx, prediction_length=1, quantile_levels=list(LEVELS))
            out[i0:i0 + len(ts)] = q[:, 0, :].numpy()
        return out

    def predict(self, X, rows=None) -> MixtureForecast:
        return gaussian_projection(self.predict_quantiles(X, rows))


@dataclass
class TimesFM:
    """TimesFM 2.5 (200M, decoder-only, patch length 32) with its continuous quantile head."""

    context: int = 512
    batch: int = 256
    repo: str = "google/timesfm-2.5-200m-pytorch"
    name: str = "TimesFM-2.5"

    def _model(self):
        if not hasattr(self, "_m"):
            import torch
            import timesfm

            torch.set_num_threads(4)
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo)
            m.compile(timesfm.ForecastConfig(max_context=self.context, max_horizon=128, normalize_inputs=True,
                                             per_core_batch_size=self.batch, use_continuous_quantile_head=True,
                                             force_flip_invariance=True, infer_is_positive=False,
                                             fix_quantile_crossing=True))
            self._m = m
        return self._m

    def fit(self, X, y):
        return self  # zero-shot

    def predict_quantiles(self, X, rows=None) -> np.ndarray:
        """Quantiles of r_{t+1} for each row position t in `rows`, context r_{<=t}."""
        r = X["r"].to_numpy(np.float64)
        rows = np.arange(len(r)) if rows is None else np.asarray(rows)
        m = self._model()
        out = np.empty((len(rows), len(LEVELS)))
        for i0 in range(0, len(rows), self.batch):
            ts = rows[i0:i0 + self.batch]
            ctx = [r[max(0, t + 1 - self.context):t + 1].copy() for t in ts]
            _, q = m.forecast(horizon=1, inputs=ctx)  # q: (n, 1, 10) = mean, then deciles 0.1..0.9
            out[i0:i0 + len(ts)] = q[:, 0, 1:]
        return out

    def predict(self, X, rows=None) -> MixtureForecast:
        return gaussian_projection(self.predict_quantiles(X, rows))
