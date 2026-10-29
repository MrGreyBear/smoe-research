import numpy as np
import pandas as pd
import pytest

from smoe.distributions import MixtureForecast
from smoe.features import build_features
from smoe.models import SMoE
from smoe.models.benchmarks import GARCH, MarkovSwitching, HistGauss, XGBGauss, student_t_mixture
from smoe.walkforward import walk_forward
from tests.test_leakage import synthetic_ohlc


class Spy:
    """Fails if fit sees a target realised on/after the first date it will forecast."""
    name = "spy"
    log = []

    def fit(self, X, y):
        self.last_train = X.index[-1]
        return self

    def predict(self, X):
        Spy.log.append((self.last_train, X.index[-1]))
        assert "y" not in X.columns
        return MixtureForecast.gaussian(X["r"].to_numpy() * 0, np.ones(len(X)))


def test_runner_trains_strictly_before_test_block():
    fs = build_features(synthetic_ohlc(1400))
    Spy.log.clear()
    panel = walk_forward(Spy, fs, start=fs.X.index[900], freq="QS", min_train=300)
    for (train_end, _), d in zip(Spy.log, panel.index[:1]):
        assert train_end < d
    assert len(panel.index) == len(np.unique(panel.index))


@pytest.mark.parametrize("factory", [lambda: SMoE(n_init=1, max_iter=60), lambda: GARCH(ar=1, gjr=True),
                                     lambda: MarkovSwitching(), HistGauss, lambda: XGBGauss(n_estimators=50),
                                     lambda: __import__("smoe.models.deep", fromlist=["DeepSMoE"]).DeepSMoE(epochs=3, seeds=(0,))])
def test_forecast_at_t_invariant_to_future_corruption(factory):
    df = synthetic_ohlc(1300, seed=4)
    fs = build_features(df)
    start = fs.X.index[1000]
    cut = fs.X.index[1100]
    base = walk_forward(factory, fs, start=start, end=cut, freq="YS", min_train=300)
    corrupted = df.copy()
    corrupted.loc[corrupted.index > cut, ["open", "high", "low", "close"]] *= 1.3
    alt = walk_forward(factory, build_features(corrupted), start=start, end=cut, freq="YS", min_train=300)
    np.testing.assert_allclose(base.fc.mean(), alt.fc.mean(), rtol=1e-8)
    np.testing.assert_allclose(base.fc.var(), alt.fc.var(), rtol=1e-8)


def test_student_t_mixture_matches_t_density():
    from scipy.stats import t as tdist
    f = student_t_mixture(np.array([0.1]), np.array([1.3]), nu=5.0, n_nodes=32)
    ys = np.array([-4.0, -1.0, 0.0, 2.5])
    for y in ys:
        np.testing.assert_allclose(np.exp(f.logpdf([y])), tdist.pdf(y, 5.0, loc=0.1, scale=1.3), rtol=2e-3)
