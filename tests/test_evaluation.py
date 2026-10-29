import numpy as np
import pandas as pd
from scipy import stats

from smoe.evaluation import tests as T
from smoe.evaluation.losses import qlike, fz0

rng = np.random.default_rng(7)


def rejection_rate(fn, reps=400):
    return np.mean([fn(np.random.default_rng(i)) < 0.05 for i in range(reps)])


def test_dm_size_close_to_nominal():
    rate = rejection_rate(lambda g: T.dm_test(g.normal(size=300), g.normal(size=300))["p_two"])
    assert 0.02 < rate < 0.09


def test_dm_detects_better_model():
    a = rng.normal(1.0, 1, 500)
    b = rng.normal(0.7, 1, 500)
    assert T.dm_test(a, b)["p_b_better"] < 0.01


def test_pesaran_timmermann_size_and_power():
    rate = rejection_rate(lambda g: T.pesaran_timmermann(g.normal(size=400), g.normal(size=400))["p"])
    assert rate < 0.09
    y = rng.normal(size=2000)
    yhat = y + rng.normal(0, 1.0, 2000)
    assert T.pesaran_timmermann(y, yhat)["p"] < 1e-6


def test_kupiec_and_christoffersen_under_correct_coverage():
    rate = rejection_rate(lambda g: T.kupiec(g.random(1000) < 0.05, 0.05)["p"])
    assert rate < 0.09
    rate = rejection_rate(lambda g: T.christoffersen(g.random(1000) < 0.05, 0.05)["p_ind"])
    assert rate < 0.09


def test_berkowitz_size_and_power():
    rate = rejection_rate(lambda g: T.berkowitz(g.random(500))["p"])
    assert rate < 0.09
    bad = stats.norm.cdf(rng.normal(0, 1.6, 1000))  # overconfident forecasts
    assert T.berkowitz(bad)["p"] < 1e-6


def test_clark_west_null_and_alternative():
    rate = rejection_rate(lambda g: T.clark_west(g.normal(size=500), np.zeros(500),
                                                 0.1 * g.normal(size=500))["p"])
    assert rate < 0.09
    x = rng.normal(size=3000)
    y = 0.2 * x + rng.normal(size=3000)
    assert T.clark_west(y, np.zeros(3000), 0.2 * x)["p"] < 1e-4


def test_mcs_keeps_best_and_drops_clearly_worse():
    n = 800
    base = rng.normal(0, 1, n)
    L = pd.DataFrame({"best": base + 1.0, "tie": base + 1.0 + rng.normal(0, 0.3, n),
                      "bad": base + 1.6 + rng.normal(0, 0.3, n)})
    out = T.model_confidence_set(L, B=500)
    assert out.loc["best", "in_mcs"] and not out.loc["bad", "in_mcs"]


def test_losses_minimised_at_truth():
    assert np.isclose(qlike(2.0, 2.0), 0.0)
    # FZ0 expected loss smaller at the true (VaR, ES) of N(0,1) than at a shifted pair
    y = rng.normal(size=200_000)
    a = 0.05
    q = stats.norm.ppf(a)
    es = -stats.norm.pdf(q) / a
    assert fz0(y, q, es, a).mean() < fz0(y, q * 0.8, es * 0.8, a).mean()


def test_murphy_decomposition_identity_for_binned_forecasts():
    p = rng.choice([0.2, 0.5, 0.8], size=5000)
    o = rng.random(5000) < p
    d = T.murphy_brier(p, o, bins=3)
    assert np.isclose(d["brier"], d["reliability"] - d["resolution"] + d["uncertainty"], atol=1e-10)
