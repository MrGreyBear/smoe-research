import numpy as np
from scipy.stats import norm

from smoe.distributions import MixtureForecast

rng = np.random.default_rng(0)


def random_mixture(T=5, K=3):
    w = rng.dirichlet(np.ones(K), size=T)
    mu = rng.normal(0, 1, (T, K))
    sd = rng.uniform(0.5, 2.0, (T, K))
    return MixtureForecast(w, mu, sd)


def test_variance_decomposition_matches_samples():
    f = random_mixture()
    s = f.sample(rng, 400_000)
    np.testing.assert_allclose(f.mean(), s.mean(axis=1), atol=0.02)
    np.testing.assert_allclose(f.var(), s.var(axis=1), rtol=0.02)
    np.testing.assert_allclose(f.var(), f.within_var() + f.between_var())


def test_quantile_inverts_cdf():
    f = random_mixture()
    for a in (0.01, 0.05, 0.5, 0.95):
        np.testing.assert_allclose(f.cdf(f.quantile(a)), a, atol=1e-8)


def test_es_matches_monte_carlo():
    f = random_mixture()
    s = f.sample(rng, 400_000)
    q = f.quantile(0.05)
    mc = np.array([row[row <= qi].mean() for row, qi in zip(s, q)])
    np.testing.assert_allclose(f.es(0.05), mc, rtol=0.03)


def test_crps_matches_energy_form():
    f = random_mixture(T=3)
    y = rng.normal(0, 1, 3)
    x1, x2 = f.sample(rng, 400_000), f.sample(rng, 400_000)
    mc = np.abs(x1 - y[:, None]).mean(1) - 0.5 * np.abs(x1 - x2).mean(1)
    np.testing.assert_allclose(f.crps(y), mc, atol=0.01)


def test_single_gaussian_crps_closed_form():
    f = MixtureForecast.gaussian([0.3], [1.7])
    y = 1.1
    z = (y - 0.3) / 1.7
    exact = 1.7 * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(f.crps([y]), exact, rtol=1e-10)


def test_logpdf_and_prob_up():
    f = random_mixture(T=1)
    y = 0.4
    dens = np.sum(f.w * norm.pdf(y, f.mu, f.sd))
    np.testing.assert_allclose(f.logpdf([y]), np.log(dens))
    np.testing.assert_allclose(f.prob_up(), np.sum(f.w * norm.cdf(f.mu / f.sd)))
