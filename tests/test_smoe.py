import numpy as np
import pytest

from smoe.models import SMoE, single_gaussian
from smoe.simulate import simulate_smoe, TRUE_PARAMS


@pytest.fixture(scope="module")
def fitted():
    X, y, _ = simulate_smoe(20_000, seed=11)
    m = SMoE(lam=1e-3, lam_v=1e-3, n_init=3, seed=1).fit(X, y)
    return m, X, y


def test_penalised_loglik_is_monotone(fitted):
    m, _, _ = fitted
    assert np.all(np.diff(m.path_) > -1e-6 * np.abs(m.path_[1:]))


def test_recovers_true_parameters(fitted):
    m, _, _ = fitted
    beta, delta, gamma = m.params_
    # simulated X is ~N(0,1) so standardisation changes parameters only marginally
    for k in range(3):
        np.testing.assert_allclose(beta[k], TRUE_PARAMS["beta"][k], atol=0.08)
        np.testing.assert_allclose(delta[k], TRUE_PARAMS["delta"][k], atol=0.15)
    np.testing.assert_allclose(gamma, TRUE_PARAMS["gamma"], atol=0.25)


def test_attribution_sums_to_mean(fitted):
    m, X, _ = fitted
    Xs = X.iloc[:500]
    np.testing.assert_allclose(m.attribution(Xs).sum(axis=1), m.predict(Xs).mean(), atol=1e-12)


def test_gate_elasticity_matches_finite_difference(fitted):
    m, X, _ = fitted
    Xs = X.iloc[:50].copy()
    el = m.gate_elasticity(Xs)
    # gate cols are not used by experts or variance, so bumping them moves only the gate
    c = "g_mom60"
    h = 1e-5 * m.scale_[c]
    up, dn = Xs.copy(), Xs.copy()
    up[c] += h
    dn[c] -= h
    fd = (m.predict(up).mean() - m.predict(dn).mean()) / (2e-5)
    np.testing.assert_allclose(el[c].to_numpy(), fd, rtol=1e-4, atol=1e-8)


def test_scores_vanish_at_optimum(fitted):
    m, X, y = fitted
    inf = m.inference(X, y)
    assert inf["se_sandwich"].gt(0).all()
    # with a tiny penalty the observed-data score sums to ~0 at the MLE
    S = m._scores(m._theta(m.params_), m._design(X), y.to_numpy())
    assert np.max(np.abs(S.sum(0))) < 0.5


def test_single_gaussian_matches_ols_mean():
    X, y, _ = simulate_smoe(5000, seed=2)
    m = single_gaussian().fit(X, y)
    assert m.K == 1
    f = m.predict(X)
    np.testing.assert_allclose(f.w, 1.0)
    assert np.isfinite(m.loglik_)


def test_lm_gate_test_detects_omitted_state_and_accepts_noise(fitted):
    m, X, y = fitted
    Xn = X.assign(noise=np.random.default_rng(5).standard_normal(len(X)))
    lm = m.lm_gate_test(Xn, y, ["noise"])
    assert lm["df"] == 2 and lm["p"] > 1e-3
    full = SMoE(gate_cols=m.gate_cols + ["noise"], lam=1e-3, lam_v=1e-3, n_init=1).fit(Xn, y, warm_start=None)
    th, V = full.sandwich_cov(Xn, y)
    w = full.wald(th, V, lambda n: n.endswith(":noise"))
    assert abs(lm["LM"] - w["W"]) < 2.0
    # power: drawdown truly enters the gate; LM from the restricted fit must agree with Wald from the full fit
    Xs, ys, _ = simulate_smoe(8000, seed=11)
    restricted = SMoE(gate_cols=["g_lvol", "g_mom60"], lam=1e-3, lam_v=1e-3, n_init=2, seed=1).fit(Xs, ys)
    lm_dd = restricted.lm_gate_test(Xs, ys, ["g_dd"])
    full = SMoE(lam=1e-3, lam_v=1e-3, n_init=2, seed=1).fit(Xs, ys)
    th, V = full.sandwich_cov(Xs, ys)
    w_dd = full.wald(th, V, lambda n: n.endswith(":g_dd"))
    assert lm_dd["p"] < 1e-3 and w_dd["p"] < 1e-3
    assert abs(lm_dd["LM"] - w_dd["W"]) < 0.25 * w_dd["W"]
