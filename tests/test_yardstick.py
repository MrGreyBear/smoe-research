import numpy as np
import pandas as pd

from smoe.conformal import aci_quantiles
from smoe.distributions import MixtureForecast
from smoe.interpret.yardstick import distill, expected_kl, fidelity
from smoe.models import SMoE
from smoe.simulate import simulate_smoe


def test_aci_coverage_bound_holds_under_misspecification():
    rng = np.random.default_rng(0)
    T = 3000
    y = rng.standard_t(3, T) * np.r_[np.ones(1500), 3 * np.ones(1500)]  # variance break
    fc = MixtureForecast.gaussian(np.zeros(T), np.ones(T))  # badly misspecified
    out = aci_quantiles(fc, y, a=0.05, eta=0.01)
    assert abs(out["rate"] - 0.05) <= out["bound"] + 1e-12


def test_kl_zero_for_identical_and_positive_otherwise():
    p = MixtureForecast(np.array([[0.3, 0.7]]), np.array([[-1.0, 0.5]]), np.array([[1.0, 0.6]]))
    assert abs(expected_kl(p, p, 2000)[0]) < 1e-12
    q = MixtureForecast.gaussian([0.0], [1.0])
    assert expected_kl(p, q, 4000)[0] > 0.01


def test_distilling_an_smoe_teacher_recovers_it():
    X, y, _ = simulate_smoe(4000, seed=3)
    teacher = SMoE(n_init=2, seed=0).fit(X, y)
    fc_t = teacher.predict(X)
    student = distill(fc_t, X, M=10)
    fid = fidelity(y, fc_t, student.predict(X), MixtureForecast.gaussian(np.full(len(y), y.mean()),
                                                                          np.full(len(y), y.std())))
    assert fid["Phi"] > 0.95 and fid["R2_mean"] > 0.95
