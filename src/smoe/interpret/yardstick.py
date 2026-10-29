"""The white-box yardstick: project any density forecaster onto the SMoE family.

Distillation (information projection). For a teacher with predictive densities p_T(.|x),
the student solves
    theta* = argmin_theta E_x KL(p_T(.|x) || p_theta(.|x)) = argmax_theta E_x E_{y~p_T} log p_theta(y|x),
estimated by maximum likelihood on M teacher draws per training row (EM unchanged).

Fidelity. With the true conditional density p0, expected log score satisfies
    E log p(y|x) = -H(p0) - E KL(p0 || p),
so the out-of-sample log-score gain of the teacher over a reference model R splits into a
part reproduced by the white box and a residual:
    Phi = (LS_S - LS_R) / (LS_T - LS_R).
Phi = 1: the black box's whole advantage is expressible by the semantic mixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe.distributions import MixtureForecast
from smoe.models import SMoE


def distill(teacher_fc: MixtureForecast, X: pd.DataFrame, M: int = 10, seed: int = 0,
            student: SMoE | None = None, warm_start: SMoE | None = None) -> SMoE:
    """Fit an SMoE student to samples from the teacher's predictive distributions on rows X."""
    rng = np.random.default_rng(seed)
    draws = teacher_fc.sample(rng, M)  # (n, M)
    Xr = X.loc[X.index.repeat(M)]
    Xr = Xr.reset_index(drop=True)
    yr = pd.Series(draws.ravel(), name="y")
    st = student or SMoE(n_init=2)
    return st.fit(Xr, yr, warm_start=warm_start)


def expected_kl(p: MixtureForecast, q: MixtureForecast, n_draws: int = 200, seed: int = 0) -> np.ndarray:
    """Monte Carlo estimate of KL(p || q) for each row."""
    rng = np.random.default_rng(seed)
    s = p.sample(rng, n_draws)
    T = len(p)
    kl = np.empty(T)
    for t in range(T):
        pt, qt = p.subset(slice(t, t + 1)), q.subset(slice(t, t + 1))
        ys = s[t]
        lp = pt.logpdf(ys) if False else _row_logpdf(pt, ys)
        lq = _row_logpdf(qt, ys)
        kl[t] = np.mean(lp - lq)
    return kl


def _row_logpdf(f: MixtureForecast, ys: np.ndarray) -> np.ndarray:
    rep = MixtureForecast(np.repeat(f.w, len(ys), 0), np.repeat(f.mu, len(ys), 0), np.repeat(f.sd, len(ys), 0))
    return rep.logpdf(ys)


def fidelity(y, teacher: MixtureForecast, student: MixtureForecast, reference: MixtureForecast) -> dict:
    y = np.asarray(y, float)
    ls_t, ls_s, ls_r = (np.mean(f.logpdf(y)) for f in (teacher, student, reference))
    gain_t = ls_t - ls_r
    return {
        "LS_teacher": ls_t, "LS_student": ls_s, "LS_reference": ls_r,
        "Phi": (ls_s - ls_r) / gain_t if abs(gain_t) > 1e-12 else np.nan,
        "R2_mean": np.corrcoef(teacher.mean(), student.mean())[0, 1] ** 2,
        "R2_logsd": np.corrcoef(np.log(teacher.var()), np.log(student.var()))[0, 1] ** 2,
        "R2_probup": np.corrcoef(teacher.prob_up(), student.prob_up())[0, 1] ** 2,
    }


def probe(gates: pd.DataFrame, Z: pd.DataFrame) -> pd.Series:
    """R^2 of linear probes: each black-box gate logit regressed on observable state variables."""
    out = {}
    Zc = np.c_[np.ones(len(Z)), ((Z - Z.mean()) / Z.std()).to_numpy()]
    P = np.clip(gates.to_numpy(), 1e-6, 1)
    logit = np.log(P) - np.log(P).mean(1, keepdims=True)
    for j, c in enumerate(gates.columns):
        b = np.linalg.lstsq(Zc, logit[:, j], rcond=None)[0]
        e = logit[:, j] - Zc @ b
        out[c] = 1 - e.var() / logit[:, j].var()
    return pd.Series(out)
