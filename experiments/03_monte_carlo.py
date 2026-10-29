"""Monte Carlo study of the SMoE estimator.

Design (R replications per sample size, DGP = smoe.simulate.TRUE_PARAMS):
  1. bias and RMSE of every parameter for n in {1000, 2500, 5000};
  2. empirical coverage of 95% Wald intervals from sandwich and model-based standard errors;
  3. size and power of the robust Wald test of gate relevance H0: gamma slopes = 0
     (size: DGP with constant gate; power: the baseline DGP).
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401  (thread env vars)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

from smoe import TABLES  # noqa: E402
from smoe.models import SMoE  # noqa: E402
from smoe.simulate import TRUE_PARAMS, simulate_smoe  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SIZES = (1000, 2500, 5000)
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 14


def true_theta(params):
    return np.concatenate([*params["beta"], *params["delta"], params["gamma"].ravel()])


def one_rep(n, rep, constant_gate):
    params = dict(TRUE_PARAMS)
    if constant_gate:
        g = TRUE_PARAMS["gamma"].copy()
        g[:, 1:] = 0.0
        params = {**TRUE_PARAMS, "gamma": g}
    X, y, _ = simulate_smoe(n, seed=10_000 * int(constant_gate) + 100 * rep + n % 97, params=params)
    m = SMoE(lam=1e-3, lam_v=1e-3, n_init=3, seed=rep).fit(X, y)
    inf = m.inference(X, y)
    est, se_s, se_m = inf["estimate"].to_numpy(), inf["se_sandwich"].to_numpy(), inf["se_mle"].to_numpy()
    # robust Wald test of all gate slopes
    th = m._theta(m.params_)
    D = m._design(X)
    S = m._scores(th, D, y.to_numpy())
    P = len(th)
    A = np.zeros((P, P))
    h = 1e-5
    for j in range(P):
        e = np.zeros(P)
        e[j] = h
        A[:, j] = (m._scores(th + e, D, y.to_numpy()).sum(0) - m._scores(th - e, D, y.to_numpy()).sum(0)) / (2 * h)
    Ainv = np.linalg.pinv(0.5 * (A + A.T))
    V = Ainv @ (S.T @ S) @ Ainv
    idx = [i for i, nm in enumerate(m.param_names()) if nm.startswith("gamma") and not nm.endswith(":const")]
    g = th[idx]
    W = g @ np.linalg.solve(V[np.ix_(idx, idx)], g)
    return {"n": n, "rep": rep, "constant_gate": constant_gate, "est": est, "se_s": se_s, "se_m": se_m,
            "wald_p": stats.chi2.sf(W, len(idx)), "names": m.param_names()}


if __name__ == "__main__":
    jobs = [(n, r, cg) for n in SIZES for r in range(R) for cg in (False, True)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one_rep)(*j) for j in jobs)
    names = res[0]["names"]
    rows = []
    for n in SIZES:
        sub = [r for r in res if r["n"] == n and not r["constant_gate"]]
        th0 = true_theta(TRUE_PARAMS)
        E = np.array([r["est"] for r in sub])
        Ss = np.array([r["se_s"] for r in sub])
        Sm = np.array([r["se_m"] for r in sub])
        cov_s = np.mean(np.abs(E - th0) <= 1.96 * Ss, axis=0)
        cov_m = np.mean(np.abs(E - th0) <= 1.96 * Sm, axis=0)
        for j, nm in enumerate(names):
            rows.append({"n": n, "param": nm, "true": th0[j], "bias": E[:, j].mean() - th0[j],
                         "rmse": np.sqrt(np.mean((E[:, j] - th0[j]) ** 2)), "sd_est": E[:, j].std(),
                         "mean_se_sandwich": Ss[:, j].mean(), "mean_se_mle": Sm[:, j].mean(),
                         "cover_sandwich": cov_s[j], "cover_mle": cov_m[j]})
    pr = pd.DataFrame(rows)
    pr.to_csv(TABLES / "mc_parameters.csv", index=False)
    tests = []
    for n in SIZES:
        for cg in (True, False):
            ps = np.array([r["wald_p"] for r in res if r["n"] == n and r["constant_gate"] == cg])
            tests.append({"n": n, "dgp": "constant gate (size)" if cg else "state-dependent gate (power)",
                          "reject_5pct": np.mean(ps < 0.05), "reject_10pct": np.mean(ps < 0.10), "R": len(ps)})
    tt = pd.DataFrame(tests)
    tt.to_csv(TABLES / "mc_tests.csv", index=False)
    summ = pr.groupby("n").agg(abs_bias=("bias", lambda b: np.mean(np.abs(b))), rmse=("rmse", "mean"),
                               cover_sandwich=("cover_sandwich", "mean"), cover_mle=("cover_mle", "mean"))
    summ.to_csv(TABLES / "mc_summary.csv")
    print(summ.round(3).to_string())
    print(tt.to_string(index=False))
