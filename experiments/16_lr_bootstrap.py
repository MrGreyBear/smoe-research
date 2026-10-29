"""Parametric bootstrap of the likelihood-ratio statistic for the number of experts.

Under H0 the extra expert's parameters are unidentified, so LR has no chi-square limit. We use the
fixed-design parametric bootstrap of McLachlan (1987): covariates stay at their observed values,
y* is drawn from the fitted null model, and both models are re-estimated on every draw.

  K = 1 (Gauss-HAR)  vs  K = 3 (SMoE)
  K = 3 (SMoE)       vs  K = 4 (SMoE-K4c: a second intercept-only expert)
on the CSI 300 and the S&P 500, development sample. Also: LM tests of candidate gate variables
computed from the baseline fit alone. Writes results/tables/kg_lr_bootstrap.csv, kg_lm_gate.csv.
usage: python experiments/16_lr_bootstrap.py [n_jobs]
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import ASSETS, DEV_END, features  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import load_daily  # noqa: E402
from smoe.features.extra import add_vix  # noqa: E402
from smoe.features.semantic import SEMANTIC_GROUPS as G  # noqa: E402
from smoe.models import SMoE, single_gaussian  # noqa: E402

VIX = load_daily("vix")["close"]
K4C = {"trend": G["trend"], "reversal": G["reversal"], "calm": [], "crisis": []}
TESTS = {"K1 vs K3": (single_gaussian, SMoE, 199), "K3 vs K4": (SMoE, lambda **kw: SMoE(experts=K4C, **kw), 99)}
CANDIDATES = {"VIX": ["g_vix"], "last return": ["rev1"], "1-day variance": ["lv1"],
              "VIX + return + 1d var": ["g_vix", "rev1", "lv1"]}


def data(asset):
    fs = add_vix(features(asset), VIX).slice(None, DEV_END)
    ok = fs.y.notna()
    return fs.X[ok], fs.y[ok]


def fit_alt(f_alt, X, y, start):
    """Best of a warm start at the observed-data estimate and two random starts."""
    a = f_alt(n_init=2).fit(X, y)
    b = f_alt(n_init=1).fit(X, y, warm_start=start)
    return a if a.loglik_ >= b.loglik_ else b


def observed(asset, test):
    f0, f1, _ = TESTS[test]
    X, y = data(asset)
    m0, m1 = f0().fit(X, y), fit_alt(f1, X, y, None)
    return asset, test, 2 * (m1.loglik_ - m0.loglik_), m0, m1


def one_draw(asset, test, b, m0, m1):
    f0, f1, _ = TESTS[test]
    X, _ = data(asset)
    ystar = pd.Series(m0.predict(X).sample(np.random.default_rng(1000 + b), 1)[:, 0], index=X.index)
    b0 = f0().fit(X, ystar, warm_start=m0)
    b1 = fit_alt(f1, X, ystar, m1)
    return asset, test, b, 2 * (b1.loglik_ - b0.loglik_)


def lm_screen(asset):
    X, y = data(asset)
    m = SMoE().fit(X, y)
    return [{"asset": asset, "candidate": k, **m.lm_gate_test(X, y, cols)} for k, cols in CANDIDATES.items()]


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    lm = Parallel(n_jobs=n_jobs)(delayed(lm_screen)(a) for a in ASSETS)
    pd.DataFrame([r for rows in lm for r in rows]).to_csv(TABLES / "kg_lm_gate.csv", index=False)
    assets = ["csi300", "spx"]
    fits = {(a, t): rest for a, t, *rest in Parallel(n_jobs=n_jobs)(delayed(observed)(a, t) for a in assets
                                                                       for t in TESTS)}
    obs = {k: v[0] for k, v in fits.items()}
    draws = Parallel(n_jobs=n_jobs)(delayed(one_draw)(a, t, b, *fits[(a, t)][1:]) for a in assets
                                    for t, (_, _, B) in TESTS.items() for b in range(B))
    D = pd.DataFrame(draws, columns=["asset", "test", "b", "LR"])
    D.to_csv(TABLES / "kg_lr_bootstrap_draws.csv", index=False)
    rows = []
    for (a, t), g in D.groupby(["asset", "test"]):
        lr = obs[(a, t)]
        rows.append({"asset": a, "test": t, "LR": lr, "B": len(g), "boot_p": (1 + (g["LR"] >= lr).sum()) / (1 + len(g)),
                     "boot_q95": g["LR"].quantile(0.95), "boot_mean": g["LR"].mean()})
    pd.DataFrame(rows).to_csv(TABLES / "kg_lr_bootstrap.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
