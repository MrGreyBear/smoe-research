"""Release-date test for pretraining overlap in zero-shot foundation models.

A foundation model released on date C cannot have been trained on returns after C. If its
pretraining corpus contained the evaluation series before C, its loss relative to a real-time
model should rise after C. With d_t = NLL_FM - NLL_SMoE averaged across series on date t,
    d_t = a + b 1{t >= C} + e_t,
estimated with Newey-West errors; b > 0 is the signature of memorisation. A placebo replaces the
foundation model by GARCH-t, which is estimated in real time, and the difference-in-differences
b_FM - b_GARCH removes market-wide shifts in relative performance at C.
Writes results/tables/fm_release_test.csv
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import ASSETS, load_panels  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.evaluation.tests import nw_lrv  # noqa: E402

RELEASE = {"Chronos-Bolt-small": "2024-11-26", "Chronos-Bolt-base": "2024-11-26", "TimesFM-2.5": "2025-09-15"}
REF, PLACEBO = "SMoE", "GARCH-t"


def nll_series(asset, model):
    """Out-of-sample NLL of `model`, development and lockbox panels concatenated."""
    out = []
    for period in ("dev", "lockbox"):
        P = load_panels(period, asset, [model])
        if model in P:
            p = P[model]
            out.append(pd.Series(-p.fc.logpdf(p.y), index=p.index))
    s = pd.concat(out)
    return s[~s.index.duplicated()]


def ols_hac(d, post):
    X = np.c_[np.ones(len(d)), post]
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    u = d - X @ coef
    # HAC variance of the post-period mean difference via the moment u * (post - mean(post))
    g = u * (post - post.mean())
    se = np.sqrt(nw_lrv(g) / len(d)) / post.var()
    return coef[1], se


def relative(model):
    rows = {}
    for a in ASSETS:
        ref = nll_series(a, REF)
        m = nll_series(a, model)
        idx = ref.index.intersection(m.index)
        rows[a] = (m[idx] - ref[idx])
    return pd.DataFrame(rows).sort_index()


if __name__ == "__main__":
    placebo = relative(PLACEBO)
    out = []
    for fm, c in RELEASE.items():
        D = relative(fm)
        for name, frame in [(fm, D), (f"placebo {PLACEBO}", placebo), ("difference", D - placebo)]:
            d = frame.mean(axis=1).dropna()
            post = (d.index >= pd.Timestamp(c)).astype(float)
            b, se = ols_hac(d.to_numpy(), post)
            out.append({"model": fm, "series": name, "release": c, "n_pre": int((post == 0).sum()),
                        "n_post": int(post.sum()), "mean_pre": d[post == 0].mean(), "mean_post": d[post == 1].mean(),
                        "b": b, "se": se, "z": b / se, "p_one_sided": stats.norm.sf(b / se)})
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "fm_release_test.csv", index=False)
    print(df.round(4).to_string())
