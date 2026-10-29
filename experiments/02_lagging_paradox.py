"""The lagging paradox, quantified.

For each asset over the development OOS window:
  * RW   : random-walk price forecast  P_hat_{t+1} = P_t (the MSE-optimal forecast when E r = 0)
  * AR60 : OLS on the last 60 price levels, refitted yearly (a linear stand-in for an MSE-trained
           price-level network); we report sum of coefficients and weight on lag 1.
Metrics
  * RMSE on price levels and the ratio AR60 / RW,
  * the undergraduate "diff-DA": mean 1{sign(diff P_hat) = sign(diff P)},
  * its theoretical value for RW: P(sign r_t = sign r_{t+1}) (sign persistence),
  * proper DA: mean 1{sign(P_hat_{t+1} - P_t) = sign(P_{t+1} - P_t)} (undefined for RW).
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ASSETS, DEV_END, OOS_START  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import UNIVERSE, load_daily  # noqa: E402

LAGS = 60
rows = []
for a in ASSETS:
    P = load_daily(a)["close"].loc[:DEV_END].to_numpy()
    idx = load_daily(a).loc[:DEV_END].index
    X = np.column_stack([P[LAGS - 1 - j:len(P) - 1 - j] for j in range(LAGS)])  # lag 1..60 at time t+1
    target = P[LAGS:]
    dates = idx[LAGS:]
    pred = np.full(len(target), np.nan)
    coefs = []
    for yr in range(max(pd.Timestamp(OOS_START).year, dates[0].year + 3), dates[-1].year + 1):
        tr = dates < pd.Timestamp(f"{yr}-01-01")
        te = (dates >= pd.Timestamp(f"{yr}-01-01")) & (dates < pd.Timestamp(f"{yr + 1}-01-01"))
        if tr.sum() < 500 or te.sum() == 0:
            continue
        A = np.c_[np.ones(tr.sum()), X[tr]]
        b = np.linalg.lstsq(A, target[tr], rcond=None)[0]
        coefs.append(b)
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ b
    ok = np.isfinite(pred)
    y, yh, prev = target[ok], pred[ok], X[ok, 0]
    rw = prev
    dy = np.sign(np.diff(y))
    r = np.sign(y - prev)
    B = np.array(coefs)
    rows.append({
        "asset": UNIVERSE[a].label, "N": ok.sum(),
        "RMSE_RW": np.sqrt(np.mean((y - rw) ** 2)), "RMSE_AR60": np.sqrt(np.mean((y - yh) ** 2)),
        "ratio": np.sqrt(np.mean((y - yh) ** 2)) / np.sqrt(np.mean((y - rw) ** 2)),
        "sum_coef": B[:, 1:].sum(1).mean(), "w_lag1": B[:, 1].mean(),
        "corr_pred_lag": np.corrcoef(yh, prev)[0, 1],
        "diffDA_RW": np.mean(np.sign(np.diff(rw)) == dy), "diffDA_AR60": np.mean(np.sign(np.diff(yh)) == dy),
        "sign_persist": np.mean(r[1:] == r[:-1]),
        "properDA_AR60": np.mean(np.sign(yh - prev) == r),
    })
out = pd.DataFrame(rows)
out.to_csv(TABLES / "lagging_paradox.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
