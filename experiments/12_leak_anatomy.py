"""Anatomy of a leak: reproduce the two look-ahead mechanisms of the undergraduate thesis on clean data.

Leak A ("target in the inputs"): the 3-day forward mean return is used as a feature alongside the
causal features, exactly as cols = [target_col] + feature_cols did, with a 10-day input window.
Leak B ("residual correction"): a price-level forecast is corrected by 0.6 x the 3-day trailing mean
of the realised error, which includes today's realised price.
Each is compared with its leak-free counterpart on CSI 300, OOS 2012 -> June 2024, yearly refits.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from common import DEV_END, OOS_START, features  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import load_daily  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402

fs = features("csi300").slice(None, DEV_END)
X = fs.X.assign(r=fs.r)
r = fs.r
y3 = r.rolling(3).mean().shift(-3).rename("y3")  # mean of r_{t+1..t+3}: the thesis's 3-day target
cols = list(fs.X.columns) + ["r"]


def windows(df, L=10):
    A = df.to_numpy()
    out = np.full((len(A), L * A.shape[1]), np.nan)
    for i in range(L - 1, len(A)):
        out[i] = A[i - L + 1:i + 1].ravel()
    return pd.DataFrame(out, index=df.index)


def walk(Xw, target, model_fn):
    pred = pd.Series(np.nan, index=Xw.index)
    for yr in range(pd.Timestamp(OOS_START).year, pd.Timestamp(DEV_END).year + 1):
        tr = (Xw.index < f"{yr}-01-01") & Xw.notna().all(1) & target.notna()
        te = (Xw.index >= f"{yr}-01-01") & (Xw.index < f"{yr + 1}-01-01") & Xw.notna().all(1) & target.notna()
        if te.sum() == 0:
            continue
        m = model_fn().fit(Xw[tr], target[tr])
        pred[te] = m.predict(Xw[te])
    return pred


rows = []
xgb = lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0, n_jobs=4)  # noqa: E731
# The thesis's input window ended at row t-1 and contained Target_Ret[t-1] = mean(r_t, r_{t+1}, r_{t+2}),
# which overlaps two of the three days of the target at row t. With our windows ending at row t this is
# the 3-day target lagged by one row.
for label, feats in [("leak-free features", X[cols]),
                     ("features + lagged 3-day target (leak A)", X[cols].assign(y3=y3.shift(1)))]:
    Xw = windows(feats)
    for target_name, target in [("3-day mean", y3), ("next day", fs.y)]:
        p = walk(Xw, target, xgb)
        ok = p.notna() & target.notna()
        pt = T.pesaran_timmermann(target[ok], p[ok])
        rows.append({"mechanism": label, "target": target_name, "n": int(ok.sum()),
                     "hit_rate": pt["hit_rate"], "PT_p": pt["p"],
                     "R2_oos": T.oos_r2(target[ok], p[ok], np.zeros(ok.sum())),
                     "longonly_sharpe": (np.where(p[ok] > 0, fs.y[ok], 0).mean() / np.where(p[ok] > 0, fs.y[ok], 0).std()
                                         * np.sqrt(252))})

# Leak B on price levels: 60-lag OLS on prices, then the "residual correction" using realised prices
P = load_daily("csi300")["close"].loc[:DEV_END]
L = 60
lagM = np.column_stack([P.shift(j) for j in range(1, L + 1)])
dfp = pd.DataFrame(lagM, index=P.index).dropna()
target = P.loc[dfp.index]
pred = walk(dfp, target, lambda: Ridge(alpha=1e-6))
ok = pred.notna()
yv, ph, prev = target[ok].to_numpy(), pred[ok].to_numpy(), dfp[ok].iloc[:, 0].to_numpy()
corr = pd.Series(0.6 * (yv - ph)).rolling(3, min_periods=1).mean().to_numpy()  # uses today's realised price
ph_leak = ph + corr
for label, f in [("price model, leak-free", ph), ("price model + residual correction (leak B)", ph_leak),
                 ("random walk", prev)]:
    rows.append({"mechanism": label, "target": "price level", "n": int(ok.sum()),
                 "hit_rate": np.mean(np.sign(f - prev) == np.sign(yv - prev)) if label != "random walk" else np.nan,
                 "PT_p": np.nan, "R2_oos": np.nan, "longonly_sharpe": np.nan,
                 "RMSE": np.sqrt(np.mean((yv - f) ** 2)),
                 "diffDA": np.mean(np.sign(np.diff(f)) == np.sign(np.diff(yv)))})
out = pd.DataFrame(rows)
out.to_csv(TABLES / "leak_anatomy.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
