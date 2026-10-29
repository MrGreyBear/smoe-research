"""Descriptive statistics and stylised-fact tests for every asset (full sample through DEV_END)."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from common import ASSETS, DEV_END  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import UNIVERSE, load_daily  # noqa: E402

rows = []
for a in ASSETS:
    df = load_daily(a).loc[:DEV_END]
    r = (100 * np.log(df["close"]).diff()).dropna()
    lb = acorr_ljungbox(r, lags=[10])
    lb2 = acorr_ljungbox(r**2, lags=[10])
    rows.append({
        "asset": UNIVERSE[a].label, "group": UNIVERSE[a].group, "start": r.index[0].strftime("%Y-%m"),
        "N": len(r), "mean": r.mean(), "sd": r.std(), "skew": stats.skew(r), "exkurt": stats.kurtosis(r),
        "JB_p": stats.jarque_bera(r).pvalue, "ADF_logP_p": adfuller(np.log(df["close"]), autolag="AIC")[1],
        "ADF_r_p": adfuller(r, autolag="AIC")[1], "LB10_r_p": lb["lb_pvalue"].iloc[0],
        "LB10_r2_p": lb2["lb_pvalue"].iloc[0], "ARCH5_p": het_arch(r, nlags=5)[1],
        "rho1": r.autocorr(1), "rho1_abs": r.abs().autocorr(1),
    })
out = pd.DataFrame(rows)
TABLES.mkdir(parents=True, exist_ok=True)
out.to_csv(TABLES / "descriptives.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
