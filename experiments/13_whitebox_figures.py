"""White-box diagnostics for CSI 300 (development sample, full-sample SMoE).

1. Exact attribution of the predictive mean to experts (20-day averages).
2. Gate elasticity of the predictive mean and of the 1% VaR with respect to log volatility
   (closed forms of Proposition 'gate elasticity' and Appendix 'gate elasticity of tail risk').
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from common import DEV_END, features  # noqa: E402
from smoe import FIGURES, TABLES  # noqa: E402
from smoe import plotstyle as ps  # noqa: E402
from smoe.models import SMoE  # noqa: E402

ps.use()
fs = features("csi300").slice(None, DEV_END)
ok = fs.y.notna()
X, y = fs.X[ok], fs.y[ok]
m = SMoE().fit(X, y)

# 1) attribution by expert
A = m.attribution(X)
by_exp = pd.DataFrame({k: A[[c for c in A.columns if c.startswith(k + ":")]].sum(1) for k in m.expert_names})
sm = by_exp.rolling(20, min_periods=1).mean()
fc = m.predict(X)
assert np.allclose(by_exp.sum(1), fc.mean())

# 2) elasticities w.r.t. the gate's log-volatility state (per one standard deviation)
el_mean = m.gate_elasticity(X)["g_lvol"]
beta, delta, gamma = m.params_
G = np.vstack([gamma, np.zeros((1, gamma.shape[1]))])
j = 1 + m.gate_cols.index("g_lvol")
pi = fc.w
gbar = pi @ G[:, j]
a = 0.01
q = fc.quantile(a)
z = (q[:, None] - fc.mu) / fc.sd
dens = np.sum(pi * norm.pdf(z) / fc.sd, axis=1)
dF = np.sum(pi * (G[:, j][None, :] - gbar[:, None]) * norm.cdf(z), axis=1)
el_var = pd.Series(-dF / dens, index=X.index)

fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
for k in m.expert_names:
    axes[0].plot(sm.index, sm[k], color=ps.EXPERT_COLORS[k], label=k, lw=0.9)
axes[0].axhline(0, color=ps.INK2, lw=0.5)
axes[0].set_ylabel("contribution to\npredictive mean (%)")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
axes[1].plot(X.index, pd.Series(fc.prob_up(), index=X.index).rolling(20, min_periods=1).mean(), color=ps.SERIES[0], lw=0.9)
axes[1].axhline(0.5, color=ps.INK2, lw=0.5, ls="--")
axes[1].set_ylabel("$P(y_t>0)$\n(20-day mean)")
axes[2].plot(el_var.index, el_var.rolling(20, min_periods=1).mean(), color=ps.SERIES[7], lw=0.9)
axes[2].axhline(0, color=ps.INK2, lw=0.5)
axes[2].set_ylabel("$\\partial\\,\\mathrm{VaR}_{1\\%}/\\partial z_{\\mathrm{vol}}$\nvia gate (pp per s.d.)")
ps.save(fig, FIGURES / "csi300_whitebox.pdf")

summ = pd.DataFrame({
    "quantity": ["mean contribution: trend", "mean contribution: reversal", "mean contribution: crisis",
                 "share of days P(up) > 0.5", "mean gate elasticity of mean (pp/s.d.)",
                 "mean gate elasticity of 1% VaR (pp/s.d.)", "5th pct gate elasticity of 1% VaR",
                 "95th pct gate elasticity of 1% VaR"],
    "value": [by_exp["trend"].mean(), by_exp["reversal"].mean(), by_exp["crisis"].mean(),
              float(np.mean(fc.prob_up() > 0.5)), el_mean.mean(), el_var.mean(), el_var.quantile(0.05),
              el_var.quantile(0.95)]})
summ.to_csv(TABLES / "csi300_whitebox.csv", index=False)
print(summ.round(4).to_string(index=False))
