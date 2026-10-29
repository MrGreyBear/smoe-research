"""Summary figures for the out-of-sample results (reads cached forecasts and result CSVs)."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from common import ASSETS, load_panels  # noqa: E402
from smoe import FIGURES, TABLES  # noqa: E402
from smoe import plotstyle as ps  # noqa: E402
from smoe.data import UNIVERSE  # noqa: E402

ps.use()
# diverging blue <-> red with the reference neutral grey midpoint (blue = better than GARCH-t)
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a92", "#b22d2c"])
ORDER = ["HistGauss", "MS-AR", "XGB-Gauss", "Gauss-HAR", "Chronos-Bolt-small", "Chronos-Bolt-base", "TimesFM-2.5",
         "AR-GJR-t",
         "MAR", "LSTAR", "Deep-SMoE", "SMoE"]

# 1) log-score difference vs GARCH-t, assets x models
L = pd.read_csv(TABLES / "eval_losses.csv")
piv = L.pivot(index="asset", columns="model", values="NLL")
d = piv.sub(piv["GARCH-t"], axis=0)[[m for m in ORDER if m in piv.columns]]
d = d.reindex([a for a in ASSETS if a in d.index])
fig, ax = plt.subplots(figsize=(7.0, 5.2))
lim = 0.06
im = ax.imshow(d.clip(-lim, lim).to_numpy(), cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
ax.set_xticks(range(d.shape[1]), d.columns, rotation=40, ha="right")
ax.set_yticks(range(d.shape[0]), [UNIVERSE[a].label for a in d.index])
ax.grid(False)
for i in range(d.shape[0]):
    for j in range(d.shape[1]):
        v = d.iat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 * lim else ps.INK)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("log-score loss minus GARCH-$t$ (blue = better)")
ax.set_title("Out-of-sample negative log score relative to GARCH-$t$ (cell labels $\\times 100$)", loc="left")
ps.save(fig, FIGURES / "heatmap_nll.pdf")

# 2) PIT histograms: SMoE vs GARCH-t vs Chronos on CSI 300 and S&P 500
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True, sharey=True)
for r, asset in enumerate(["csi300", "spx"]):
    P = load_panels("dev", asset, ["SMoE", "GARCH-t", "Chronos-Bolt-small"])
    for c, m in enumerate(["SMoE", "GARCH-t", "Chronos-Bolt-small"]):
        ax = axes[r, c]
        if m in P:
            pit = P[m].fc.pit(P[m].y)
            ax.hist(pit, bins=20, range=(0, 1), density=True, color=ps.SERIES[c], edgecolor="white", lw=0.5)
        ax.axhline(1.0, color=ps.INK2, lw=0.6, ls="--")
        ax.set_ylim(0, 2.2)
        if r == 0:
            ax.set_title(m)
        if c == 0:
            ax.set_ylabel(UNIVERSE[asset].label + "\ndensity")
for ax in axes[1]:
    ax.set_xlabel("PIT")
ps.save(fig, FIGURES / "pit_hist.pdf")

# 3) cross-market fingerprints: robust t-statistics of short-horizon and slow-momentum loadings
fp = pd.read_csv(TABLES / "insample_fingerprints.csv")
fp["label"] = fp["asset"].map(lambda a: UNIVERSE[a].label)
fp = fp.iloc[::-1]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.4), sharey=True)
for ax, (col, title) in zip(axes, [("beta[reversal]:rev1_t", "short-horizon expert\n1-day return"),
                                   ("beta[reversal]:rev5_t", "short-horizon expert\n5-day return"),
                                   ("beta[trend]:mom120_t", "trend expert\n120-day momentum")]):
    v = fp[col].to_numpy()
    colors = [ps.SERIES[0] if x > 0 else ps.SERIES[7] for x in v]
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.barh(range(len(v)), v, color=colors, height=0.6)
    ax.axvline(0, color=ps.INK2, lw=0.6)
    for x in (-1.96, 1.96):
        ax.axvline(x, color=ps.MUTED, lw=0.5, ls=":")
    ax.set_title(title)
    ax.set_xlabel("robust $t$")
axes[0].set_yticks(range(len(fp)), fp["label"])
ps.save(fig, FIGURES / "fingerprints.pdf")
print("figures written")

# 4) yardstick: fidelity Phi per teacher and series, plus teacher vs shadow log-score gains
ys = pd.read_csv(TABLES / "yardstick.csv")
ys["gain_T"] = ys["LS_teacher"] - ys["LS_reference"]
ys["gain_S"] = ys["LS_student"] - ys["LS_reference"]
teachers = [t for t in ["Deep-SMoE", "XGB-Gauss", "MS-AR", "GARCH-t", "Chronos-Bolt-small", "Chronos-Bolt-base",
                        "TimesFM-2.5"]
            if t in set(ys["teacher"])]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
ax = axes[0]
for i, t in enumerate(teachers):
    v = ys.loc[ys.teacher == t, "Phi"].clip(-1, 1.5).to_numpy()
    jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.3
    ax.scatter(np.full(len(v), i) + jitter, v, s=10, color=ps.SERIES[i % 8], alpha=0.85, lw=0)
    ax.hlines(np.median(v), i - 0.3, i + 0.3, color=ps.INK, lw=1.2)
ax.axhline(1, color=ps.MUTED, lw=0.6, ls="--")
ax.set_xticks(range(len(teachers)), teachers, rotation=35, ha="right")
ax.set_ylabel(r"fidelity $\Phi$ (clipped to $[-1,1.5]$)")
ax.set_title("Share of teacher's log-score gain reproduced", loc="left")
ax = axes[1]
for i, t in enumerate(teachers):
    s = ys[ys.teacher == t]
    ax.scatter(s["gain_T"], s["gain_S"], s=10, color=ps.SERIES[i % 8], label=t, lw=0)
lo = min(ys["gain_T"].min(), ys["gain_S"].min())
hi = max(ys["gain_T"].max(), ys["gain_S"].max())
ax.plot([lo, hi], [lo, hi], color=ps.MUTED, lw=0.6, ls="--")
ax.set_xlabel("teacher log-score gain over HistGauss")
ax.set_ylabel("shadow log-score gain")
ax.legend(fontsize=6, loc="upper left")
ps.save(fig, FIGURES / "yardstick.pdf")
print("yardstick figure written")

# 5) out-of-sample SMoE gate weights for every series (small multiples)
for part, names in enumerate([ASSETS[:10], ASSETS[10:]]):
    fig, axes = plt.subplots(len(names), 1, figsize=(7.0, 1.0 * len(names) + 0.6), sharex=True)
    for ax, a in zip(np.atleast_1d(axes), names):
        P = load_panels("dev", a, ["SMoE"])
        if "SMoE" not in P:
            continue
        p = P["SMoE"]
        w = pd.DataFrame(p.fc.w, index=p.index, columns=["trend", "reversal", "crisis"]).rolling(20, min_periods=1).mean()
        ax.stackplot(w.index, *[w[k] for k in w.columns], colors=[ps.EXPERT_COLORS[k] for k in w.columns], lw=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.005, 0.92, UNIVERSE[a].label, transform=ax.transAxes, fontsize=7, va="top", color=ps.INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
        ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ps.EXPERT_COLORS[k]) for k in ["trend", "reversal", "crisis"]]
    np.atleast_1d(axes)[0].legend(handles, ["trend", "reversal", "crisis"], ncol=3, loc="lower left",
                                  bbox_to_anchor=(0, 1.02))
    ps.save(fig, FIGURES / f"gates_oos_{part + 1}.pdf")
print("gate small multiples written")
