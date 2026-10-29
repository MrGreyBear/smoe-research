"""Figures for the extension sections (realised variance, number of experts and gate)."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib.util  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DEV_END  # noqa: E402
from smoe import FIGURES, TABLES  # noqa: E402
from smoe import plotstyle as ps  # noqa: E402
from smoe.data.realized import load_csi300_realized  # noqa: E402

ps.use()

# 1) HAR loadings with range-based and realised daily variance (K = 1), and the EIV prediction
if (TABLES / "rv_measurement.csv").exists():
    spec = importlib.util.spec_from_file_location("rv", "experiments/14_rv_robustness.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    fs = rv.rv_features("csi300", load_csi300_realized()).slice(None, DEV_END)
    ok = fs.y.notna()
    X, y = fs.X[ok], fs.y[ok]
    est = {}
    for name, lab in [("Gauss-HAR-PK", "daily range (Parkinson)"), ("Gauss-HAR-RV", "5-minute realised variance")]:
        m = rv.VARIANTS[name]().fit(X, y)
        inf = m.inference(X, y)
        est[lab] = [(inf.loc[f"delta[single]:{c}", "estimate"] / m.scale_[c],
                     inf.loc[f"delta[single]:{c}", "se_sandwich"] / m.scale_[c]) for c in m.var_cols]
    meas = pd.read_csv(TABLES / "rv_measurement.csv").set_index("estimator")
    att_pk = meas.loc["Parkinson range (BM)", "attenuation"]
    att_rv = meas.loc["5-min RV (M=48)", "attenuation"]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    xs = np.arange(3)
    for i, (lab, vals) in enumerate(est.items()):
        b = np.array([v[0] for v in vals])
        s = np.array([v[1] for v in vals])
        ax.errorbar(xs + (i - 0.5) * 0.22, b, yerr=1.96 * s, fmt="o", ms=5, color=ps.SERIES[i], lw=1.2,
                    capsize=0, label=lab)
    pred = est["5-minute realised variance"][0][0] / att_rv * att_pk
    ax.scatter([-0.11], [pred], s=48, facecolors="none", edgecolors=ps.SERIES[0], lw=1.2, zorder=3,
               label="range, daily: predicted from the RV estimate")
    ax.set_xticks(xs, ["daily", "weekly", "monthly"])
    ax.set_ylabel("loading on log variance component")
    ax.axhline(0, color=ps.MUTED, lw=0.6)
    ax.legend(loc="upper left", fontsize=7)
    ps.save(fig, FIGURES / "rv_har_loadings.pdf")
    print("rv figure written")

# 2) every specification against the SMoE, by series
if (TABLES / "kg_losses.csv").exists():
    L = pd.read_csv(TABLES / "kg_losses.csv")
    D = pd.read_csv(TABLES / "kg_dm.csv")
    piv = L.pivot(index="asset", columns="model", values="NLL")
    diff = piv.sub(piv["SMoE"], axis=0) * 100
    order = ["Gauss-HAR", "SMoE-K2tc", "SMoE-K2rc", "SMoE-K2tr", "LSTAR", "MAR", "SMoE-K4c", "SMoE-K4p", "SMoE-K5",
             "SMoE-Gvol", "SMoE-Gvix", "SMoE-Gasym", "SMoE-Grich"]
    order = [m for m in order if m in diff.columns]
    sig = D[D.loss == "NLL"].set_index(["asset", "model"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for j, m in enumerate(order):
        v = diff[m].clip(-3, 8)
        better = np.array([sig.loc[(a, m), "p_model_better"] < 0.05 for a in v.index])
        worse = np.array([(sig.loc[(a, m), "p_two"] < 0.05) and (sig.loc[(a, m), "mean_diff"] < 0) for a in v.index])
        jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.35
        col = ps.SERIES[0] if m.startswith("SMoE-G") or m in ("MAR", "LSTAR") else ps.SERIES[1]
        ax.scatter(v[~better & ~worse], j + jit[~better & ~worse], s=12, facecolors="none", edgecolors=col, lw=0.8)
        ax.scatter(v[better | worse], j + jit[better | worse], s=14, color=col, lw=0)
        ax.vlines(v.median(), j - 0.35, j + 0.35, color=ps.INK, lw=1.4)
    ax.axvline(0, color=ps.MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel(r"log-score loss minus SMoE ($\times100$; right = worse than SMoE; clipped to $[-3,8]$)")
    ax.scatter([], [], s=14, color=ps.SERIES[1], label="number of experts")
    ax.scatter([], [], s=14, color=ps.SERIES[0], label="gate covariates")
    ax.scatter([], [], s=12, facecolors="none", edgecolors=ps.MUTED, label="not significant")
    ax.legend(loc="lower right", fontsize=7)
    ps.save(fig, FIGURES / "kg_strip.pdf")
    print("kg figure written")
