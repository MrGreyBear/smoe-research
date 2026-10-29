"""Systematic comparison of the number of experts K and of the gate's state variables.

All 19 series, development period, same walk-forward protocol as the main run. Existing panels
(Gauss-HAR = K1, SMoE = K3 with the baseline gate, MAR = constant gate, LSTAR) are reused.

K dimension (baseline gate: log 22d variance, 60d momentum, drawdown)
  SMoE-K2tc   trend + crisis           SMoE-K4c   + calm (second intercept-only expert)
  SMoE-K2rc   reversal + crisis        SMoE-K4p   + premium (mean in log 22d variance, drawdown)
  SMoE-K2tr   trend + reversal         SMoE-K5    + calm + premium
Gate dimension (baseline three experts)
  SMoE-Gvol   log 22d variance only    SMoE-Gasym  baseline + signed last return
  SMoE-Gvix   baseline + log VIX       SMoE-Grich  baseline + VIX + last return + log 1d variance

In-sample (development sample): log-likelihood, AIC, BIC of every specification; sandwich Wald
tests for the VIX gate loadings and t-statistics of the premium expert.
Writes results/tables/kg_*.csv
usage: python experiments/15_k_gate.py [n_jobs]
"""

import shutil
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import ASSETS, DEV_END, FORECASTS, OOS_START, features, load_panels  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import load_daily  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402
from smoe.evaluation.losses import all_losses  # noqa: E402
from smoe.features.extra import add_vix  # noqa: E402
from smoe.features.semantic import SEMANTIC_GROUPS as G  # noqa: E402
from smoe.models import SMoE, constant_gate, lstar, single_gaussian  # noqa: E402
from smoe.walkforward import ForecastPanel, walk_forward  # noqa: E402

TR, RE, PREM = G["trend"], G["reversal"], ["lv22", "g_dd"]
GATE = G["gate"]
VARIANTS = {
    "SMoE-K2tc": lambda: SMoE(experts={"trend": TR, "crisis": []}, name="SMoE-K2tc"),
    "SMoE-K2rc": lambda: SMoE(experts={"reversal": RE, "crisis": []}, name="SMoE-K2rc"),
    "SMoE-K2tr": lambda: SMoE(experts={"trend": TR, "reversal": RE}, name="SMoE-K2tr"),
    "SMoE-K4c": lambda: SMoE(experts={"trend": TR, "reversal": RE, "calm": [], "crisis": []}, name="SMoE-K4c"),
    "SMoE-K4p": lambda: SMoE(experts={"trend": TR, "reversal": RE, "premium": PREM, "crisis": []}, name="SMoE-K4p"),
    "SMoE-K5": lambda: SMoE(experts={"trend": TR, "reversal": RE, "premium": PREM, "calm": [], "crisis": []},
                            name="SMoE-K5"),
    "SMoE-Gvol": lambda: SMoE(gate_cols=["g_lvol"], name="SMoE-Gvol"),
    "SMoE-Gvix": lambda: SMoE(gate_cols=GATE + ["g_vix"], name="SMoE-Gvix"),
    "SMoE-Gasym": lambda: SMoE(gate_cols=GATE + ["rev1"], name="SMoE-Gasym"),
    "SMoE-Grich": lambda: SMoE(gate_cols=GATE + ["g_vix", "rev1", "lv1"], name="SMoE-Grich"),
}
EXISTING = {"Gauss-HAR": single_gaussian, "SMoE": SMoE, "MAR": constant_gate, "LSTAR": lstar}
# identical specifications already estimated by the ablation study (07_ablation.py)
ALIASES = {"SMoE-K2tc": "SMoE-K2", "SMoE-Gvol": "SMoE-volgate"}
K_OF = {"Gauss-HAR": 1, "SMoE-K2tc": 2, "SMoE-K2rc": 2, "SMoE-K2tr": 2, "LSTAR": 2, "SMoE": 3, "MAR": 3,
        "SMoE-K4c": 4, "SMoE-K4p": 4, "SMoE-K5": 5, "SMoE-Gvol": 3, "SMoE-Gvix": 3, "SMoE-Gasym": 3,
        "SMoE-Grich": 3}
LOSSES = ("NLL", "CRPS", "QLIKE", "FZ5")
VIX = load_daily("vix")["close"]


def kg_features(asset):
    return add_vix(features(asset), VIX)


def run(asset, name):
    path = FORECASTS / "dev" / f"{asset}__{name}.npz"
    alias = FORECASTS / "dev" / f"{asset}__{ALIASES.get(name, '-')}.npz"
    if not path.exists() and alias.exists():
        shutil.copy(alias, path)
    if not path.exists():
        walk_forward(VARIANTS[name], kg_features(asset), start=OOS_START, end=DEV_END, freq="QS",
                     asset=asset).to_npz(path)
    return asset, name


def insample(asset):
    fs = kg_features(asset).slice(None, DEV_END)
    ok = fs.y.notna()
    X, y = fs.X[ok], fs.y[ok]
    rows, extra = [], {"asset": asset}
    for name, f in {**EXISTING, **VARIANTS}.items():
        m = f().fit(X, y)
        rows.append({"asset": asset, "model": name, "K": K_OF[name], "loglik": m.loglik_, "params": m.n_params_,
                     "AIC": m.aic, "BIC": m.bic})
        if name == "SMoE-Gvix":
            th, V = m.sandwich_cov(X, y)
            extra.update({f"vix_{k}": v for k, v in m.wald(th, V, lambda n: n.endswith(":g_vix")).items()})
            inf = m.inference(X, y)
            for e in m.expert_names[:-1]:
                extra[f"vix_z[{e}]"] = inf.loc[f"gamma[{e}]:g_vix", "z"]
        if name == "SMoE-K4p":
            inf = m.inference(X, y)
            for c in PREM:
                extra[f"prem_{c}"] = inf.loc[f"beta[premium]:{c}", "estimate"]
                extra[f"prem_{c}_z"] = inf.loc[f"beta[premium]:{c}", "z"]
            extra["pi_premium"] = m.gate_probs(X)["premium"].mean()
        if name == "SMoE-Gasym":
            inf = m.inference(X, y)
            for e in m.expert_names[:-1]:
                extra[f"asym_z[{e}]"] = inf.loc[f"gamma[{e}]:rev1", "z"]
    return rows, extra


def evaluate():
    models = list(EXISTING) + list(VARIANTS)
    loss_rows, dm_rows, mcs_rows = [], [], []
    for a in ASSETS:
        P = load_panels("dev", a, models)
        idx = sorted(set.intersection(*[set(p.index) for p in P.values()]))
        L = {}
        for m, p in P.items():
            pos = pd.Index(p.index).get_indexer(idx)
            fc, y = p.fc.subset(pos), p.y[pos]
            L[m] = all_losses(fc, y)
            hits = y < fc.quantile(0.01)
            loss_rows.append({"asset": a, "model": m, "K": K_OF[m], "n": len(y), **L[m][list(LOSSES)].mean(),
                              "kupiec1_p": T.kupiec(hits, 0.01)["p"]})
        for m in L:
            if m == "SMoE":
                continue
            for loss in LOSSES:
                d = T.dm_test(L["SMoE"][loss].to_numpy(), L[m][loss].to_numpy())
                dm_rows.append({"asset": a, "model": m, "loss": loss, "mean_diff": d["mean_diff"],
                                "dm_stat": d["stat"], "p_model_better": d["p_b_better"], "p_two": d["p_two"]})
        for loss in ("NLL", "QLIKE", "FZ5"):
            mc = T.model_confidence_set(pd.DataFrame({m: L[m][loss].to_numpy() for m in L}), alpha=0.10, B=1000)
            for m, r in mc.iterrows():
                mcs_rows.append({"asset": a, "loss": loss, "model": m, "mcs_p": r["mcs_p"], "in_mcs": r["in_mcs"]})
        print("evaluated", a, len(idx), flush=True)
    pd.DataFrame(loss_rows).to_csv(TABLES / "kg_losses.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(TABLES / "kg_dm.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(TABLES / "kg_mcs.csv", index=False)


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # longest jobs first so the pool does not idle at the end
    order = ["SMoE-K5", "SMoE-K4c", "SMoE-K4p", "SMoE-Grich", "SMoE-Gvix", "SMoE-Gasym", "SMoE-K2tr",
             "SMoE-K2rc", "SMoE-K2tc", "SMoE-Gvol"]
    if sys.argv[2:3] == ["reverse"]:  # helper worker: same jobs from the other end, then stop
        Parallel(n_jobs=n_jobs)(delayed(run)(a, v) for v in order[::-1] for a in ASSETS[::-1])
        sys.exit()
    Parallel(n_jobs=n_jobs)(delayed(run)(a, v) for v in order for a in ASSETS)
    res = Parallel(n_jobs=n_jobs)(delayed(insample)(a) for a in ASSETS)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(TABLES / "kg_insample.csv", index=False)
    pd.DataFrame([e for _, e in res]).to_csv(TABLES / "kg_insample_extra.csv", index=False)
    evaluate()
