"""Out-of-sample ablations of the SMoE design (walk-forward, same protocol as the main run).

Variants
  SMoE                 the full model
  SMoE-unrestricted    every expert sees trend + reversal covariates (no semantic exclusions)
  SMoE-K2              trend and crisis experts only
  SMoE-constvar        expert variances constant (no HAR variance dynamics)
  SMoE-volgate         gate depends on volatility only
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import FORECASTS, OOS_START, DEV_END, features, load_panels  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402
from smoe.evaluation.losses import all_losses  # noqa: E402
from smoe.features.semantic import SEMANTIC_GROUPS as G  # noqa: E402
from smoe.models import SMoE  # noqa: E402
from smoe.walkforward import ForecastPanel, walk_forward  # noqa: E402

ASSETS = ["csi300", "csi500", "sse50", "chinext", "spx", "hsi", "gc_fut", "btc"]
VARIANTS = {
    "SMoE-unrestricted": lambda: SMoE(experts={"trend": G["trend"] + G["reversal"], "reversal": G["trend"] + G["reversal"],
                                               "crisis": []}, name="SMoE-unrestricted"),
    "SMoE-K2": lambda: SMoE(experts={"trend": G["trend"], "crisis": []}, name="SMoE-K2"),
    "SMoE-constvar": lambda: SMoE(var_cols=[], name="SMoE-constvar"),
    "SMoE-volgate": lambda: SMoE(gate_cols=["g_lvol"], name="SMoE-volgate"),
}


def run(asset, name):
    path = FORECASTS / "dev" / f"{asset}__{name}.npz"
    if not path.exists():
        walk_forward(VARIANTS[name], features(asset), start=OOS_START, end=DEV_END, freq="QS",
                     asset=asset).to_npz(path)
    return asset, name


if __name__ == "__main__":
    Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 10)(
        delayed(run)(a, v) for a in ASSETS for v in VARIANTS)
    rows = []
    for a in ASSETS:
        P = load_panels("dev", a, ["SMoE"])
        for v in VARIANTS:
            P[v] = ForecastPanel.from_npz(FORECASTS / "dev" / f"{a}__{v}.npz")
        idx = sorted(set.intersection(*[set(p.index) for p in P.values()]))
        L = {}
        for m, p in P.items():
            pos = pd.Index(p.index).get_indexer(idx)
            L[m] = all_losses(p.fc.subset(pos), p.y[pos])
        for v in VARIANTS:
            for loss in ("NLL", "CRPS", "QLIKE", "FZ5"):
                d = T.dm_test(L[v][loss].to_numpy(), L["SMoE"][loss].to_numpy())
                rows.append({"asset": a, "variant": v, "loss": loss, "diff_vs_full": d["mean_diff"],
                             "dm_stat": d["stat"], "p_full_better": d["p_b_better"], "p_two": d["p_two"]})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ablation.csv", index=False)
    s = df.groupby(["variant", "loss"]).agg(median_diff=("diff_vs_full", "median"),
                                             full_sig_better=("p_full_better", lambda p: int((p < 0.05).sum())),
                                             full_sig_worse=("dm_stat", lambda z: 0))
    s["full_sig_worse"] = df.assign(w=(df.p_two < 0.05) & (df.diff_vs_full < 0)).groupby(["variant", "loss"])["w"].sum()
    s.to_csv(TABLES / "ablation_summary.csv")
    print(s.to_string())
