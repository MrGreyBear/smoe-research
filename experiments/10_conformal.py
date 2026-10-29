"""Adaptive conformal calibration of VaR for every asset and model (before vs after)."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import ASSETS, load_panels  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.conformal import aci_quantiles  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
MODELS = ["SMoE", "GARCH-t", "Gauss-HAR", "Deep-SMoE", "HistGauss"]
ETA = 0.005


def run(asset):
    rows = []
    for m, p in load_panels(period, asset, MODELS).items():
        for a in (0.01, 0.05):
            raw = p.y < p.fc.quantile(a)
            cal = aci_quantiles(p.fc, p.y, a, eta=ETA)
            for kind, hits in (("raw", raw), ("ACI", cal["hits"])):
                k, c = T.kupiec(hits, a), T.christoffersen(hits, a)
                rows.append({"asset": asset, "model": m, "a": a, "kind": kind, "rate": hits.mean(),
                             "kupiec_p": k["p"], "cc_p": c["p_cc"], "ind_p": c["p_ind"],
                             "bound": cal["bound"] if kind == "ACI" else np.nan, "n": len(hits)})
    return rows


if __name__ == "__main__":
    rows = sum(Parallel(n_jobs=10)(delayed(run)(a) for a in ASSETS), [])
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"conformal{'' if period == 'dev' else '_' + period}.csv", index=False)
    summ = df.groupby(["model", "a", "kind"]).agg(mean_rate=("rate", "mean"), max_abs_dev=("rate", lambda r: 0),
                                                  frac_kupiec_reject=("kupiec_p", lambda p: np.mean(p < 0.05)),
                                                  frac_cc_reject=("cc_p", lambda p: np.mean(p < 0.05)))
    summ["max_abs_dev"] = df.assign(dev=(df["rate"] - df["a"]).abs()).groupby(["model", "a", "kind"])["dev"].max()
    summ.to_csv(TABLES / f"conformal_summary{'' if period == 'dev' else '_' + period}.csv")
    print(summ.round(4).to_string())
