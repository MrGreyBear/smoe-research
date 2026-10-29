"""Out-of-sample evaluation of every model on every asset (common dates per asset).

Writes
  results/tables/eval_losses.csv      mean losses per asset x model
  results/tables/eval_tests.csv       calibration / directional / nested tests per asset x model
  results/tables/eval_dm.csv          DM-HLN tests SMoE vs each benchmark per loss
  results/tables/eval_mcs.csv         MCS p-values per asset x loss x model
  results/tables/eval_csi300_rv.csv   CSI 300 variance forecasts against 5-minute RV (+ overnight^2)
usage: python experiments/06_evaluate.py [dev|final] [start] [end]
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ASSETS, load_panels  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.data import UNIVERSE  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402
from smoe.evaluation.losses import all_losses, qlike  # noqa: E402

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
start = sys.argv[2] if len(sys.argv) > 2 else None
end = sys.argv[3] if len(sys.argv) > 3 else None
suffix = "" if period == "dev" and start is None else f"_{period}" + (f"_{start}" if start else "")
LOSSES_FOR_DM = ["NLL", "CRPS", "SE", "Brier", "QLIKE", "FZ5", "FZ1"]
MCS_LOSSES = ["NLL", "CRPS", "QLIKE", "FZ5"]
FOCAL = "SMoE"


def common_panels(asset):
    P = load_panels(period, asset, None)
    from common import panel_path
    import glob
    from smoe.walkforward import ForecastPanel
    for f in glob.glob(str(panel_path(period, asset, "*"))):
        name = f.split("__")[-1][:-4]
        # main comparison only: ablation variants (SMoE-*) and yardstick shadows are evaluated elsewhere
        if name.startswith("SMoE-") or name.startswith("Shadow["):
            continue
        if name not in P:
            P[name] = ForecastPanel.from_npz(f)
    P = {k: v.loc(start, end) for k, v in P.items()}
    P = {k: v for k, v in P.items() if len(v.index)}
    if not P:
        return {}
    common = sorted(set.intersection(*[set(v.index) for v in P.values()]))
    out = {}
    for k, v in P.items():
        pos = pd.Index(v.index).get_indexer(common)
        out[k] = (pd.DatetimeIndex(common), v.y[pos], v.fc.subset(pos))
    return out


loss_rows, test_rows, dm_rows, mcs_rows = [], [], [], []
for asset in ASSETS:
    P = common_panels(asset)
    if FOCAL not in P:
        print("skip", asset)
        continue
    idx, y, _ = P[FOCAL]
    L = {}
    for m, (_, ym, fc) in P.items():
        L[m] = all_losses(fc, ym)
        row = {"asset": asset, "model": m, "n": len(ym), **L[m].mean().to_dict()}
        loss_rows.append(row)
        mean = fc.mean()
        hist = P["HistGauss"][2].mean() if "HistGauss" in P else np.zeros_like(mean)
        pt = T.pesaran_timmermann(ym, mean)
        pu = fc.prob_up()
        mb = T.murphy_brier(pu, ym > 0)
        bk = T.berkowitz(fc.pit(ym))
        tr = {"asset": asset, "model": m, "R2oos_zero": T.oos_r2(ym, mean, np.zeros_like(mean)),
              "R2oos_hist": T.oos_r2(ym, mean, hist), "CW_p_zero": T.clark_west(ym, np.zeros_like(mean), mean)["p"],
              "hit_rate": pt["hit_rate"], "PT_p": pt["p"], "brier": mb["brier"], "brier_rel": mb["reliability"],
              "brier_res": mb["resolution"], "berk_p": bk["p"], "pit_sd": bk["sigma"]}
        for a in (0.01, 0.05):
            hits = ym < fc.quantile(a)
            tr[f"viol{int(a * 100)}"] = hits.mean()
            tr[f"kupiec{int(a * 100)}_p"] = T.kupiec(hits, a)["p"]
            tr[f"chris{int(a * 100)}_p"] = T.christoffersen(hits, a)["p_cc"]
        mz = T.mincer_zarnowitz(ym**2, fc.var())
        tr.update({"MZ_b": mz["b"], "MZ_R2": mz["R2"], "MZ_p": mz["p"]})
        test_rows.append(tr)
    for m in L:
        if m == FOCAL:
            continue
        for loss in LOSSES_FOR_DM:
            d = T.dm_test(L[m][loss].to_numpy(), L[FOCAL][loss].to_numpy())
            dm_rows.append({"asset": asset, "benchmark": m, "loss": loss, "mean_diff": d["mean_diff"],
                            "dm_stat": d["stat"], "p_smoe_better": d["p_b_better"], "p_two": d["p_two"]})
    for loss in MCS_LOSSES:
        mat = pd.DataFrame({m: L[m][loss].to_numpy() for m in L})
        mc = T.model_confidence_set(mat, alpha=0.10, B=1000)
        for m, r in mc.iterrows():
            mcs_rows.append({"asset": asset, "loss": loss, "model": m, "mean_loss": r["mean_loss"],
                             "mcs_p": r["mcs_p"], "in_mcs": r["in_mcs"]})
    print("done", asset, len(idx), flush=True)

pd.DataFrame(loss_rows).to_csv(TABLES / f"eval_losses{suffix}.csv", index=False)
pd.DataFrame(test_rows).to_csv(TABLES / f"eval_tests{suffix}.csv", index=False)
pd.DataFrame(dm_rows).to_csv(TABLES / f"eval_dm{suffix}.csv", index=False)
pd.DataFrame(mcs_rows).to_csv(TABLES / f"eval_mcs{suffix}.csv", index=False)

# ---- CSI 300 variance forecasts against realised variance from 5-minute data
try:
    from smoe.data.realized import build_csi300_realized
    rm = build_csi300_realized()
    proxy = (rm["rv"] + rm["ret_overnight"].fillna(0) ** 2)
    P = common_panels("csi300")
    rows = []
    for m, (idx, ym, fc) in P.items():
        # forecast at t targets day t+1: align proxy of the next trading day
        nxt = pd.Series(proxy.index, index=proxy.index).shift(-1)
        s = pd.Series(fc.var(), index=idx)
        target_day = pd.Series(idx, index=idx).map(lambda d: nxt.get(d, pd.NaT))
        ok = target_day.notna()
        pv = proxy.reindex(target_day[ok]).to_numpy()
        hv = s[ok].to_numpy()
        good = np.isfinite(pv)
        rows.append({"model": m, "n": int(good.sum()), "QLIKE_RV": qlike(hv[good], pv[good]).mean(),
                     "MSE_RV": np.mean((pv[good] - hv[good]) ** 2),
                     **{f"MZ_{k}": v for k, v in T.mincer_zarnowitz(pv[good], hv[good]).items() if k in ("b", "R2", "p")}})
    pd.DataFrame(rows).to_csv(TABLES / f"eval_csi300_rv{suffix}.csv", index=False)
except Exception as e:  # 5-minute data only covers 2009-05 .. 2025-07
    print("RV evaluation skipped:", e)
print("evaluation complete")
