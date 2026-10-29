"""White-box yardstick: project black-box forecasters onto the SMoE, out of sample.

For teacher T and asset a: at the start of each calendar year Y >= first teacher year + BURN,
fit an SMoE student on M draws from T's out-of-sample predictive distributions on all dates < Y
(warm-started from the previous year's student) and forecast year Y. Fidelity is measured on the
student's out-of-sample years only.

usage: python experiments/08_yardstick.py [dev|final]
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import ASSETS, FORECASTS, features, panel_path  # noqa: E402
from smoe import TABLES  # noqa: E402
from smoe.distributions import MixtureForecast  # noqa: E402
from smoe.evaluation import tests as T  # noqa: E402
from smoe.interpret.yardstick import distill, expected_kl, fidelity, probe  # noqa: E402
from smoe.models import SMoE  # noqa: E402
from smoe.walkforward import ForecastPanel  # noqa: E402

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
ALL_TEACHERS = ["Deep-SMoE", "Chronos-Bolt-small", "Chronos-Bolt-base", "TimesFM-2.5", "XGB-Gauss", "MS-AR",
                "GARCH-t"]
REFERENCE = "HistGauss"
TEACHERS = sys.argv[2:] or ALL_TEACHERS
BURN = 3
M = 10


def run(asset, teacher):
    pt = panel_path(period, asset, teacher)
    pr = panel_path(period, asset, REFERENCE)
    if not (pt.exists() and pr.exists()):
        return None
    P, R = ForecastPanel.from_npz(pt), ForecastPanel.from_npz(pr)
    fs = features(asset)
    X = fs.X
    common_idx = P.index.intersection(R.index).intersection(X.index)
    pos_t = pd.Index(P.index).get_indexer(common_idx)
    pos_r = pd.Index(R.index).get_indexer(common_idx)
    years = sorted(set(common_idx.year))
    eval_years = years[BURN:]
    if not eval_years:
        return None
    student, out = None, []
    for Y in eval_years:
        tr = common_idx < pd.Timestamp(f"{Y}-01-01")
        te = common_idx.year == Y
        if te.sum() == 0:
            continue
        fc_tr = P.fc.subset(pos_t[tr])
        student = distill(fc_tr, X.loc[common_idx[tr]], M=M, seed=Y,
                          student=SMoE(n_init=2 if student is None else 1), warm_start=student)
        fs_te = student.predict(X.loc[common_idx[te]])
        out.append((np.where(te)[0], fs_te))
    sel = np.concatenate([o[0] for o in out])
    stu = MixtureForecast(np.vstack([o[1].w for o in out]), np.vstack([o[1].mu for o in out]),
                          np.vstack([o[1].sd for o in out]))
    tea = P.fc.subset(pos_t[sel])
    ref = R.fc.subset(pos_r[sel])
    y = P.y[pos_t[sel]]
    fid = fidelity(y, tea, stu, ref)
    ag = T.weighted_lr_test(tea.logpdf(y), stu.logpdf(y), y)
    agt = T.weighted_lr_test(tea.logpdf(y), stu.logpdf(y), y, weight="tails")
    sub = np.linspace(0, len(y) - 1, min(300, len(y))).astype(int)
    kl = expected_kl(tea.subset(sub), stu.subset(sub), n_draws=200).mean()
    row = {"asset": asset, "teacher": teacher, "n_eval": len(y), "first_eval_year": eval_years[0], **fid,
           "AG_stat": ag["stat"], "AG_p": ag["p_two"], "AG_tails_stat": agt["stat"], "AG_tails_p": agt["p_two"],
           "KL": kl}
    # probes: does the deep gate track the SMoE state variables?
    if teacher == "Deep-SMoE":
        K = 3
        w = sum(tea.w[:, i * K:(i + 1) * K] for i in range(tea.w.shape[1] // K))
        gates = pd.DataFrame(w, columns=["trend", "reversal", "crisis"])
        Z = X.loc[common_idx[sel], ["g_lvol", "g_mom60", "g_dd"]].reset_index(drop=True)
        pr2 = probe(gates, Z)
        row.update({f"probeR2_{k}": v for k, v in pr2.items()})
    # save the student panel for later analysis
    ForecastPanel(common_idx[sel], y, stu, f"Shadow[{teacher}]", asset).to_npz(
        FORECASTS / period / f"{asset}__Shadow[{teacher}].npz")
    return row


if __name__ == "__main__":
    jobs = [(a, t) for a in ASSETS for t in TEACHERS]
    rows = [r for r in Parallel(n_jobs=14, verbose=5)(delayed(run)(a, t) for a, t in jobs) if r]
    df = pd.DataFrame(rows)
    out = TABLES / f"yardstick{'' if period == 'dev' else '_' + period}.csv"
    if out.exists():  # merge with teachers computed in earlier runs
        old = pd.read_csv(out)
        df = pd.concat([old[~old["teacher"].isin(TEACHERS)], df], ignore_index=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
