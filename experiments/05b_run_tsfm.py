"""Zero-shot foundation-model forecasts (Chronos-Bolt, TimesFM 2.5) for the out-of-sample rows of every asset.

No refitting is needed (zero-shot); each forecast uses only r_{<=t} as context.
usage: python experiments/05b_run_tsfm.py [dev|lockbox] [small|base|timesfm]
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401  (load torch's OpenMP before numpy/MKL)
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ASSETS, DEV_END, OOS_START, features, panel_path  # noqa: E402
from smoe import LOCKBOX_START  # noqa: E402
from smoe.models.tsfm import ChronosBolt, TimesFM  # noqa: E402
from smoe.walkforward import ForecastPanel  # noqa: E402

if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    size = sys.argv[2] if len(sys.argv) > 2 else "small"
    model = TimesFM() if size == "timesfm" else ChronosBolt(size=size)
    for asset in ASSETS:
        path = panel_path(period, asset, model.name)
        if path.exists():
            continue
        fs = features(asset)
        X = fs.X.assign(r=fs.r)
        end = pd.Timestamp(DEV_END) if period == "dev" else X.index[-1]
        # match the statistical models: first OOS row is the first date with >= 750 training rows
        first = max(pd.Timestamp(OOS_START), X.index[min(750, len(X) - 1)])
        if period == "lockbox":
            first = max(first, pd.Timestamp(LOCKBOX_START))
        mask = (X.index >= first) & (X.index <= end) & fs.y.notna().to_numpy()
        rows = np.where(mask)[0]
        t0 = time.time()
        fc = model.predict(X, rows)
        ForecastPanel(X.index[rows], fs.y.to_numpy()[rows], fc, model.name, asset, time.time() - t0).to_npz(path)
        print(asset, len(rows), f"{time.time() - t0:.0f}s", flush=True)
