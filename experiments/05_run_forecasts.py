"""Run every (asset, model) walk-forward in parallel and cache forecasts.

usage: python experiments/05_run_forecasts.py [dev|final] [n_jobs] [model ...]
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ASSETS, MODELS, run_one  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402


def safe_run(a, m, period):
    try:
        return run_one(a, m, period)
    except Exception as e:  # one failing (asset, model) must not abort the pool
        return f"FAILED {a} {m}: {type(e).__name__}: {e}"

if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    models = sys.argv[3:] or list(MODELS)
    # slow models first so the pool stays busy
    order = sorted(models, key=lambda m: {"Deep-SMoE": 0, "SMoE": 1, "MS-AR": 2}.get(m, 9))
    jobs = [(a, m) for m in order for a in ASSETS]
    for msg in Parallel(n_jobs=n_jobs, verbose=0, return_as="generator_unordered")(
            delayed(safe_run)(a, m, period) for a, m in jobs):
        print(msg, flush=True)
