"""Simulate from an SMoE data-generating process (used by tests and the Monte Carlo study)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe.features.semantic import SEMANTIC_GROUPS

TRUE_PARAMS = {
    # expert order: trend, reversal, crisis
    "beta": [np.array([0.05, 0.30, 0.15]), np.array([0.02, -0.35, -0.10]), np.array([-0.40])],
    "delta": [np.array([-0.2, 0.3, 0.1, 0.2]), np.array([-0.4, 0.2, 0.1, 0.1]), np.array([0.9, 0.4, 0.0, 0.3])],
    # gate logits relative to crisis (reference); columns: const, g_lvol, g_mom60, g_dd
    "gamma": np.array([[1.0, -0.8, 0.6, 0.3], [0.8, -0.6, -0.5, 0.2]]),
}


def simulate_smoe(n: int, seed: int = 0, params: dict | None = None) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Draw iid standard-normal covariates and y from the SMoE; returns (X, y, true_component)."""
    p = params or TRUE_PARAMS
    rng = np.random.default_rng(seed)
    cols = SEMANTIC_GROUPS["trend"] + SEMANTIC_GROUPS["reversal"] + SEMANTIC_GROUPS["var"] + SEMANTIC_GROUPS["gate"]
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols,
                     index=pd.RangeIndex(n))
    one = np.ones((n, 1))
    Xk = [np.hstack([one, X[SEMANTIC_GROUPS["trend"]].to_numpy()]),
          np.hstack([one, X[SEMANTIC_GROUPS["reversal"]].to_numpy()]),
          one]
    V = np.hstack([one, X[SEMANTIC_GROUPS["var"]].to_numpy()])
    Z = np.hstack([one, X[SEMANTIC_GROUPS["gate"]].to_numpy()])
    logits = np.hstack([Z @ p["gamma"].T, np.zeros((n, 1))])
    pi = np.exp(logits - logits.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    comp = (rng.random(n)[:, None] > np.cumsum(pi, 1)).sum(1)
    mu = np.column_stack([Xk[k] @ p["beta"][k] for k in range(3)])
    sd = np.exp(0.5 * np.column_stack([V @ p["delta"][k] for k in range(3)]))
    y = mu[np.arange(n), comp] + sd[np.arange(n), comp] * rng.standard_normal(n)
    return X, pd.Series(y, index=X.index, name="y"), comp
