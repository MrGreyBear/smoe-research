"""Deep SMoE: the black-box counterpart (heterogeneous neural experts, attention gate).

Same output object as the white-box SMoE — a K = 3 Gaussian mixture per day — so the
two are directly comparable and the white box can be fitted to the deep model.

Experts (as in the undergraduate design, now probabilistic):
  trend    : LSTM over the trend-feature window
  reversal : MLP over the last 5 days of reversal features
  crisis   : 1D-CNN over the variance-feature window
Each expert outputs (mu_k, log sigma_k^2); variance heads also see the current variance features.
Gate: LSTM over the gate-feature window + additive temporal attention -> softmax.
Loss: mixture NLL + lambda_ent * (negative entropy of the batch-average gate) to avoid collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from smoe.distributions import MixtureForecast
from smoe.features.semantic import SEMANTIC_GROUPS

torch.set_num_threads(4)


def _windows(A: np.ndarray, L: int) -> np.ndarray:
    """(n, d) -> (n, L, d) with rows padded at the start by repeating the first row."""
    pad = np.repeat(A[:1], L - 1, axis=0)
    B = np.vstack([pad, A])
    idx = np.arange(L)[None, :] + np.arange(len(A))[:, None]
    return B[idx]


class _Net(nn.Module):
    def __init__(self, dims: dict, L: int, hidden: int = 16):
        super().__init__()
        dv = dims["var"]
        self.trend = nn.LSTM(dims["trend"], hidden, batch_first=True)
        self.trend_head = nn.Linear(hidden + dv, 2)
        self.rev = nn.Sequential(nn.Flatten(), nn.Linear(5 * dims["reversal"], hidden), nn.ReLU())
        self.rev_head = nn.Linear(hidden + dv, 2)
        self.cnn = nn.Sequential(nn.Conv1d(dv, hidden, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.cnn_head = nn.Linear(hidden + dv, 2)
        self.gate_rnn = nn.LSTM(dims["gate"], hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.gate_out = nn.Linear(hidden, 3)
        self.drop = nn.Dropout(0.1)

    def forward(self, xt, xr, xv, xg):
        v_now = xv[:, -1, :]
        h_t = self.trend(xt)[0][:, -1, :]
        o1 = self.trend_head(torch.cat([self.drop(h_t), v_now], 1))
        h_r = self.rev(xr[:, -5:, :])
        o2 = self.rev_head(torch.cat([self.drop(h_r), v_now], 1))
        h_v = self.cnn(xv.transpose(1, 2))
        o3 = self.cnn_head(torch.cat([self.drop(h_v), v_now], 1))
        H = self.gate_rnn(xg)[0]
        a = torch.softmax(self.attn(H).squeeze(-1), dim=1)
        ctx = (a.unsqueeze(-1) * H).sum(1)
        logits = self.gate_out(self.drop(ctx))
        mu = torch.stack([o1[:, 0], o2[:, 0], o3[:, 0]], 1)
        lv = torch.stack([o1[:, 1], o2[:, 1], o3[:, 1]], 1).clamp(-8, 6)
        return torch.log_softmax(logits, 1), mu, lv


def _mixture_nll(lpi, mu, lv, y):
    lc = lpi - 0.5 * (np.log(2 * np.pi) + lv + (y[:, None] - mu) ** 2 / lv.exp())
    return -torch.logsumexp(lc, 1).mean()


@dataclass
class DeepSMoE:
    L: int = 20
    hidden: int = 16
    epochs: int = 150
    patience: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam_ent: float = 0.01
    batch: int = 128
    seeds: tuple = (0, 1, 2)
    name: str = "Deep-SMoE"
    groups: dict = field(default_factory=lambda: {k: SEMANTIC_GROUPS[k] for k in ("trend", "reversal", "var", "gate")})

    def _tensors(self, X: pd.DataFrame):
        out = []
        for g in ("trend", "reversal", "var", "gate"):
            A = ((X[self.groups[g]] - self.loc_[self.groups[g]]) / self.scale_[self.groups[g]]).to_numpy(np.float32)
            out.append(torch.from_numpy(_windows(A, self.L)))
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ok = y.notna().to_numpy()
        cols = sum(self.groups.values(), [])
        self.loc_, self.scale_ = X[cols].mean(), X[cols].std().replace(0, 1)
        self.ys_ = float(y[ok].std())
        T = self._tensors(X)  # windows built on the full history, then rows selected
        Tn = [t[ok] for t in T]
        yt = torch.from_numpy((y[ok].to_numpy(np.float32)) / self.ys_)
        n = len(yt)
        n_val = max(int(0.15 * n), 100)
        tr, va = slice(0, n - n_val), slice(n - n_val, n)
        dims = {g: len(c) for g, c in self.groups.items()}
        self.nets_ = []
        for s in self.seeds:
            torch.manual_seed(s)
            net = _Net(dims, self.L, self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_state, bad = np.inf, None, 0
            g = torch.Generator().manual_seed(s)
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(n - n_val, generator=g)
                for i in range(0, len(perm), self.batch):
                    b = perm[i:i + self.batch]
                    lpi, mu, lv = net(*[t[tr][b] for t in Tn])
                    pbar = lpi.exp().mean(0)
                    loss = _mixture_nll(lpi, mu, lv, yt[tr][b]) + self.lam_ent * torch.sum(pbar * pbar.log())
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = _mixture_nll(*net(*[t[va] for t in Tn]), yt[va]).item()
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
        return self

    def predict(self, X: pd.DataFrame) -> MixtureForecast:
        T = self._tensors(X)
        ws, mus, sds = [], [], []
        with torch.no_grad():
            for net in self.nets_:
                lpi, mu, lv = net(*T)
                ws.append(lpi.exp().numpy() / len(self.nets_))
                mus.append(mu.numpy() * self.ys_)
                sds.append(np.exp(0.5 * lv.numpy()) * self.ys_)
        return MixtureForecast(np.hstack(ws), np.hstack(mus), np.hstack(sds))

    def gate_probs(self, X: pd.DataFrame) -> pd.DataFrame:
        f = self.predict(X)
        K = 3
        w = sum(f.w[:, i * K:(i + 1) * K] for i in range(len(self.nets_)))
        return pd.DataFrame(w, index=X.index, columns=["trend", "reversal", "crisis"])
