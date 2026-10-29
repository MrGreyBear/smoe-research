# SMoE — Semantic Mixture of Experts

White-box distributional forecasting for financial returns, a yardstick for black-box forecasters,
and a cost-aware trading layer. Graduate-level rebuild of an undergraduate FYP (see `docs/AUDIT.md`).

## Layout

| Path | Content |
|---|---|
| `src/smoe/` | package: data loaders, causal features, mixture closed forms, SMoE (ECM + sandwich inference), benchmarks, deep SMoE, Chronos wrapper, walk-forward runner, evaluation tests, conformal layer, yardstick, backtest |
| `tests/` | pytest suite (leakage, EM monotonicity, parameter recovery, closed forms vs Monte Carlo, test-statistic size/power) |
| `experiments/` | numbered scripts producing every table and figure in the paper |
| `results/` | forecasts (cached), tables (CSV + LaTeX), figures (PDF), logs |
| `paper/` | LaTeX working paper (`main.tex`, `sections/`, `refs.bib`) |
| `trading/live_signal.py` | daily signal for CSI 300 ETF / IF futures after the close |
| `docs/` | audit of the undergraduate code (`AUDIT.md`) |
| `archive/` | undergraduate material (separate repository: [undergrad-fyp-csi300-moe](https://github.com/MrGreyBear/undergrad-fyp-csi300-moe)) |

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e . websockets>=13 chronos-forecasting
.venv/Scripts/python -m pytest
bash run_all.sh dev        # development period 2012 - 2024-06
bash run_all.sh lockbox    # held-out period 2024-07 onwards
```

Forecasts are cached in `results/forecasts/<period>/`; delete a file to recompute it. HuggingFace
models are fetched through `HF_ENDPOINT=https://hf-mirror.com` by default.

## Rules

- No look-ahead: every model is tested by corrupting future prices and checking forecasts are unchanged.
- No number in the paper is typed by hand: tables are generated from CSVs by `experiments/20_make_tables.py`.
