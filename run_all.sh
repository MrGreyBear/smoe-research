#!/usr/bin/env bash
# Full replication pipeline. Forecasts are cached in results/forecasts/<period>/ and only
# recomputed when missing. Usage: bash run_all.sh [dev|final]
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python
PERIOD=${1:-dev}
mkdir -p results/logs

$PY experiments/00_download_data.py            # only needed for a fresh checkout or the final period
$PY experiments/01_descriptives.py
$PY experiments/02_lagging_paradox.py
$PY experiments/03_monte_carlo.py 200 12
$PY experiments/05_run_forecasts.py "$PERIOD" 14
$PY experiments/05b_run_tsfm.py "$PERIOD" small
$PY experiments/05b_run_tsfm.py "$PERIOD" base
$PY experiments/04_insample.py
$PY experiments/06_evaluate.py "$PERIOD"
$PY experiments/07_ablation.py 10
$PY experiments/08_yardstick.py "$PERIOD"
$PY experiments/10_conformal.py "$PERIOD"
$PY experiments/11_economic.py "$PERIOD"
$PY experiments/20_make_tables.py
$PY experiments/21_figures.py
(cd paper && ../tools/tectonic.exe -X compile main.tex)
