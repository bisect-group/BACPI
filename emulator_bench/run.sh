#!/usr/bin/env bash
# BACPI Ki, Kd, EC50, IC50 paper runs: fp32, torch.compile. train also evaluates (test metrics).
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
# shellcheck disable=SC2086  # PY, GPUS, SEEDS and the task lists split into words on purpose
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/BACPI}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-"666 777 888"}
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache|train|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

# IC50 uses its own cache dir, as in the paper runs.
IC50_EMB=$OUT_ROOT/embeddings_ic50

want cache && {
  run $PY emulator_bench/cache_embeddings.py --tasks ki kd ec50 --datasets_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" --embeddings_dir "$OUT_ROOT/embeddings"
  run $PY emulator_bench/cache_embeddings.py --tasks ic50 --datasets_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" --embeddings_dir "$IC50_EMB"
}
want train && {
  run $PY emulator_bench/run_split_benchmarks.py --tasks ki kd ec50 --seeds $SEEDS \
    --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" --num_epochs 20 --datasets_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" --embeddings_dir "$OUT_ROOT/embeddings"
  run $PY emulator_bench/run_split_benchmarks.py --tasks ic50 --seeds $SEEDS \
    --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" --num_epochs 20 --datasets_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" --embeddings_dir "$IC50_EMB"
}
want aggregate && run $PY emulator_bench/aggregate_retrain_metrics.py --tasks ki kd ec50 ic50 --out_root "$OUT_ROOT"
exit 0
