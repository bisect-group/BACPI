#!/usr/bin/env bash
# Checks that the BACPI wrapper and its speed-ups match the released code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, rdkit, pandas, pytest). The script does not create envs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export EMULATOR_SYNTHETIC_FIXTURE=1
files=(tests/test_official_equivalence.py tests/test_compile_and_fused_adam.py tests/test_cache_build.py tests/test_train_loop.py)
if [[ ${1:-} == --negative-control ]]; then
    # Scales the candidate weights by 1.05: every gpu test must fail. A passing gpu test has a vacuous bound.
    export EQUIV_NEGATIVE_CONTROL=1
    status=0
    out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no -m gpu -s 2>&1) || status=$?
    echo "$out"
    if [[ $status -ne 1 ]] || grep -Eq "[0-9]+ (passed|skipped|errors?)" <<<"$out"; then
        echo "FAIL: a gpu test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every gpu test failed under the negative control"
    exit 0
fi
marker=(-m "not gpu")
allow_skip=
if [[ ${1:-} == --gpu ]]; then
    marker=(-s)
    if ! python -c "import torch,sys; sys.exit(not torch.cuda.is_available())"; then
        echo "No CUDA device visible: gpu tests will be skipped."
        allow_skip=1
    fi
else
    export CUDA_VISIBLE_DEVICES=""
fi
out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no "${marker[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if [[ -z ${allow_skip:-} ]] && grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
