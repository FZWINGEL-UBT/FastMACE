#!/usr/bin/env bash
set -eu
run=/mnt/homes/zwingel/fm-dmqe-20261109
cd "$run/src"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$run/tmp"
export TORCHINDUCTOR_CACHE_DIR="$run/cache/inductor-outer"
export TRITON_CACHE_DIR="$run/cache/triton-outer"
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=2
envpath=/mnt/homes/zwingel/environments/mace_cueq_latest
export LD_LIBRARY_PATH="$envpath/lib/python3.10/site-packages/nvidia/cu13/lib:$envpath/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}"
for jitter in 0 0.07; do
 python3 tools/local/dmqe_workstation_guard.py --record "$run/results/leto-outer-final-$jitter-idle.json" -- \
  "$envpath/bin/python" tools/dmqe_outer_lab.py --output "$run/results/leto-outer-final-$jitter" \
  --jitter "$jitter" --rounds 31
done
