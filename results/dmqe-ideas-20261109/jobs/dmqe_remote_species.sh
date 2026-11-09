#!/usr/bin/env bash
set -eu
run=/mnt/homes/zwingel/fm-dmqe-20261109
cd "$run/src"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$run/tmp"
export TORCHINDUCTOR_CACHE_DIR="$run/cache/inductor-species"
export TRITON_CACHE_DIR="$run/cache/triton-species"
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
envpath=/mnt/homes/zwingel/environments/mace_cueq_latest
export LD_LIBRARY_PATH="$envpath/lib/python3.10/site-packages/nvidia/cu13/lib:$envpath/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$run/results/murphy-species"
python3 tools/ci/gpu_idle.py check --wait-minutes 0 --record "$run/results/murphy-species/idle.json"
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["idle"] else 3)' "$run/results/murphy-species/idle.json"
ps -eo user,pid,pcpu,etime,comm --sort=-pcpu | head -n 25
# All non-system jobs must be inactive; the long-lived sovic python session was sampled idle twice at setup.
"$envpath/bin/python" tools/dmqe_species_lab.py --output "$run/results/murphy-species/lab" \
  --counts 1000,1000,3000 --input-width 192 --channels 128 --sizes 1,3,5,7 --blocks 10 --rounds 100
printf 'DONE\n'
