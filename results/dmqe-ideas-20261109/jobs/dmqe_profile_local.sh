#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
out=results/dmqe-ideas-20261109/local-profile
mkdir -p "$out"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export MACE_COMPILE_CACHE="$HOME/compile-cache/mace_cueq_latest"
python3 tools/ci/local_gpu.py run --purpose dmqe-current-profile --wait-minutes 0 -- \
  "$HOME/environments/mace_cueq_latest/bin/python" tools/step_breakdown.py \
  --output "$out/breakdown" --checkpoint "$HOME/models/mace-omat-0-medium.model" --model omat0-medium \
  --structure results/loup-a100-20261103/SrTiO3.cif@6x6x6 --backend cueq \
  --equilibrate-steps 20 --steps 30 --calls 2 > "$out/run.log" 2>&1
