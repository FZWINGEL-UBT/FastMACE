#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
out=results/dmqe-ideas-20261109/local-species
mkdir -p "$out"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
python3 tools/ci/local_gpu.py run --purpose dmqe-species-plan --wait-minutes 30 -- \
  "$HOME/environments/mace_cueq_latest/bin/python" tools/dmqe_species_lab.py \
  --output "$out/lab" --counts 1000,1000,3000 --input-width 192 --channels 128 \
  --sizes 1,3,5,7 --blocks 10 --rounds 100 > "$out/run.log" 2>&1
