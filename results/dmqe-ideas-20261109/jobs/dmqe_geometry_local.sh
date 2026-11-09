#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
out=results/dmqe-ideas-20261109/local-geometry-final
mkdir -p "$out"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
python3 tools/ci/local_gpu.py run --purpose dmqe-geometry --wait-minutes 30 -- \
 "$HOME/environments/mace_cueq_latest/bin/python" tools/dmqe_geometry_lab.py \
 --device cuda --cells 10 --rounds 31 --upstream-scale 0.37 --output "$out/lab" > "$out/run.log" 2>&1
