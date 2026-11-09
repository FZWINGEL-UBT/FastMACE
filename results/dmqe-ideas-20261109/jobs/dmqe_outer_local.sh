#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
out=results/dmqe-ideas-20261109/local-outer-final
mkdir -p "$out"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
python3 tools/ci/local_gpu.py run --purpose dmqe-outer-registers --wait-minutes 30 -- \
 "$HOME/environments/mace_cueq_latest/bin/python" tools/dmqe_outer_lab.py \
 --output "$out/lab" --jitter 0.07 --rounds 31 > "$out/run.log" 2>&1
