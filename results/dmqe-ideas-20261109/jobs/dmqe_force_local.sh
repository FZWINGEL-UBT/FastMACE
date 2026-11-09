#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
out=results/dmqe-ideas-20261109/local-force
mkdir -p "$out"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
python3 tools/ci/local_gpu.py run --purpose dmqe-force-static-plan --wait-minutes 30 -- \
 "$HOME/environments/mace_cueq_oeq/bin/python" tools/dmqe_species_force_lab.py \
 --checkpoint "$HOME/models/mace-omat-0-medium.model" \
 --structure results/loup-a100-20261103/SrTiO3.cif@6x6x6 \
 --output "$out/lab" --blocks 10 --rounds 50 > "$out/run.log" 2>&1
