#!/usr/bin/env bash
set -eu
run=/mnt/homes/zwingel/fm-dmqe-20261109
cd "$run/src"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$run/tmp"
export TORCHINDUCTOR_CACHE_DIR="$run/cache/inductor-force"
export TRITON_CACHE_DIR="$run/cache/triton-force"
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
envpath=/mnt/homes/zwingel/environments/mace_cueq_oeq
export LD_LIBRARY_PATH="$envpath/lib/python3.10/site-packages/nvidia/cu13/lib:$envpath/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}"
python3 tools/local/dmqe_workstation_guard.py --record "$run/results/leto-force-idle.json" -- \
 "$envpath/bin/python" tools/dmqe_species_force_lab.py \
 --checkpoint /mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model \
 --structure results/loup-a100-20261103/SrTiO3.cif@6x6x6 \
 --output "$run/results/leto-force" --blocks 10 --rounds 50 --stress
