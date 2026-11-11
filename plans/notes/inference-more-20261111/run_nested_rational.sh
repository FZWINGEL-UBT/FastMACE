#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$HOME/environments/fastmace_cpu/bin/python" /mnt/d/Margraf/inference-more-20261111/nested_radial_basis.py
exec "$HOME/environments/fastmace_cpu/bin/python" /mnt/d/Margraf/inference-more-20261111/shared_denominator.py
