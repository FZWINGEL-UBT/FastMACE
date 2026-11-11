#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$HOME/environments/fastmace_cpu/bin/python" /mnt/d/Margraf/inference-ideas-20261111/gauge_probe.py
exec "$HOME/environments/fastmace_cpu/bin/python" /mnt/d/Margraf/inference-ideas-20261111/decode_pairs_probe.py
