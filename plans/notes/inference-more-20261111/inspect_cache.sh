#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
free -m
exec "$HOME/environments/fastmace_cpu/bin/python" /mnt/d/Margraf/inference-more-20261111/inspect_cache.py
