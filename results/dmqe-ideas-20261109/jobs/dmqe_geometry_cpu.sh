#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
"$HOME/environments/fastmace_cpu/bin/python" tools/dmqe_geometry_lab.py --device cpu --cells 2 \
 --rounds 2 --warmup 1 --no-compile --upstream-scale 0.37 \
 --output results/dmqe-ideas-20261109/cpu-geometry > results/dmqe-ideas-20261109/cpu-geometry.log 2>&1
