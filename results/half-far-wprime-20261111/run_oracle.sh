#!/bin/bash
# CPU oracle runs (WSL fastmace_cpu): SrTiO3 4x4x4 and NbCl5 2x, as the 11 Nov screen.
cd /mnt/d/Margraf/fm-a100-colab/results/half-far-wprime-20261111
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
PY=~/environments/fastmace_cpu/bin/python
$PY half_wprime_oracle.py "$1" "$2" "$3" $4 $5 > "${3%.json}.log" 2>&1
echo "exit $?" >> "${3%.json}.log"
