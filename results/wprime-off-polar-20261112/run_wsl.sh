#!/bin/bash
# usage: wsl.exe bash run_wsl.sh SCRIPT.py [args]  (CPU only, mace_cueq_oeq; run from this directory)
export CUDA_VISIBLE_DEVICES=
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/wprime-off-polar/results/wprime-off-polar-20261112
export PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/wprime-off-polar
exec ~/environments/mace_cueq_oeq/bin/python "$@"
