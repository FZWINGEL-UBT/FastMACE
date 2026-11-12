#!/bin/bash
# usage: wsl.exe bash run_wsl.sh SCRIPT.py [args]  (CPU only, mace_cueq_oeq; runs from this directory)
export CUDA_VISIBLE_DEVICES=
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/mh1-catchup/results/mh1-catchup-20261112
export PYTHONPATH=/mnt/d/Margraf/FastMACE/.claude/worktrees/mh1-catchup
exec ~/environments/mace_cueq_oeq/bin/python "$@"
