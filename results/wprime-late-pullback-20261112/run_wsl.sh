#!/bin/bash
# usage: wsl.exe bash run_wsl.sh SCRIPT.py [args]  (CPU only, mace_cueq_oeq; its activate script points elsewhere)
export CUDA_VISIBLE_DEVICES=
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/late-pullback/results/wprime-late-pullback-20261112
exec ~/environments/mace_cueq_oeq/bin/python "$@"
