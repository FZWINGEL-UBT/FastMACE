#!/bin/bash
# 1b-T17 GPU smoke test on the local RTX 5060 Ti (WSL), detached; logs in ~/runs/1b-T17/.
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t17-8b0e5a
R=$PWD/results/1b-T17-rtx5060ti-20261019
mkdir -p $R ~/runs/1b-T17
nvidia-smi > $R/idle_check.txt
free -g > $R/free.txt
git rev-parse HEAD > $R/commit.txt
TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_RESULTS=$R FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python FASTMACE_MODELS=$HOME/models PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $HOME/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -m "" -v -s tests/gpu/test_simulation_gpu.py > ~/runs/1b-T17/smoke.log 2>&1
echo "exit $?" >> ~/runs/1b-T17/smoke.log
echo DONE > ~/runs/1b-T17/done
