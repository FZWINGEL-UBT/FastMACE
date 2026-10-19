#!/bin/bash
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/1b-t14-d20c78
R=$PWD/results/1b-T14-rtx5060ti-20261019
mkdir -p $R
(while true; do echo "$(date -u +%H:%M:%S) $(free -m | awk "/Mem:/{print \$7}")" >> ~/runs/1b-T14/free.log; sleep 30; done) &
MON=$!
nvidia-smi > $R/idle_check.txt
git rev-parse HEAD > $R/commit.txt
TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_RESULTS=$R FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python FASTMACE_MODELS=$HOME/models PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $HOME/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -m "" -v -s tests/gpu/test_canonical_restart_gpu.py -k plain_cueq > ~/runs/1b-T14/plain.log 2>&1
TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_RESULTS=$R FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python FASTMACE_MODELS=$HOME/models PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 $HOME/environments/mace_cueq_latest/bin/python -m pytest -p no:cacheprovider -m "" -v -s tests/gpu/test_canonical_restart_gpu.py -k hybrid > ~/runs/1b-T14/hybrid.log 2>&1
kill $MON
echo DONE > ~/runs/1b-T14/done
