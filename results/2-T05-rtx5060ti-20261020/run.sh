#!/bin/bash
# 2-T05 acceptance 3 on the local RTX 5060 Ti: launched with setsid -f, under the local GPU reservation
# (tools/ci/local_gpu.py, 2-T18). Every 5 s it logs free -m, the GPU's memory and the reference
# children's resident memory, and it kills this process group if available memory falls below 2 GiB
# (attempt-3 took down WSL on 20 October 2026). Usage: run.sh <attempt directory name>.
cd /mnt/d/Margraf/FastMACE/.claude/worktrees/2-t05-b56d7a
OUT=results/2-T05-rtx5060ti-20261020/${1:-attempt}
mkdir -p $OUT
GROUP=$(ps -o pgid= -p $$ | tr -d ' ')
(
  while true; do
    available=$(free -m | awk '/^Mem:/{print $7}')
    child=$(ps -eo rss,args | grep '[f]astmace.planner.selfcheck' | awk '{s+=$1} END {printf "%.0f", s/1024}')
    echo "$(date +%T) available_MiB=$available child_rss_MiB=${child:-0} gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    if [ "$available" -lt 2048 ]; then
      echo "$(date +%T) WATCHDOG: available below 2 GiB, killing this group's pytest and reference children"
      # Python processes of this group except local_gpu.py itself, which then releases the reservation.
      ps -eo pid=,pgid=,args= | awk -v g=$GROUP '$2 == g && /python/ && !/local_gpu.py/ {print $1}' | xargs -r kill -KILL
      break
    fi
    sleep 5
  done
) > $OUT/free.log 2>&1 &
LOGGER=$!
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "2-T05 OOM cases rerun" -- \
  env TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python \
  FASTMACE_MODELS=$HOME/models PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 \
  timeout 7200 $HOME/environments/mace_cueq_latest/bin/python -m pytest -s -v -p no:cacheprovider \
  --junitxml=$OUT/junit.xml tests/gpu/test_gate_resources_gpu.py > $OUT/pytest.log 2>&1
echo "exit $?" >> $OUT/pytest.log
kill $LOGGER
echo done > $OUT/DONE
