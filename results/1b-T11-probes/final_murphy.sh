#!/usr/bin/env bash
# 1b-T11 after the fix, on murphy, one job at a time (start detached: setsid nohup bash final_murphy.sh SRC &):
#   1. MP-0 medium same-process A/B, tools/ab_step_time.py, 12 blocks ABAB, NaCl and NbCl5, baseline built first;
#   2. the same with FastMACE built first;
#   3. the CPU suite in the NAS mace_cueq_latest (CUDA hidden);
#   4. the whole GPU suite (mace_cueq_latest, hybrid children in mace_cueq_oeq).
# Each GPU job starts only when murphy has no compute process from anyone. Log: $work/runs/final.log
set -uo pipefail
src=$1
work=/mnt/homes/zwingel/fastmace-work/1b-T11
here=$src/results/1b-T11-probes
log=$work/runs/final.log
B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
CK=/mnt/homes/zwingel/ResearchModule/Models/2023-12-03-mace-128-L1_epoch-199.model
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  for pid in $pids; do note "busy: pid $pid user $(ps -o user= -p "$pid")"; done
  [ -z "$pids" ]
}
for order in "baseline fastmace" "fastmace baseline"; do
  name=ab-mp0-medium-${order%% *}-first
  idle || { note "GPU busy before $name; stopping"; exit 3; }
  note "start $name"
  SRC=$src FOREGROUND=1 bash "$here/run_murphy.sh" "$name" hybrid-launcher "$src/tools/ab_step_time.py" \
    --output "$work/runs/$name/out" --baseline $B --model mp0-medium --checkpoint $CK --structures NaCl NbCl5 \
    --sides $order --warmup 300 --block 300 --blocks 12 --pattern abab --ensemble nvt --backend hybrid --dump-forces
  note "done $name exit $?"
done
E=/mnt/homes/zwingel/environments/mace_cueq_latest
SP=$E/lib/python3.10/site-packages
cache=/mnt/homes/zwingel/fastmace-cache/1b-T11-suite
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/cuequivariance_ops/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$src
export FASTMACE_CACHE=$cache/fastmace XDG_CACHE_HOME=$cache/xdg TRITON_CACHE_DIR=$cache/triton
export TORCHINDUCTOR_CACHE_DIR=$cache/inductor MPLCONFIGDIR=$cache/matplotlib
export TMPDIR=$cache/tmp TMP=$cache/tmp TEMP=$cache/tmp
export FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models
export FASTMACE_HYBRID_PYTHON=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python
export FASTMACE_BASELINE=$B FASTMACE_RESULTS=$work/runs/suite-results
mkdir -p "$TMPDIR" "$FASTMACE_RESULTS"
cd "$src" || exit 2
note "start CPU suite"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 $E/bin/python -m pytest -n 8 -p no:cacheprovider -q \
  --basetemp=$cache/pytest-cpu tests/cpu > "$work/runs/suite-cpu.log" 2>&1
note "done CPU suite exit $?"
idle || { note "GPU busy before the GPU suite; stopping"; exit 3; }
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader >> "$log"
note "start GPU suite"
$E/bin/python -m pytest -p no:cacheprovider -q -rs --basetemp=$cache/pytest-gpu tests/gpu \
  > "$work/runs/suite-gpu.log" 2>&1
note "done GPU suite exit $?"
note "final complete"
