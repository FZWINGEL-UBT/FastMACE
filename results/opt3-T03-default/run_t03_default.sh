#!/bin/bash
# OPT3-T03-default local GPU checks, each in its own short local_gpu.py reservation:
# the force gate on NaCl/NbCl5 with the new defaults, tests/gpu/test_simulation_gpu.py, and the
# reproducible-mode NVT group of tests/gpu/test_fused_integrator_gpu.py (default against off, bitwise).
set -u
W=$HOME/opt3-t03-default; mkdir -p $W/tmp
SRC=$W/src  # pinned copy of the branch
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export TMPDIR=$W/tmp TORCHINDUCTOR_COMPILE_THREADS=4 MACE_COMPILE_CACHE=$HOME/compile-cache/opt3-t03 TORCHINDUCTOR_CACHE_DIR=$HOME/compile-cache/opt3-t03-inductor
export FASTMACE_RESULTS=$W/gpu-results FASTMACE_HYBRID_PYTHON=$HOME/environments/mace_cueq_oeq/bin/python
P=$HOME/environments/mace_cueq_oeq/bin/python
PT=$HOME/environments/mace_cueq_latest/bin/python
GPU="python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --wait-minutes 240 --min-available-gib 6"
log() { echo "$(date -u +%FT%TZ) $*" >> $W/run.log; }
log "source $(cat $SRC/.source_commit) gpu $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader)"
LAUNCH="bash $HOME/fastmace-baseline/1a-T18/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh $HOME/environments/mace_cueq_oeq $HOME/fastmace-cache/opt3-t03-default/oeq-launcher $P"
rm -rf $W/gate
$GPU --purpose "OPT3-T03-default force gate NaCl NbCl5 (<10 min)" -- nice $P -W ignore $SRC/tools/force_stress_gate.py \
  --output $W/gate --engine fastmace --structures-root $HOME/fastmace-baseline/1a-T18 \
  --checkpoint $HOME/models/mace-mh-1.model --structures NaCl NbCl5 \
  --reference-command $PT --production-command "$LAUNCH" > $W/gate.log 2>&1
log "gate exit=$?"
$GPU --purpose "OPT3-T03-default test_simulation_gpu (<15 min)" -- bash -c \
  "cd $SRC && nice $PT -m pytest -q -p no:cacheprovider tests/gpu/test_simulation_gpu.py --basetemp=$W/tmp/sim -rfs" \
  > $W/pytest-simulation.log 2>&1
log "test_simulation_gpu exit=$?"
$GPU --purpose "OPT3-T03-default reproducible default vs off (<10 min)" -- bash -c \
  "cd $SRC && nice $PT -m pytest -q -p no:cacheprovider tests/gpu/test_fused_integrator_gpu.py -k 'trajectories and nvt' --basetemp=$W/tmp/fi -rfs" \
  > $W/pytest-fused-nvt.log 2>&1
log "fused nvt exit=$?"
log "all done"
