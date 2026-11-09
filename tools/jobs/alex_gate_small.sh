#!/bin/bash -l
# OMAT-0 small on the A100 (9 November 2026): the 3x force/stress gate on the plan's defaults of SRC (small's decision
# 66 pair on the A100), then a packed A/B against the pair off in both orders (b = the per-model lists without the
# A100), and the step breakdown. Usage: sbatch --output=... gate-small.sh <SRC dir>
#SBATCH --job-name=fm-smallgate
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:30:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
B=/home/atuin/v130cb/v130cb18/fm-omat0small
src=${1:?SRC}
out=$B/runs/gate-small-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
module purge >/dev/null 2>&1
site=$(echo $B/env/lib/python3.*/site-packages)
export PATH=$B/env/bin:/usr/bin:/bin VIRTUAL_ENV=$B/env PYTHONPATH=$src PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$B/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd $src || exit 4
note "gate-small job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit)"
ck=$FASTMACE_MODELS/mace-omat-0-small.model
st=results/loup-a100-20261103/SrTiO3.cif
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-small --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl NbCl5 > $out/gate.log 2>&1 < /dev/null
note "gate exit=$?"
grep -hE "summary" $out/gate.log | tail -n 3 | sed 's/^/  /' | tee -a "$log"
python tools/step_breakdown.py --output $out/breakdown --checkpoint $ck --model omat0-small \
  --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200 \
  > $out/breakdown.log 2>&1 < /dev/null
note "breakdown exit=$? $(grep -hoE '"(wall_ms_per_step|force_graph_ms_per_replay)": [0-9.]+' $out/breakdown.log | tr '\n' ' ')"
rm -f $out/breakdown/trace.json
note done
