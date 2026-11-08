#!/bin/bash -l
# Force/stress gate on one Alex A100 (8 November 2026): GATE=default (the A100 plan) or GATE=all (the ten H200 switches),
# or GATE=allpb (all + POWER_BACKWARD). With GATE != default it first runs tools/spline_w_check.py (spline and compose in
# use, forces against the default build on a rattled SrTiO3 10x10x10).
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=05:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -uo pipefail
here=/home/atuin/v130cb/v130cb18/fm-a100omat/jobs
GATE=${GATE:-${SLURM_JOB_NAME#fm-a100g-}}
source $here/common.sh
out=$B/runs/gate-$GATE-$SLURM_JOB_ID; mkdir -p "$out"; log=$out/run.log
note() { echo "$(date +%FT%T%z) $*" | tee -a "$log"; }
module purge >/dev/null 2>&1
t0=$(date +%s)  # env on $WORK, not staged: concurrent staged jobs sharing one cache break the OEq stand-in CUDA home
site=$(echo $B/env/lib/python3.*/site-packages)
export PATH=$B/env/bin:/usr/bin:/bin VIRTUAL_ENV=$B/env PYTHONPATH=$B/src-1bd4f3be PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$site/nvidia/cu13/lib:$site/cuequivariance_ops/lib
export FASTMACE_CACHE=$B/cache FASTMACE_MODELS=$B/models FASTMACE_STRUCTURES=$B/baseline
n=$(nproc); export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n TORCHINDUCTOR_COMPILE_THREADS=$n
cd $B/src-1bd4f3be
note "gate $GATE job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit); set up in $(( $(date +%s) - t0 )) s"
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
case $GATE in default) set -- ;; all) set -- $ALL ;; allpb) set -- $ALL $p:POWER_BACKWARD=True ;; *) note "GATE?"; exit 2 ;; esac
if [ $# -gt 0 ]; then
  python tools/spline_w_check.py --checkpoint $ck --structure "$st@10x10x10" --output $out/check.json --constant "$@" \
    --require-compose --rattle 0.05 --compare-constant $s:SPLINE_W=False $c:COMPOSE_SKIP=False $q:CUEQ_PROJECTED=False $t:GENERIC_FOLD_READOUT_TAIL=False > $out/check.log 2>&1 < /dev/null
  note "check exit=$? $(grep -hoE '"(spline_in_use|compose_skip_in_use|max_abs|same_build_max_abs|max_force)": [^,]+' $out/check.log | tr '\n' ' ' | cut -c1-300)"
  pc=(--production-constant "$@")
else
  pc=()
fi
python tools/force_stress_gate.py --output $out/gate --engine fastmace --model omat0-medium --checkpoint $ck \
  --structures-root $FASTMACE_STRUCTURES --structures "$st@3x3x3" NaCl NbCl5 "${pc[@]}" > $out/gate.log 2>&1 < /dev/null
note "gate exit=$? $(grep -h -iE 'verdict' $out/gate.log | tail -n 1 | cut -c1-200)"
