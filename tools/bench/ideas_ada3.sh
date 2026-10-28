#!/usr/bin/env bash
# ideas-20261027 per-GPU confirmation on Ada (murphy): same-process A/Bs in both build orders (tools/ab_rewrites.py
# --precision float64) for w_tables and save_transcendentals on NbCl5, NaNbO3 and NaCl, and D1's ABBA on NaNbO3
# (separate processes: the emulation setting is process-wide). Run from a scratch folder under
# /mnt/homes/zwingel with a git archive of the branch (FM); the shared-workstation idle check (GPU compute
# processes, other users' CPU jobs, our own pytest suites) runs before every job.
#   bash ideas_ada.sh ROOT FM
set -u -o pipefail
ROOT=$1 FM=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM:$FM/tools PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
M=CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {
  local gpu busy suite
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  suite=$(pgrep -f "[p]ytest" | paste -sd, -)
  [ -n "$gpu$busy$suite" ] && { echo "gpu=[$gpu] busy=[$busy] pytest=[$suite]"; return 0; }
  return 1
}
job() {  # name command...
  local name=$1; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"; "$@" > "$ROOT/$name.log" 2>&1; note "end $name exit=$?"
}
SB="$FM/tools/bench/sb64b.py --checkpoint $CK --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
cd "$ROOT"
for rec in w_tables save_transcendentals; do
  for o in ab ba; do
    job ab-$rec-$o $E/bin/python $FM/tools/ab_rewrites.py --output "$ROOT/ab-$rec-$o" --checkpoint $CK --precision float64       --a $rec=off --b $rec=on --structures NbCl5 NaNbO3 NaCl --warmup 60 --block 60 --blocks 6 --order $o
  done
done
job sb-NaNbO3-a1 $E/bin/python $SB --output "$ROOT/sb-NaNbO3-a1" --structure NaNbO3
job sb-NaNbO3-b1 env $M=56 $E/bin/python $SB --output "$ROOT/sb-NaNbO3-b1" --structure NaNbO3
job sb-NaNbO3-b2 env $M=56 $E/bin/python $SB --output "$ROOT/sb-NaNbO3-b2" --structure NaNbO3
job sb-NaNbO3-a2 $E/bin/python $SB --output "$ROOT/sb-NaNbO3-a2" --structure NaNbO3
note DONE
