#!/usr/bin/env bash
# ideas-20261027 D2, D3 and D1+D2+D3 on Ada (murphy): float64 package against w_tables, save_transcendentals and
# the three together (56-bit emulation through its environment variable), MD step ABBA in separate processes,
# and forces of the combination against the package. Run from a scratch folder under
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
job acc-NbCl5-pkg $E/bin/python $FM/tools/bench/f64pkg_acc.py pkg NbCl5
job acc-NbCl5-all env $M=56 $E/bin/python $FM/tools/bench/f64pkg_acc.py all NbCl5 rewrite.w_tables=on rewrite.save_transcendentals=on
for s in NbCl5 NaCl; do
  for v in w st all; do
    case $v in
      w) B=(--rewrites w_tables=on); X=();;
      st) B=(--rewrites save_transcendentals=on); X=();;
      all) B=(--rewrites w_tables=on save_transcendentals=on); X=(env $M=56);;
    esac
    job sb-$s-$v-a1 $E/bin/python $SB --output "$ROOT/sb-$s-$v-a1" --structure $s
    job sb-$s-$v-b1 "${X[@]}" $E/bin/python $SB --output "$ROOT/sb-$s-$v-b1" --structure $s "${B[@]}"
    job sb-$s-$v-b2 "${X[@]}" $E/bin/python $SB --output "$ROOT/sb-$s-$v-b2" --structure $s "${B[@]}"
    job sb-$s-$v-a2 $E/bin/python $SB --output "$ROOT/sb-$s-$v-a2" --structure $s
  done
done
note DONE
