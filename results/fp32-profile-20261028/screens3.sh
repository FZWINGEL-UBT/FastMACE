#!/usr/bin/env bash
# fp32 profile follow-up screens, third batch (29 October 2026; screens2.sh stays as it ran), local RTX 5060 Ti or a shared workstation (Ada):
#   tailcheck  tools/bench/tail_head_check.py: energies/forces with the readout head slice (TAIL_HEAD=0) against the default
#   tailhead   A/B, b = fold_readout_tail keeps only head 0's 16 hidden columns (prototype constant TAIL_HEAD)
#   specres    A/B, b = species_residual=on
# Usage: bash screens.sh local|ws ROOT FM MODE...
# local: inside `tools/ci/local_gpu.py run --quiet`; ws: from a scratch folder under /mnt/homes/zwingel, and every job
# first checks that no other user's job runs and the GPU has no compute process (AGENTS.md, "Shared workstations").
set -u -o pipefail
KIND=$1 ROOT=$2 FM=$3; shift 3
mkdir -p "$ROOT/tmp" "$ROOT/cache"
if [ "$KIND" = ws ]; then
  E=/mnt/homes/zwingel/environments/mace_cueq_oeq
  CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
  B=/mnt/homes/zwingel/fastmace-baseline/1a-T18
  export FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models CUDA_CACHE_DISABLE=1 \
    LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
  CACHE=$ROOT/cache
else
  E=$HOME/environments/mace_cueq_oeq
  CK=$HOME/models/mace-mh-1.model
  B=$HOME/fastmace-baseline/1a-T18
  export FASTMACE_MODELS=$HOME/models
  CACHE=$HOME/fastmace-cache/fp32-profile
fi
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_STRUCTURES=$B \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile \
  XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
PY=$E/bin/python
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {  # ws only: other users' processes using CPU (a job), or any GPU compute process
  [ "$KIND" = ws ] || return 1
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
run() {  # name command...
  local name=$1 why; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
ab() {  # name "structures" side args...: tools/ab_rewrites.py, 1 fs, 1000 K, both build orders per structure
  local name=$1 structures=$2; shift 2
  for s in $structures; do for order in ab ba; do
    if [ "$order" = ab ]; then pattern=abab; else pattern=abba; fi
    run "$name-$s-$order" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/$name-$s-$order" --checkpoint "$CK"       --model mh1-r2scan --structures "$s" --timestep-fs 1 --temperature-K 1000 --order "$order"       --pattern "$pattern" "$@"
  done; done
}
note "fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) modes=$*"
for mode in "$@"; do
  case $mode in
    tailcheck) run tailcheck "$PY" "$FM/tools/bench/tail_head_check.py" --output "$ROOT/tailcheck" --checkpoint "$CK" ;;
    tailhead) ab tailhead "NaCl NbCl5 NaNbO3" --b-constant "fastmace.rewrites.species:TAIL_HEAD=0" ;;
    specres) ab specres "NbCl5 NaNbO3 NaCl" --b species_residual=on ;;
  esac
done
note DONE
