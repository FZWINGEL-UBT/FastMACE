#!/usr/bin/env bash
# Decision 63 on Ada (sm_89): float64 package against stock (step times, tools/step_breakdown.py with --dtype and
# --engine, as sb64b.py), plus the GEMM format-rate probe. Run from a scratch folder under /mnt/homes/zwingel with a
# git archive of the source (FM); the shared-workstation idle check runs before every job.
#   bash ada.sh ROOT FM
set -u -o pipefail
ROOT=$1 FM=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
CK=/mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton MACE_COMPILE_CACHE=$ROOT/cache/compile XDG_CACHE_HOME=$ROOT/cache/xdg \
  TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cu13/lib:$E/lib/python3.10/site-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$log"; }
others() {
  local gpu busy
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20 {print $1":"$3":"$2}' | paste -sd, -)
  [ -n "$gpu$busy" ] && { echo "gpu=[$gpu] busy=[$busy]"; return 0; }
  return 1
}
job() {  # name command...
  local name=$1; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"; "$@" > "$ROOT/$name.log" 2>&1; note "end $name exit=$?"
}
STOCK="radial_tables=off species_tables=off fold_readout_tail=off fold_product_linear=off"
SB="$FM/tools/sb64b.py --checkpoint $CK --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
cd "$ROOT"
job rates $E/bin/python $FM/tools/rates.py
for s in NaCl NbCl5; do
  job sb-$s-stock $E/bin/python $SB --output "$ROOT/sb-$s-stock" --structure $s --rewrites $STOCK --engine fp64_emulation=off
  job sb-$s-pkg $E/bin/python $SB --output "$ROOT/sb-$s-pkg" --structure $s
done
note DONE
