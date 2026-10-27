#!/usr/bin/env bash
# ideas-20261027 D1 on Ada (murphy, RTX 4000 Ada Generation): float64 package against the fixed 56-bit emulation
# (EngineConfig.fp64_emulation_mantissa_bits=56, here through its environment variable), MD step ABBA (separate
# processes: the emulation setting is process-wide) and forces against stock. Run from a scratch folder under
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
  suite=$(pgrep -f pytest | paste -sd, -)
  [ -n "$gpu$busy$suite" ] && { echo "gpu=[$gpu] busy=[$busy] pytest=[$suite]"; return 0; }
  return 1
}
job() {  # name command...
  local name=$1; shift
  if why=$(others); then note "STOP before $name: $why"; exit 3; fi
  note "begin $name"; "$@" > "$ROOT/$name.log" 2>&1; note "end $name exit=$?"
}
SB="$FM/tools/bench/sb64b.py --checkpoint $CK --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
STOCK_RW="rewrite.radial_tables=off rewrite.species_tables=off rewrite.fold_readout_tail=off rewrite.fold_product_linear=off"
cd "$ROOT"
job acc-NbCl5-stock $E/bin/python $FM/tools/bench/f64pkg_acc.py stock NbCl5 $STOCK_RW engine.fp64_emulation=off
job acc-NbCl5-pkg $E/bin/python $FM/tools/bench/f64pkg_acc.py pkg NbCl5
job acc-NbCl5-m56 env $M=56 $E/bin/python $FM/tools/bench/f64pkg_acc.py pkg-m56 NbCl5
for s in NbCl5 NaCl; do
  job sb-$s-a1 $E/bin/python $SB --output "$ROOT/sb-$s-a1" --structure $s
  job sb-$s-b1 env $M=56 $E/bin/python $SB --output "$ROOT/sb-$s-b1" --structure $s
  job sb-$s-b2 env $M=56 $E/bin/python $SB --output "$ROOT/sb-$s-b2" --structure $s
  job sb-$s-a2 $E/bin/python $SB --output "$ROOT/sb-$s-a2" --structure $s
done
note DONE
