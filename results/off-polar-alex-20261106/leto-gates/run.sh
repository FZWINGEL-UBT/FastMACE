#!/usr/bin/env bash
# Workstation gate runner (off-polar-alex, 7 Nov 2026): bash run.sh ROOT TASK... ; idle check before every gate.
set -u -o pipefail
ROOT=$1; shift
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SITE=$E/lib/python3.10/site-packages
mkdir -p "$ROOT/tmp" "$ROOT/cache" "$ROOT/runs"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton XDG_CACHE_HOME=$ROOT/cache/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
ORG=/mnt/homes/zwingel/ResearchModule/Structures/Organic
B=/mnt/homes/zwingel/fm-offpolar-gates
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
off_gate() {  # tag src constant-or-rewrite-args models...
  local tag=$1 src=$2 cand=$3; shift 3
  for p in "$@"; do
    local m=${p%%:*} c=${p#*:} GS
    if [ $m = off23-large ]; then GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif@2x2x2 $ORG/water_216.extxyz@1x1x1"
    else GS="$ORG/urea_cod1008787.cif $ORG/naphthalene_cod2311090.cif $ORG/paracetamol_cod2104364.cif $ORG/water_216.extxyz@1x1x1"; fi
    gate_pair $tag $src $m $c "$GS" "$cand"
  done
}
polar_gate() {  # tag src candidate-args
  local tag=$1 src=$2 cand=$3
  for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
    gate_pair $tag $src ${p%%:*} ${p#*:} "NaCl NbCl5" "$cand"
  done
}
polar_gate_sm() {  # tag src candidate-args: POLAR-S and -M (POLAR-L's float64 reference needs 80 GB)
  local tag=$1 src=$2 cand=$3
  for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model; do
    gate_pair $tag $src ${p%%:*} ${p#*:} "NaCl NbCl5" "$cand"
  done
}
gate_pair() {  # tag src model ckpt structs candidate-args
  local tag=$1 src=$2 m=$3 c=$4 gs=$5 cand=$6
  if why=$(others); then note "STOP before $tag $m: $why"; exit 3; fi
  local R=$ROOT/runs/gate-$tag-$m
  note "begin $tag $m"
  ( cd $B/src-$src && export PYTHONPATH=$B/src-$src && \
    OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=default GATE_ARGS="--seed 20261015" bash tools/offpolar_ab.sh && \
    OUT=$R MODEL=$m CKPT=$c STRUCTS=x GATE_STRUCTS="$gs" GATE_NAME=candidate GATE_ARGS="--seed 20261015 $cand" bash tools/offpolar_ab.sh )
  note "end $tag $m exit=$?"
  echo "GATE $tag $m"; python $B/gatesum.py $R/gate-*.log
}
OFFA="off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23-large:MACE-OFF23_large.model"
OFFB="off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model"
for t in "$@"; do
  case $t in
    stock) off_gate stock 313cdf73 "--production-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True" $OFFA $OFFB ;;
    tables) off_gate tables 6c9d303f "--rewrites radial_tables=on" $OFFA $OFFB ;;
    e3nn) polar_gate e3nn 527adb70 "--production-constant fastmace.rewrites.linear:E3NN_BLOCK_LINEAR=True" ;;
    geo) polar_gate geo ecbba373 "--production-constant fastmace.longrange.polar:SHARED_GEOMETRY=True" ;;
    spin) polar_gate_sm spin 1e6ef273 "--production-constant fastmace.longrange.polar:SPIN_BATCHED=True" ;;
    mm) polar_gate_sm mm 1e6ef273 "--production-constant fastmace.rewrites.inductor:SMALL_MATMULS=True" ;;
    offst) off_gate offst 3b82ad14 "--rewrites radial_tables=on --production-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True" off23-medium:MACE-OFF23_medium.model off24-medium:MACE-OFF24_medium.model ;;
    fold) polar_gate_sm fold 0cf1dc1f "--production-constant fastmace.rewrites.polar.update_fold:FOLD_UPDATE=True" ;;
    mmoff) off_gate mmoff 1e6ef273 "--production-constant fastmace.rewrites.inductor:SMALL_MATMULS=True" off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model ;;
  esac
done
note "all done"; echo "WSDONE $(hostname)"
