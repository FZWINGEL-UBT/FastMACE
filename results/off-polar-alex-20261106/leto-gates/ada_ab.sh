#!/usr/bin/env bash
# murphy (RTX 4000 Ada) packed A/B, off-polar-alex 7 Nov 2026: POLAR spin-off/fold, OFF stock skip + tables.
set -u -o pipefail
ROOT=$1; SRC=$2
E=/mnt/homes/zwingel/environments/mace_cueq_oeq
SITE=$E/lib/python3.10/site-packages
B=/mnt/homes/zwingel/fm-offpolar-gates
ORG=/mnt/homes/zwingel/ResearchModule/Structures/Organic
mkdir -p "$ROOT/tmp" "$ROOT/cache" "$ROOT/sets"
export PATH=$E/bin:$PATH PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$ROOT/cache FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models \
  FASTMACE_STRUCTURES=/mnt/homes/zwingel/ResearchModule TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor \
  TRITON_CACHE_DIR=$ROOT/cache/triton XDG_CACHE_HOME=$ROOT/cache/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
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
printf 'spinoff\t--a-constant fastmace.longrange.polar:SPIN_BATCHED=False\nfold\t--b-constant fastmace.rewrites.polar.update_fold:FOLD_UPDATE=True\n' > $ROOT/sets/polar.tsv
printf 'offst\t--b radial_tables=on --b-constant fastmace.rewrites.generic.modules:STOCK_FIRST_SKIP=True\n' > $ROOT/sets/off.tsv
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> $log
cd $B/src-$SRC && export PYTHONPATH=$B/src-$SRC
run() {  # model ckpt structs setsfile extra-env
  local m=$1 c=$2 st=$3 sets=$4
  if why=$(others); then note "STOP before $m: $why"; exit 3; fi
  note "begin $m"
  OUT=$ROOT/$m MODEL=$m CKPT=$c STRUCTS="$st" SETS_FILE=$sets "${@:5}" bash tools/offpolar_ab.sh
  note "end $m exit=$?"
}
for p in polar1-s:MACE-POLAR-1-S.model polar1-m:MACE-POLAR-1-M.model polar1-l:MACE-POLAR-1-L.model; do
  run ${p%%:*} ${p#*:} "NaCl NbCl5" $ROOT/sets/polar.tsv env BLOCK=400 BLOCKS=12 STOP_AFTER=4
done
for p in off23-small:MACE-OFF23_small.model off23-medium:MACE-OFF23_medium.model off23b-medium:MACE-OFF23b_medium.model off24-medium:MACE-OFF24_medium.model off23-large:MACE-OFF23_large.model; do
  run ${p%%:*} ${p#*:} "$ORG/water_216.extxyz@1x1x1" $ROOT/sets/off.tsv env WARMUP=500 BLOCK=1000 BLOCKS=16 STOP_AFTER=4
done
note done
grep -H "b/a" $ROOT/*/ab-*.log
