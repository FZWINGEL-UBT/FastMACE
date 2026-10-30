#!/bin/bash
# Setup F1-F3/F5: setup time before and after, one source tree per call (TAG before|after; FM a pinned copy).
#   cold-<p>: fresh process, fresh empty cache; warm-<p>: fresh process, its cold run's cache;
#   chain-cold / chain-warm: NaCl, NbCl5, TaCl5, NbCl5 at 0.95 in one process, one shared cache (fresh, then warm).
set -u -o pipefail
TAG=$1; FM=$2; ROOT=$HOME/fastmace-capacity/setupfix/$TAG
mkdir -p "$ROOT/tmp"
ST=$FM/results/setup-across-compositions/structures
PY=$HOME/environments/mace_cueq_oeq/bin/python
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models TMPDIR=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
cache_env() { export FASTMACE_CACHE=$1 TORCHINDUCTOR_CACHE_DIR=$1/inductor TRITON_CACHE_DIR=$1/triton MACE_COMPILE_CACHE=$1/compile XDG_CACHE_HOME=$1/xdg; }
declare -A CIF=([NaCl]=mp-22862_NaCl [NbCl5]=mp-23307_NbCl5 [TaCl5]=mp-29831_TaCl5)
run() {  # name structures...  (REPEAT overrides 3x3x3)
  local name=$1; shift
  [ -e "$ROOT/$name" ] && return 0
  echo "$(date -u +%T) start $name" >> "$ROOT/run.log"
  "$PY" -W ignore "$FM/tools/bench/setup_compositions.py" --output "$ROOT/$name" --checkpoint $HOME/models/mace-mh-1.model \
    --label "$TAG-$name" --repeat "${REPEAT:-3x3x3}" --structures "$@" > "$ROOT/$name.log" 2>&1
  echo "$(date -u +%T) exit $? $name" >> "$ROOT/run.log"
}
echo "$(date -u +%T) $TAG fastmace=$(cat $FM/.source_commit)" >> "$ROOT/run.log"
for p in NaCl NbCl5; do cache_env "$ROOT/cache-$p"; run "cold-$p" "$p=$ST/${CIF[$p]}.cif"; done
for p in NaCl NbCl5; do cache_env "$ROOT/cache-$p"; run "warm-$p" "$p=$ST/${CIF[$p]}.cif"; done
# another atom count in the same size regime, in a fresh process on NbCl5's warm cache
cache_env "$ROOT/cache-NbCl5"; REPEAT=3x3x4 run "size-NbCl5-3x3x4" "NbCl5=$ST/${CIF[NbCl5]}.cif"
cache_env "$ROOT/cache-chain"
CH=("NaCl=$ST/${CIF[NaCl]}.cif" "NbCl5=$ST/${CIF[NbCl5]}.cif" "TaCl5=$ST/${CIF[TaCl5]}.cif" "NbCl5-0.95=$ST/${CIF[NbCl5]}.cif:0.95")
run chain-cold "${CH[@]}"
run chain-warm "${CH[@]}"
