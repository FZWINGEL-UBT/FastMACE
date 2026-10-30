#!/usr/bin/env bash
# Compile-cache probe (conductor addendum 1, 30 October 2026): does Dynamo's caching_precompile skip tracing of the
# first compiled force call in a fresh process with a warm cache, at an edge count other than the cached one?
# NaCl 3x3x3 (MH-1 float32, default plan), one fresh shared cache, four processes in order:
#   pre-cold   NaCl,      caching_precompile=True   (fills the cache)
#   pre-warm   NaCl 0.97, caching_precompile=True   (other edge count; warm)
#   pre-warm2  NaCl 0.97, caching_precompile=True   (the same again)
#   pre-base   NaCl 0.97, default                   (warm, tracing as today)
# Inside a local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "setup across compositions precompile (<15 min)" -- \
#     bash results/setup-across-compositions/run_precompile.sh ROOT
set -u -o pipefail
ROOT=$1
FM=/mnt/d/Margraf/fm-setup-compositions
S=$FM/results/setup-across-compositions/structures
PY=$HOME/environments/mace_cueq_oeq/bin/python
CKPT=$HOME/models/mace-mh-1.model
C=$ROOT/cache-precompile
mkdir -p "$ROOT/tmp"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models TMPDIR=$ROOT/tmp TMP=$ROOT/tmp \
  TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_CACHE=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  TRITON_CACHE_DIR=$C/triton MACE_COMPILE_CACHE=$C/compile XDG_CACHE_HOME=$C/xdg
run() {  # name structure [extra args]
  local name=$1 structure=$2; shift 2
  [ -e "$ROOT/$name" ] && { echo "$ROOT/$name exists; skipped"; return 0; }
  echo "$(date -u +%FT%TZ) start $name $*" | tee -a "$ROOT/run.log"
  "$PY" -W ignore "$FM/tools/bench/setup_compositions.py" --output "$ROOT/$name" --checkpoint "$CKPT" \
    --label "$name" --structures "$structure" "$@" > "$ROOT/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) exit $? $name" | tee -a "$ROOT/run.log"
  grep -v "^USDT\|^\s*$" "$ROOT/$name.log" | tail -2
}
NACL=$S/mp-22862_NaCl.cif
run pre-cold "NaCl=$NACL" --dynamo-config caching_precompile=True
run pre-warm "NaCl-0.97=$NACL:0.97" --dynamo-config caching_precompile=True
run pre-warm2 "NaCl-0.97=$NACL:0.97" --dynamo-config caching_precompile=True
run pre-base "NaCl-0.97=$NACL:0.97"
