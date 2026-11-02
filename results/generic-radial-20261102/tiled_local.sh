#!/usr/bin/env bash
# ab_arms.py's tiled A/B of the generic radial tables on the local RTX 5060 Ti (WSL2), inside a reservation:
#   python3 tools/ci/local_gpu.py run --purpose "..." -- env FM=<pinned source copy> \
#     bash results/generic-radial-20261102/tiled_local.sh ROOT REPEAT ARMS
set -u
ROOT=$1 REPEAT=$2 ARMS=$3
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/capacity-local
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_SOURCE_COMMIT=$(cat "$FM/.source_commit")
name=tiled-SrTiO3-$REPEAT-$(echo "$ARMS" | tr ',+' '__')
"$PY" "$FM/results/tiled-speed-20261101/ab_arms.py" --output "$ROOT/$name" --structure SrTiO3 --model omat0-medium \
  --repeat "$REPEAT" --arms "$ARMS" --calls 1 > "$ROOT/$name.log" 2>&1 < /dev/null
echo "end $name exit=$?" >> "$ROOT/run.log"
