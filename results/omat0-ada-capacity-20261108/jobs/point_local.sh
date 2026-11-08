#!/usr/bin/env bash
# One capacity_scan.py point on the local RTX 5060 Ti with extra flags (e.g. --diagnose), scan_local.sh's environment:
#   python3 tools/ci/local_gpu.py run --quiet --purpose "..." -- env FM=<src> bash point_local.sh OUT STRUCTURE REPEAT [FLAGS...]
set -u
OUT=$1 STRUCTURE=$2 REPEAT=$3; shift 3
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/ada-capacity-local
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
mkdir -p "$OUT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$OUT/tmp \
  TMP=$OUT/tmp TEMP=$OUT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
"$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$STRUCTURE" --model omat0-medium --precision float32 \
  --output "$OUT" --repeat "$REPEAT" "$@" > "$OUT/point-$STRUCTURE-$REPEAT.log" 2>&1 < /dev/null
echo "exit=$? $(tail -n 3 "$OUT/point-$STRUCTURE-$REPEAT.log" | tr '\n' ' ' | cut -c1-400)"
