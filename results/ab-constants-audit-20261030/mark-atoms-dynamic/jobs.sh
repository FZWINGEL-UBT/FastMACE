set -u -o pipefail
ROOT=$1 FM=$2
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/segment-gemm-order-local
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$B \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile \
  XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY
mkdir -p "$ROOT/tmp"
for s in NaCl NbCl5; do for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  echo "$(date -u +%FT%TZ) start $s-$order" >> "$ROOT/run.log"
  "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/mad-$s-$order" --checkpoint "$HOME/models/mace-mh-1.model" \
    --model mh1-r2scan --structures "$s" --prepare --a-constant fastmace.model.compile:MARK_ATOMS_DYNAMIC=False \
    --order "$order" --pattern "$pattern" > "$ROOT/mad-$s-$order.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) end $s-$order exit=$?" >> "$ROOT/run.log"
done; done
