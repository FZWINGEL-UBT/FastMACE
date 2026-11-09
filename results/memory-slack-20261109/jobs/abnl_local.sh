#!/usr/bin/env bash
# abnl.sh on the local RTX 5060 Ti (WSL2): the same-process MD-step A/B of skin.SORT_KEYS_ONLY, both build orders,
# Benchmark_loup settings (SrTiO3 10x10x10, 1 fs, skin 0.5, warm-up 200, 6 blocks of 256 steps per side); a = False,
# b = True. Run inside a reservation, with an idle check before each order:
#   python3 tools/ci/local_gpu.py run --purpose "..." -- env FM=<pinned source> bash abnl_local.sh ROOT
set -u
ROOT=$1
FM=${FM:?set FM to a pinned source copy}
PY=$HOME/environments/mace_cueq_oeq/bin/python
CACHE=$HOME/fastmace-cache/ada-capacity-local
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
CK=$HOME/models/mace-omat-0-medium.model
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$ROOT/tmp" "$CACHE"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp \
  TMP=$ROOT/tmp TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {  # 6 samples, 5 s apart, all idle
  local ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 10 ] && [ "$wutil" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 60 ]; then note "NOT IDLE pids=[$pids] util=$util win=$wutil"; return 1; fi
    sleep 5
  done
}
o=fastmace.neighbors.skin
ST=$FM/results/loup-a100-20261103/SrTiO3.cif
note "abnl local fastmace=$(cat "$FM/.source_commit") gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for order in ab ba; do
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  idle || exit 3
  "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab-$order" --checkpoint "$CK" --model omat0-medium \
    --structures "$ST@10x10x10" --warmup 200 --block 256 --blocks 6 --order "$order" --pattern "$pattern" \
    --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --a-constant $o:SORT_KEYS_ONLY=False \
    --b-constant $o:SORT_KEYS_ONLY=False --pack "keys|$o:SORT_KEYS_ONLY=True" > "$ROOT/ab-$order.log" 2>&1 < /dev/null
  note "end $order exit=$? $(grep -E 'b/a' "$ROOT/ab-$order.log" | tail -n 1 | cut -c1-200)"
done
note DONE
