#!/bin/bash
# BINNED_SCREEN on the local RTX 5060 Ti (11 November 2026, owner: "Yes, run the BINNED_SCREEN A/B on the 5060 Ti").
# (1) rebuild_parts: the neighbour-list build, tile screen against cell-list screen, SrTiO3 10^3 to 35^3 (list only).
# (2) MD-step A/B, a = BINNED_SCREEN False (the 5060 Ti default), b = True, both orders, SrTiO3 16^3 (20,480 atoms).
# (3) the same at 20^3 (40,000 atoms), only if the 16^3 peak predicts under 13,000 MiB on the card.
# Watchdog: kills a step when the card passes 14,500 MiB, on a new soft-lockup / RCU-stall kernel line, or past its
# time limit. Runs inside one local_gpu reservation (launch.sh). Stops at the first failure.
set -u
RUN=$HOME/fm-runs/binned-5060-20261111
SRC=$RUN/src
OUT=$RUN/out
PY=$HOME/environments/mace_cueq_oeq/bin/python
SITE=$HOME/environments/mace_cueq_oeq/lib/python3.10/site-packages
CACHE=$HOME/fastmace-cache/binned-5060
mkdir -p "$OUT/tmp" "$CACHE"
export PATH=/usr/lib/wsl/lib:$PATH
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$OUT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY FASTMACE_SOURCE_COMMIT=$(cat "$SRC/.source_commit") \
  LD_LIBRARY_PATH=$SITE/nvidia/cu13/lib:$SITE/cuequivariance_ops/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
unset PYTORCH_CUDA_ALLOC_CONF
cd "$SRC"
CIF=$SRC/results/loup-a100-20261103/SrTiO3.cif
K=fastmace.neighbors.pair_search
kmsg() { sudo -n dmesg 2>/dev/null || journalctl -k -b --no-pager -o cat 2>/dev/null; }
lockups() { kmsg | grep -c -E 'soft lockup|rcu.*detected stalls'; }
note() { echo "$(date +%T) $*"; }
card() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

watch() { pid=$1; name=$2; limit=$3; base=$(lockups); peak=0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    used=$(card)
    [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used && echo "$peak" > "$OUT/peak-$name.txt"
    why=""
    [ "$used" -gt 14500 ] 2>/dev/null && why="card at ${used} MiB (> 14,500)"
    [ "$(lockups)" -gt "$base" ] && why="new soft lockup / RCU stall kernel line"
    [ $(( $(date +%s) - t0 )) -gt "$limit" ] && why="past ${limit} s"
    if [ -n "$why" ]; then
      note "WATCHDOG $name: $why, killing"; pkill -9 -P "$pid"; kill -9 "$pid"; echo "$why" > "$OUT/watchdog-$name.txt"; return
    fi
    sleep 2
  done; }

step() { name=$1; limit=$2; shift 2
  note "== $name (limit ${limit} s, card $(card) MiB)"
  "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
  pid=$!; t0=$(date +%s)
  watch "$pid" "$name" "$limit"
  wait "$pid"; code=$?
  [ -e "$OUT/watchdog-$name.txt" ] && code=99
  note "end $name exit=$code took=$(( $(date +%s) - t0 ))s peak=$(cat "$OUT/peak-$name.txt" 2>/dev/null) MiB"
  grep -h -E 'b/a' "$OUT/$name.log" | tail -n 2 | cut -c1-300
  if [ "$code" != 0 ]; then tail -25 "$OUT/$name.log" | cut -c1-300; note "STOP after $name"; echo "JOB DONE (failed)"; exit 1; fi; }

ab() { cellrep=$1; order=$2; limit=$3
  pattern=$([ "$order" = ab ] && echo abab || echo abba)
  step "ab-$cellrep-$order" "$limit" "$PY" tools/ab_rewrites.py --output "$OUT/ab-$cellrep-$order" \
    --checkpoint "$HOME/models/mace-omat-0-medium.model" --model omat0-medium --structures "$CIF@$cellrep" \
    --warmup 200 --block 128 --blocks 6 --order "$order" --pattern "$pattern" --timestep-fs 1 \
    --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --a-constant $K:BINNED_SCREEN=False --b-constant $K:BINNED_SCREEN=True; }

base=$(card)
note "source $(cat "$SRC/.source_commit"); $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader); lockup lines $(lockups)"
step parts 1500 "$PY" tools/bench/rebuild_parts.py --structure "$CIF" --repeats 10 16 20 24 30 35 --binned 0 1 \
  --rounds 3 --output "$OUT/parts.json"
tail -40 "$OUT/parts.log" | cut -c1-240
ab 16x16x16 ab 2400
ab 16x16x16 ba 2400
peak16=$(cat "$OUT/peak-ab-16x16x16-ba.txt")
predicted=$(( base + (peak16 - base) * 40000 / 20480 ))
note "16^3 peak $peak16 MiB over a base of $base MiB: 20^3 predicted $predicted MiB"
if [ "$predicted" -gt 13000 ]; then note "SKIP 20^3: predicted above 13,000 MiB"; echo "JOB DONE (20^3 skipped)"; exit 0; fi
ab 20x20x20 ab 3600
ab 20x20x20 ba 3600
echo "JOB DONE"
