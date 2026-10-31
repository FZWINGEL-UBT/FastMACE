#!/usr/bin/env bash
# recompute_first_table (rebuild-a) on the local RTX 5060 Ti (sm_120, 16 GB, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm rebuild-a JOB" -- \
#     env FM=<pinned branch copy> [CACHE=<fresh compiler cache>] bash results/rebuild-a-20261031/local.sh ROOT JOB...
# Jobs:
#   gputest: tests/gpu/test_kernels_triton_cuda.py (the radial table and edge_outer_sum kernels);
#   gate: tools/force_stress_gate.py with recompute_first_table on (NaCl NbCl5 NaNbO3);
#   ledger-<structure>-<repeat>-<mode>: tools/bench/memory_ledger.py, mode nodeseg (recompute_conv_weights and
#     recompute_node_segments on, recompute_first_table off: the speed-def falsifier's plan) or table (the same and recompute_first_table on);
#   ab-<structure>-<size>-<order>: tools/ab_rewrites.py, a = the default plan, b = recompute_first_table on; size small
#     (NaCl and NbCl5 3x3x3, NaNbO3 4x4x3: 216, 1,944, 1,920 atoms; 12 blocks of 300 steps) or big (NbCl5 4x4x8,
#     NaNbO3 6x6x6: 9,216, 8,640 atoms; 8 blocks of 100 steps after 100 warm-up steps), order ab (abab) or ba (abba);
#   point-<structure>-<repeat>: tools/bench/capacity_scan.py point, default plan (rule 12 decides), 200 steps.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/speed-def-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
PYT=$HOME/environments/mace_cueq_latest/bin/python
CACHE=${CACHE:-$HOME/fastmace-cache/rebuild-a-local}
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
CK=$HOME/models/mace-mh-1.model
mkdir -p "$ROOT/tmp" "$ROOT/ab" "$CACHE"
export PYTHONDONTWRITEBYTECODE=1 FASTMACE_CACHE=$CACHE FASTMACE_MODELS=$HOME/models \
  FASTMACE_STRUCTURES=$B TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
  MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp \
  TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=$PY
log=$ROOT/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
idle() {  # 6 samples, 5 s apart, all idle
  local name=$1 ok=0 tries=0 pids util wutil
  while [ $ok -lt 6 ]; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    wutil=$($WSMI --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')
    if [ -z "$pids" ] && [ "$util" -le 10 ] && [ "$wutil" -le 10 ]; then ok=$((ok + 1)); else ok=0; fi
    tries=$((tries + 1))
    if [ $tries -ge 60 ]; then note "$name NOT IDLE pids=[$pids] util=$util win=$wutil"; return 1; fi
    sleep 5
  done
  note "$name idle: util=$util win=$wutil mem=$($WSMI --query-gpu=memory.used --format=csv,noheader | tr -d '\r') load=$(cut -d' ' -f1-3 /proc/loadavg) avail_mb=$(free -m | awk '/^Mem:/{print $7}')"
}
run() {  # name command...
  local name=$1; shift
  idle "$name" || exit 3
  note "start $name"
  "$@" > "$ROOT/$name.log" 2>&1 < /dev/null
  note "end $name exit=$?"
}
note "fastmace=$(cat "$FM/.source_commit") cache=$CACHE host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gputest) (cd "$FM" && PYTHONPATH=$FM run $job "$PYT" -m pytest -q -p no:cacheprovider -rs \
        --basetemp="$ROOT/tmp/pytest" tests/gpu/test_kernels_triton_cuda.py) ;;
    gate) PYTHONPATH=$FM run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
        --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
        --reference-command "$PYT" --production-command "$PY" --rewrites recompute_first_table=on ;;
    ledger-*) IFS=- read -r _ structure repeat mode <<< "$job"
      rw=(--rewrites recompute_conv_weights=on recompute_node_segments=on)
      [ "$mode" = table ] && rw+=(recompute_first_table=on) || rw+=(recompute_first_table=off)  # else rule 12 turns it on at the limits
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/memory_ledger.py" --structure "$structure" --repeat "$repeat" \
        --output "$ROOT/ledger" --tag="-$mode" "${rw[@]}" ;;
    ab-*) IFS=- read -r _ structure size order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      spec=$structure; extra=()
      if [ "$size" = big ]; then
        [ "$structure" = NbCl5 ] && spec=NbCl5@4x4x8 || spec=$structure@6x6x6
        extra=(--warmup 100 --block 100 --blocks 8)
      elif [ "$structure" = NaNbO3 ]; then
        spec=NaNbO3@4x4x3
      fi
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --b recompute_first_table=on --output "$ROOT/ab/$job" \
        --checkpoint "$CK" --model mh1-r2scan --structures "$spec" --order "$order" --pattern "$pattern" --prepare \
        "${extra[@]}" ;;
    point-*) IFS=- read -r _ structure repeat <<< "$job"
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" \
        --repeat "$repeat" --output "$ROOT/points" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
