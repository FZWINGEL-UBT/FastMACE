#!/usr/bin/env bash
# CAP-T08 recompute_node_segments on the local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm cap-t08 JOB" -- \
#     env FM=<pinned branch copy> BEFORE=<pinned main copy> bash results/cap-t08-chunks-20261030/local.sh ROOT JOB...
# Jobs:
#   gate: tools/force_stress_gate.py with recompute_node_segments on (NaCl NbCl5 NaNbO3);
#   replay: results/cap-t08-chunks-20261030/replay.py, NbCl5 3x3x3, 200 steps, two runs each with the record off and
#     on (captured force graphs replayed every step): bitwise equal runs with it on whenever they are with it off;
#   ab-<cell>-<order>: tools/ab_rewrites.py, a = the default plan (the record off), b = recompute_node_segments on;
#     cell nbcl5 (3x3x3, 12 blocks of 300 steps) or big (NbCl5 4x4x8, 8 blocks of 100 steps after 100 warm-up
#     steps); order ab (pattern abab) or ba (abba);
#   point-<side>-<structure>-<repeat>: tools/bench/capacity_scan.py point, default plan, on side "after" (FM) or
#     "before" (BEFORE): rule 12 and the memory rescue decide.
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-t06-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
BEFORE=${BEFORE:-}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-t08-segments-local
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
note "fastmace=$(cat "$FM/.source_commit") before=$([ -n "$BEFORE" ] && cat "$BEFORE/.source_commit") host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gate) PYTHONPATH=$FM run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --reference-command "$REF" --production-command "$PY" --rewrites recompute_node_segments=on ;;
    replay) PYTHONPATH=$FM run $job "$PY" "$FM/results/cap-t08-chunks-20261030/replay.py" "$ROOT/replay.json" 3x3x3 200 ;;
    point-*) IFS=- read -r _ side structure repeat <<< "$job"
      pkg=$FM; [ "$side" = before ] && pkg=${BEFORE:?set BEFORE for before points}
      PYTHONPATH=$pkg run "$job" "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure" \
          --model mh1-r2scan --precision float32 --repeat "$repeat" --output "$ROOT/points" --tag "$side" ;;
    ab-*-ab|ab-*-ba) IFS=- read -r _ cell order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      structure=NbCl5; extra=()
      [ "$cell" = big ] && structure=NbCl5@4x4x8 && extra=(--warmup 100 --block 100 --blocks 8)
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab/$job" --checkpoint "$CK" \
          --model mh1-r2scan --structures "$structure" --b recompute_node_segments=on \
          --order "$order" --pattern "$pattern" --prepare "${extra[@]}" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
