#!/usr/bin/env bash
# CAP-T06 (the noise batch capped at 64 MiB) on the local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm cap-t06 JOB" -- #     env FM=<pinned branch copy> BEFORE=<pinned main copy> bash results/cap-t06-20261030/local.sh ROOT JOB...
# Jobs:
#   bitwise: results/cap-t06-20261030/bitwise.py, reproducible float64 MH-1 NbCl5, each run a fresh process:
#     2x2x2 (288 atoms), 300 steps with a restart at 150, before and after (after: the cap set to 16 steps); after,
#     resumed from before's restart and from its own; 5x5x7 (6,300 atoms, the cap's own 221 steps), 300 steps with a
#     restart at 270 (242 unused rows, more than one batch), before and after, and after resumed from before's restart (4x4x8 runs out of memory in float64);
#   point-<side>-<structure>-<repeat>: tools/bench/capacity_scan.py point on side "after" (FM) or "before" (BEFORE);
#   ab-<cell>-<order>: tools/ab_rewrites.py, a = NOISE_BATCH_BYTES 2**62 (256 steps, as before), b = the default;
#     cell nbcl5 (3x3x3, 12 blocks of 300 steps) or big (NbCl5 4x4x8, 8 blocks of 100 steps after 100 warm-up
#     steps); order ab (pattern abab) or ba (abba).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-nlist-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
BEFORE=${BEFORE:-}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
PYT=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-t06-local
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
bitwise() {  # name package args...
  local name=$1 pkg=$2; shift 2
  PYTHONPATH=$pkg run "$name" "$PY" "$FM/results/cap-t06-20261030/bitwise.py" "$ROOT/bitwise/$name.json" "$@"
}
for job in "$@"; do
  case $job in
    bitwise) B0=${BEFORE:?set BEFORE}; D=$ROOT/bitwise; mkdir -p "$D"
      bitwise small-before "$B0" new "$D/r-small-before" NbCl5 2x2x2 300 150
      bitwise small-after "$FM" new "$D/r-small-after" NbCl5 2x2x2 300 150 16
      rm -rf "$D/r-small-before-copy" "$D/r-small-after-copy"
      cp -r "$D/r-small-before" "$D/r-small-before-copy"; cp -r "$D/r-small-after" "$D/r-small-after-copy"
      bitwise small-after-from-before "$FM" resume "$D/r-small-before-copy" 150 16
      bitwise small-after-from-after "$FM" resume "$D/r-small-after-copy" 150 16
      bitwise big-before "$B0" new "$D/r-big-before" NbCl5 5x5x7 300 270
      bitwise big-after "$FM" new "$D/r-big-after" NbCl5 5x5x7 300 270
      rm -rf "$D/r-big-before-copy"; cp -r "$D/r-big-before" "$D/r-big-before-copy"
      bitwise big-after-from-before "$FM" resume "$D/r-big-before-copy" 30 ;;
    point-*) IFS=- read -r _ side structure repeat <<< "$job"
      pkg=$FM; [ "$side" = before ] && pkg=${BEFORE:?set BEFORE for before points}
      PYTHONPATH=$pkg run "$job" "$PY" "$FM/tools/bench/capacity_scan.py" point --structure "$structure"           --model mh1-r2scan --precision float32 --repeat "$repeat" --output "$ROOT/points" --tag "$side" ;;
    ab-*-ab|ab-*-ba) IFS=- read -r _ cell order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      structure=NbCl5; extra=()
      [ "$cell" = big ] && structure=NbCl5@4x4x8 && extra=(--warmup 100 --block 100 --blocks 8)
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab/$job" --checkpoint "$CK"           --model mh1-r2scan --structures "$structure"           --a-constant "fastmace.integrators.langevin:NOISE_BATCH_BYTES=4611686018427387904"           --order "$order" --pattern "$pattern" --prepare "${extra[@]}" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
