#!/usr/bin/env bash
# cap-nlist (lean neighbour buffers and the chunked GPU build) on the local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm cap-nlist JOB" -- \
#     env FM=<pinned branch copy> BEFORE=<pinned main copy> bash results/cap-nlist-20261030/local.sh ROOT JOB...
# Jobs:
#   gputests: the neighbour-list GPU tests (mace_cueq_latest's pytest, MH-1 drivers with the hybrid Python);
#   leantests: tests/gpu/test_neighbors_lean_gpu.py alone;
#   point-<side>-<structure>-<repeat>[-off]: tools/bench/capacity_scan.py point (the branch's tool) on the package
#     of side "after" (FM) or "before" (BEFORE, main); "-off" sets LEAN_BUFFERS = False in the branch;
#   ab-<cell>-<order>: tools/ab_rewrites.py, a = LEAN_BUFFERS False (the layout before), b = the default; cell nbcl5
#     (3x3x3, 12 blocks of 300 steps) or big (NbCl5 4x4x8, 8 blocks of 100 steps after 100 warm-up steps); order ab
#     (pattern abab) or ba (abba).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-t07b-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
BEFORE=${BEFORE:-}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
PYT=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/cap-nlist-local
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
    gputests) (cd "$FM" && PYTHONPATH=$FM run $job "$PYT" -m pytest -q -p no:cacheprovider -rs \
        --basetemp="$ROOT/tmp/pytest" tests/gpu/test_neighbors_lean_gpu.py tests/gpu/test_neighbors_gpu.py \
        tests/gpu/test_async_rebuild_gpu.py tests/gpu/test_d3_list_gpu.py \
        "tests/gpu/test_fused_integrator_gpu.py::test_the_fused_stage_writes_the_pytorch_stage_bytes") ;;
    leantests) (cd "$FM" && PYTHONPATH=$FM run $job "$PYT" -m pytest -q -p no:cacheprovider -rs \
        --basetemp="$ROOT/tmp/pytest" tests/gpu/test_neighbors_lean_gpu.py) ;;
    point-*) IFS=- read -r _ side structure repeat off <<< "$job"
      pkg=$FM; [ "$side" = before ] && pkg=${BEFORE:?set BEFORE for before points}
      launcher=("$PY" "$FM/tools/bench/capacity_scan.py")
      if [ "${off:-}" = off ]; then
        launcher=("$PY" -c "import runpy, sys; import fastmace; from fastmace.neighbors import skin; \
skin.LEAN_BUFFERS = False; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
                  "$FM/tools/bench/capacity_scan.py")
      fi
      PYTHONPATH=$pkg run "$job" "${launcher[@]}" point --structure "$structure" --model mh1-r2scan \
          --precision float32 --repeat "$repeat" --output "$ROOT/points" --tag "$side${off:+-$off}" ;;
    ab-*-ab|ab-*-ba) IFS=- read -r _ cell order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      structure=NbCl5; extra=()
      [ "$cell" = big ] && structure=NbCl5@4x4x8 && extra=(--warmup 100 --block 100 --blocks 8)
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab/$job" --checkpoint "$CK" \
          --model mh1-r2scan --structures "$structure" --a-constant "fastmace.neighbors.skin:LEAN_BUFFERS=False" \
          --order "$order" --pattern "$pattern" --prepare "${extra[@]}" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
