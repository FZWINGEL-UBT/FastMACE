#!/usr/bin/env bash
# Speed items A-C of plans/capacity-ideas-20261030.md's fourth list on the local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm speed-small JOB" -- \
#     env FM=<pinned branch copy> [CACHE=<fresh compiler cache>] bash results/speed-small-20261030/local.sh ROOT JOB...
# Jobs:
#   profile-<structure>: tools/step_breakdown.py's main pass (NaCl, NbCl5 or NaNbO3 3x3x3, 1000 K, 1 fs, as the
#     28 October profile) with Inductor's generated code logged (TORCH_LOGS=output_code, <structure>.code.log), then
#     its nsys pass (force-graph replays, every graph node traced) and the gaps between nodes;
#   gputest: tests/gpu/test_segment_gemm_constexpr_gpu.py;
#   ab-<structure>-<size>-<order>: tools/bench/ab_constexpr_d.py, a = the runtime-D kernel (main's), b = the
#     constexpr-D kernel; size small (NbCl5 3x3x3, 1,944 atoms; NaNbO3 4x4x3, 1,920; 12 blocks of 300
#     steps) or big (NbCl5 4x4x8, 9,216 atoms; NaNbO3 6x6x6, 8,640; 8 blocks of 100 steps after 100 warm-up
#     steps); order ab (pattern abab) or ba (abba);
#   gate: tools/force_stress_gate.py, default plan (NaCl NbCl5 NaNbO3).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-outputs-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
PYT=$HOME/environments/mace_cueq_latest/bin/python
CACHE=${CACHE:-$HOME/fastmace-cache/speed-small-local}
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
NSYS=/usr/local/cuda/bin/nsys
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
COMMON=(--checkpoint "$CK" --model mh1-r2scan --backend hybrid --temperature-K 1000 --timestep-fs 1)
note "fastmace=$(cat "$FM/.source_commit") cache=$CACHE host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    profile-*) structure=${job#profile-}; dir=$ROOT/profile/$structure
      mkdir -p "$ROOT/profile"
      PYTHONPATH=$FM TORCH_LOGS=output_code run "$job" "$PY" "$FM/tools/step_breakdown.py" --output "$dir" \
          "${COMMON[@]}" --structure "$structure"
      mv "$ROOT/$job.log" "$ROOT/profile/$structure.code.log"
      PYTHONPATH=$FM run "$job-nsys" "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node \
          --capture-range=cudaProfilerApi --capture-range-end=stop -o "$dir/nsys" --force-overwrite true \
          "$PY" "$FM/tools/step_breakdown.py" --pass nsys --output "$dir" "${COMMON[@]}" --structure "$structure"
      "$NSYS" stats --report cuda_gpu_trace --format csv --output "$dir/nsys_stats" --force-export true \
          "$dir/nsys.nsys-rep" > "$ROOT/$job-nsys-csv.log" 2>&1
      PYTHONPATH=$FM "$PY" "$FM/tools/step_breakdown.py" --output "$dir" \
          --summarize-nsys "$dir/nsys_stats_cuda_gpu_trace.csv" > "$ROOT/$job-nsys-gaps.log" 2>&1
      note "nsys summary exit=$?" ;;
    gputest) (cd "$FM" && PYTHONPATH=$FM run $job "$PYT" -m pytest -q -p no:cacheprovider -rs \
        --basetemp="$ROOT/tmp/pytest" tests/gpu/test_segment_gemm_constexpr_gpu.py) ;;
    gate) PYTHONPATH=$FM run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --reference-command "$PYT" --production-command "$PY" ;;
    ab-*) IFS=- read -r _ structure size order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      spec=$structure; extra=()
      if [ "$size" = big ]; then
        [ "$structure" = NbCl5 ] && spec=NbCl5@4x4x8 || spec=$structure@6x6x6
        extra=(--warmup 100 --block 100 --blocks 8)
      elif [ "$structure" = NaNbO3 ]; then
        spec=NaNbO3@4x4x3
      fi
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/bench/ab_constexpr_d.py" --output "$ROOT/ab/$job" \
          --checkpoint "$CK" --model mh1-r2scan --structures "$spec" --order "$order" --pattern "$pattern" \
          --prepare "${extra[@]}" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
