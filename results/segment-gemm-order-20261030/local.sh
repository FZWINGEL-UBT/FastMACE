#!/usr/bin/env bash
# segment_gemm tile order (C1 side finding, docs/validation/segment-gemm-order.md): a row tile's column tiles run
# together when A exceeds the L2 (tile_rule.columns_together). Local RTX 5060 Ti (sm_120, WSL2):
#   python3 tools/ci/local_gpu.py run --purpose "fm sgo JOB" -- env FM=<pinned branch copy> \
#     bash results/segment-gemm-order-20261030/local.sh ROOT JOB...
# Jobs:
#   gputest: tests/gpu/test_segment_gemm_cuda.py and test_tf32x3_gemm_cuda.py (bitwise against the row order);
#   gate: tools/force_stress_gate.py, the default plan (NaCl NbCl5 NaNbO3);
#   ab-<cell>-<order>: tools/ab_rewrites.py, a = TILE_ORDER "rows" (the order before), b = the default ("auto");
#     cell nacl, nbcl5, nanbo3 (3x3x3, 12 blocks of 300 steps), nbcl5big (NbCl5 4x4x8, 9,216 atoms) or nanbo3big
#     (NaNbO3 6x6x6, 8,640 atoms; 8 blocks of 100 steps after 100 warm-up steps); order ab (abab) or ba (abba).
# Before each job: no WSL compute process and utilisation <= 10 % in both nvidia-smi views for 30 s.
# Template: results/cap-t08-chunks-20261030/local.sh.
set -u -o pipefail
ROOT=$1; shift
FM=${FM:?set FM to a pinned source copy}
B=$HOME/fastmace-baseline/1a-T18
PY=$HOME/environments/mace_cueq_oeq/bin/python
REF=$HOME/environments/mace_cueq_latest/bin/python
CACHE=$HOME/fastmace-cache/segment-gemm-order-local
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
note "fastmace=$(cat "$FM/.source_commit") host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
for job in "$@"; do
  case $job in
    gputest) (cd "$FM" && PYTHONPATH=$FM run $job "$REF" -m pytest -q -p no:cacheprovider \
                tests/gpu/test_segment_gemm_cuda.py tests/gpu/test_tf32x3_gemm_cuda.py) ;;
    gate) PYTHONPATH=$FM run $job "$PY" "$FM/tools/force_stress_gate.py" --output "$ROOT/gate" --engine fastmace \
              --checkpoint "$CK" --model mh1-r2scan --structures NaCl NbCl5 NaNbO3 --structures-root "$B" \
              --reference-command "$REF" --production-command "$PY" ;;
    ab-*) IFS=- read -r _ cell order <<< "$job"
      pattern=$([ "$order" = ab ] && echo abab || echo abba)
      extra=()
      case $cell in
        nacl) structure=NaCl ;;
        nbcl5) structure=NbCl5 ;;
        nanbo3) structure=NaNbO3 ;;
        nbcl5big) structure=NbCl5@4x4x8; extra=(--warmup 100 --block 100 --blocks 8) ;;
        nanbo3big) structure=NaNbO3@6x6x6; extra=(--warmup 100 --block 100 --blocks 8) ;;
      esac
      PYTHONPATH=$FM run "$job" "$PY" "$FM/tools/ab_rewrites.py" --output "$ROOT/ab/$job" --checkpoint "$CK" \
          --model mh1-r2scan --structures "$structure" \
          --a-constant "fastmace.kernels.triton.tile_rule:TILE_ORDER='rows'" \
          --order "$order" --pattern "$pattern" --prepare "${extra[@]}" ;;
    *) note "unknown job $job"; exit 2 ;;
  esac
done
note done
