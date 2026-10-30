#!/usr/bin/env bash
# Engine setup across compositions (owner request, 30 October 2026): MH-1 matpes_r2scan float32, default plan,
# gpu-LBFGS + FrechetCell, eight Energy_Hull step-2 phases as 3x3x3 supercells, local RTX 5060 Ti (WSL2).
# Each mode inside its own local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "setup across compositions MODE (<15 min)" -- \
#     bash results/setup-across-compositions/run.sh MODE ROOT
# Modes:
#   cold-A / cold-B   each phase alone in a fresh process with its own fresh, empty caches (A: 4 phases, B: 4)
#   warm              each phase alone in a fresh process with its cold run's caches
#   chain-cold        all eight in one process, in order, with one fresh shared cache
#   chain-warm        the same again in a new process (the shared cache now warm)
#   graphs            NaCl, NaCl compressed 0.97 (another edge size), NbCl5, NaNbO3 in one process, warm shared
#                     cache, every CUDA graph dumped as DOT (no timing)
#   graphs-edges      NaCl and NaCl compressed 0.90, NbCl5 and NbCl5 compressed 0.95 (other edge buckets), as graphs
set -u -o pipefail
MODE=$1
ROOT=$2
FM=/mnt/d/Margraf/fm-setup-compositions
S=$FM/results/setup-across-compositions/structures
PY=$HOME/environments/mace_cueq_oeq/bin/python
CKPT=$HOME/models/mace-mh-1.model
mkdir -p "$ROOT/tmp"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models TMPDIR=$ROOT/tmp TMP=$ROOT/tmp \
  TEMP=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4
cache_env() {
  export FASTMACE_CACHE=$1 TORCHINDUCTOR_CACHE_DIR=$1/inductor TRITON_CACHE_DIR=$1/triton \
    MACE_COMPILE_CACHE=$1/compile XDG_CACHE_HOME=$1/xdg
}
declare -A CIF=([NaCl]=mp-22862_NaCl [Na2O]=mp-2352_Na2O [NaNbO3]=mp-4514_NaNbO3 [NbOCl3]=mp-556422_NbCl3O
  [NbCl5]=mp-23307_NbCl5 [TaCl5]=mp-29831_TaCl5 [NaTaCl6]=mp-28179_NaTaCl6 [NaClO4]=mp-22968_NaClO4)
ORDER=(NaCl Na2O NaNbO3 NbOCl3 NbCl5 TaCl5 NaTaCl6 NaClO4)
spec() { echo "$1=$S/${CIF[$1]}.cif${2:-}"; }
run() {  # name label structures...
  local name=$1 label=$2; shift 2
  [ -e "$ROOT/$name" ] && { echo "$ROOT/$name exists; skipped"; return 0; }
  echo "$(date -u +%FT%TZ) start $name (FASTMACE_CACHE=$FASTMACE_CACHE)" | tee -a "$ROOT/run.log"
  echo "  structures: $*" >> "$ROOT/run.log"
  "$PY" -W ignore "$FM/tools/bench/setup_compositions.py" --output "$ROOT/$name" --checkpoint "$CKPT" \
    --label "$label" --structures "$@" ${EXTRA:-} > "$ROOT/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) exit $? $name" | tee -a "$ROOT/run.log"
  grep -v "^USDT\|^\s*$" "$ROOT/$name.log" | tail -3
}
echo "$(date -u +%FT%TZ) mode=$MODE fastmace=$(cat $FM/.source_commit) root=$ROOT" >> "$ROOT/run.log"
case $MODE in
  smoke) cache_env "$ROOT/cache-smoke"; run smoke smoke "$(spec NaCl)" "$(spec Na2O)" ;;
  cold-A|cold-B)
    [ "$MODE" = cold-A ] && set -- NaCl Na2O NaNbO3 NbOCl3 || set -- NbCl5 TaCl5 NaTaCl6 NaClO4
    for p in "$@"; do cache_env "$ROOT/cache-$p"; run "cold-$p" cold "$(spec $p)"; done ;;
  warm) for p in "${ORDER[@]}"; do cache_env "$ROOT/cache-$p"; run "warm-$p" warm "$(spec $p)"; done ;;
  chain-cold|chain-warm)
    cache_env "$ROOT/cache-chain"; specs=(); for p in "${ORDER[@]}"; do specs+=("$(spec $p)"); done
    run "$MODE" "$MODE" "${specs[@]}" ;;
  graphs)
    cache_env "$ROOT/cache-chain"; EXTRA=--graph-dump
    run graphs graphs "$(spec NaCl)" "NaCl-0.97=$S/${CIF[NaCl]}.cif:0.97" "$(spec NbCl5)" "$(spec NaNbO3)" ;;
  graphs-edges)  # two padded edge sizes of one composition: compressed cells have more edges
    cache_env "$ROOT/cache-chain"; EXTRA=--graph-dump
    run graphs-edges graphs-edges "$(spec NaCl)" "NaCl-0.90=$S/${CIF[NaCl]}.cif:0.90" "$(spec NbCl5)"       "NbCl5-0.95=$S/${CIF[NbCl5]}.cif:0.95" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
