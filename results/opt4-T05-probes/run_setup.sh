#!/usr/bin/env bash
# OPT4-T05: fm.md start stages, cold and warm compiler caches, one start per process, untimed local runs.
#   run_setup.sh RUN_ROOT SET    (inside one tools/ci/local_gpu.py reservation per call)
# SET cold-N: a fresh cache root per process (OEq build, Inductor, Triton, compile caches all empty).
# SET fill:   fills the shared warm root, one process per cell.
# SET warm-N: the shared warm root, one process per cell.
# Cells: MH-1 r2SCAN hybrid NaCl and NbCl5, omat0-medium (generic MACE) NaCl; 3x3x3, float32, NVT 1000 K.
set -uo pipefail
RUN_ROOT=$1 SET=$2
SRC=${SRC:-$HOME/fastmace-opt4-t05/src-13a1983}
PY=${PY:-$HOME/environments/mace_cueq_oeq/bin/python}
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
export FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"

caches() {
  local root=$1
  # Inductor and Triton set explicitly, as fastmace.kernels.triton.cache.configure would: on main 13a1983
  # nothing calls it, and Inductor's default is $TMPDIR/torchinductor_$USER, shared by every root (cold-1).
  export FASTMACE_CACHE=$root XDG_CACHE_HOME=$root/xdg TRITON_HOME=$root/triton-home TMPDIR=$root/tmp
  export TORCHINDUCTOR_CACHE_DIR=$root/inductor TRITON_CACHE_DIR=$root/triton
  unset MACE_COMPILE_CACHE
  mkdir -p "$root/tmp"
}

idle() {
  { date -u +%FT%TZ; uptime; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; free -g | sed -n 2p
    ps -eo pcpu,args --sort=-pcpu | head -5; } > "$1.idle.txt" 2>&1
}

probe() {  # name model structure [setup_md.py arguments]
  local name=$1 model=$2 structure=$3; shift 3
  idle "$name"
  echo "== $name $(date -u +%T)"
  timeout 600 "$PY" "$SRC/results/opt4-T05-probes/setup_md.py" --output "$RUN_ROOT/$name" --model "$model" \
    --structure "$structure" --label "$SET" "$@" > "$name.log" 2>&1 < /dev/null || { echo "FAILED $name"; tail -20 "$name.log"; }
  head -1 "$name.log" | cut -c1-200; grep -m1 "^NaCl\|^NbCl5" "$name.log"
}

cells() {  # prefix
  probe "$1-mh1-NaCl" mh1-r2scan NaCl
  probe "$1-mh1-NbCl5" mh1-r2scan NbCl5
  probe "$1-omat0-NaCl" omat0-medium NaCl
}

case $SET in
  cold-dyn-*)  # a fresh root per process, dynamic_edges=True
    for cell in "mh1 mh1-r2scan NaCl" "mh1 mh1-r2scan NbCl5" "omat0 omat0-medium NaCl"; do
      set -- $cell
      caches "$RUN_ROOT/cache-$SET-$1-$3"
      probe "$SET-$1-$3" "$2" "$3" --dynamic-edges
    done ;;
  cold-*)
    for cell in "mh1 mh1-r2scan NaCl" "mh1 mh1-r2scan NbCl5" "omat0 omat0-medium NaCl"; do
      set -- $cell
      caches "$RUN_ROOT/cache-$SET-$1-$3"
      probe "$SET-$1-$3" "$2" "$3"
    done ;;
  fill|warm-*)
    caches "$RUN_ROOT/cache-warm"
    cells "$SET" ;;
  dyn-fill)  # the dynamic_edges=True compile into the warm root
    caches "$RUN_ROOT/cache-warm"
    probe "$SET-mh1-NaCl" mh1-r2scan NaCl --dynamic-edges
    probe "$SET-mh1-NbCl5" mh1-r2scan NbCl5 --dynamic-edges
    probe "$SET-omat0-NaCl" omat0-medium NaCl --dynamic-edges ;;
  ab-*)  # warm root: off on on off per cell (same seed: the final states are compared)
    caches "$RUN_ROOT/cache-warm"
    for cell in "mh1 mh1-r2scan NaCl" "mh1 mh1-r2scan NbCl5" "omat0 omat0-medium NaCl"; do
      set -- $cell
      probe "$SET-$1-$3-off1" "$2" "$3"
      probe "$SET-$1-$3-on1" "$2" "$3" --dynamic-edges
      probe "$SET-$1-$3-on2" "$2" "$3" --dynamic-edges
      probe "$SET-$1-$3-off2" "$2" "$3"
    done ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
echo "done $SET $(date -u +%T)"
