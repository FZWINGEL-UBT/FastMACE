#!/usr/bin/env bash
# 2-T19: per-stage setup, 1b (this checkout) against 1a, one version per process, on an idle GPU.
#   run_setup.sh RUN_ROOT PYTHON CHECKPOINT RENAMED_DIR STRUCTURES_DIR
# Warm: one shared cache root, filled by one process per side first, then 1a 1b 1b 1a 1a 1b.
# Cold: a fresh cache root per process, 1a 1b 1b 1a.
# Gate: --path simulation, NaCl, check gate and skip, warm cache; MACE-first: the same with MACE imported first.
set -euo pipefail
RUN_ROOT=$1 PY=$2 CKPT=$3 RENAMED=$4 STRUCT=$5
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_STRUCTURES=$STRUCT
export FASTMACE_MODELS=${FASTMACE_MODELS:-$HOME/models}
mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"

caches() {  # every cache under one root
  local root=$1
  export FASTMACE_CACHE=$root TORCHINDUCTOR_CACHE_DIR=$root/inductor TRITON_CACHE_DIR=$root/triton \
         MACE_COMPILE_CACHE=$root/compile XDG_CACHE_HOME=$root/xdg TMPDIR=$RUN_ROOT/tmp
  mkdir -p "$root" "$RUN_ROOT/tmp"
}

idle() {  # record the machine state before each process
  { date -u +%FT%TZ; uptime; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; free -g | sed -n 2p; } > "$1.idle.txt" 2>&1
}

probe() {  # name, args...
  local name=$1; shift
  idle "$name"
  echo "== $name $(date -u +%T)"
  "$PY" "$@" --output "$RUN_ROOT/$name" --checkpoint "$CKPT" > "$name.log" 2>&1 || { echo "FAILED $name"; tail -20 "$name.log"; }
  tail -3 "$name.log"
}

side() { echo --path ab --side "$1" --renamed "$RENAMED"; }

caches "$RUN_ROOT/cache-warm"
probe warm-fill-1a "$HERE/setup_stages.py" $(side 1a)
probe warm-fill-1b "$HERE/setup_stages.py" $(side 1b)
i=0
for s in 1a 1b 1b 1a 1a 1b; do
  i=$((i + 1))
  probe "warm-$i-$s" "$HERE/setup_stages.py" $(side "$s")
done
probe sim-gate-1 "$HERE/setup_stages.py" --path simulation --check gate --structures NaCl
probe sim-skip-1 "$HERE/setup_stages.py" --path simulation --check skip --structures NaCl
probe mace-first-gate-fill "$HERE/mace_first.py" --path simulation --check gate --structures NaCl
probe mace-first-gate-1 "$HERE/mace_first.py" --path simulation --check gate --structures NaCl
probe sim-gate-2 "$HERE/setup_stages.py" --path simulation --check gate --structures NaCl
i=0
for s in 1a 1b 1b 1a; do
  i=$((i + 1))
  caches "$RUN_ROOT/cache-cold-$i"
  probe "cold-$i-$s" "$HERE/setup_stages.py" $(side "$s")
done
echo "done $(date -u +%T)"
