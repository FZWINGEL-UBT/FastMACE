#!/usr/bin/env bash
# OPT4-T02 module profile: POLAR-1-S, NaCl and NbCl5, hybrid float32. Inside a local_gpu.py reservation.
set -uo pipefail
src=${SRC:-$HOME/fastmace-opt4/src-t02-probe}
out=${OUT:-$HOME/fastmace-opt4/T02/profile}
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models
export FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4 TORCH_LOGS=graph_breaks,recompiles
py=$HOME/environments/mace_cueq_oeq_polar/bin/python
cd "$src"; mkdir -p "$out"
for s in ${STRUCTS:-NaCl NbCl5}; do
  echo "$(date -u +%FT%TZ) $s"
  timeout 400 "$py" tools/module_profile.py --output "$out/$s${TAG:-}" --checkpoint $HOME/models/MACE-POLAR-1-S.model \
    --structure "$s" ${EXTRA:-} > "$out/$s${TAG:-}.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) exit=$? $(tail -1 "$out/$s${TAG:-}.log" | cut -c1-300)"
done
