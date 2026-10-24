#!/usr/bin/env bash
# OPT4-T02 POLAR gate: tools/force_stress_gate.py --engine fastmace with both records on. Inside a reservation.
# Usage: t02_gate.sh LABEL CHECKPOINT STRUCTURE... (env SRC, TAG)
set -uo pipefail
label=$1 ckpt=$HOME/models/$2; shift 2
src=${SRC:-$HOME/fastmace-opt4/src-t02-a1758ad}
out=$HOME/fastmace-opt4/T02/gate/$label${TAG:-}
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
cd "$src"; mkdir -p "$(dirname "$out")"
echo "$(date -u +%FT%TZ) gate $label $* rewrites=${REWRITES-fx_e3nn_linear=on polar_graph_sums=on}"
timeout 860 ~/environments/mace_cueq_oeq_polar/bin/python tools/force_stress_gate.py --output "$out" --engine fastmace \
  --checkpoint "$ckpt" --model "$label" --structures "$@" --structures-root $HOME/fastmace-baseline/1a-T18 \
  --reference-command ~/environments/mace_cueq_latest/bin/python \
  --production-command ~/environments/mace_cueq_oeq_polar/bin/python \
  --rewrites ${REWRITES-fx_e3nn_linear=on polar_graph_sums=on} > "$out.log" 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) exit=$? $(grep -i 'summary\|pass\|fail' "$out.log" | tail -4 | cut -c1-250)"
