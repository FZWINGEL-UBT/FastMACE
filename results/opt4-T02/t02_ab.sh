#!/usr/bin/env bash
# OPT4-T02 timed A/B: POLAR-1-S, default (a) against fx_e3nn_linear + polar_graph_sums (b), same process, 12 blocks.
# Usage: t02_ab.sh ORDER PATTERN   (ab abab | ba abba). Inside a reservation, in the conductor's quiet window.
set -uo pipefail
order=$1 pattern=$2
src=${SRC:-$HOME/fastmace-opt4/src-t02-a1758ad}
out=$HOME/fastmace-opt4/T02/ab/local-$order${TAG:-}
export PYTHONPATH=$src FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=4
cd "$src"; mkdir -p "$(dirname "$out")"
echo "$(date -u +%FT%TZ) ab $order $pattern src $(cat .source_commit)"
timeout 880 ~/environments/mace_cueq_oeq_polar/bin/python tools/ab_rewrites.py --output "$out" \
  --checkpoint $HOME/models/MACE-POLAR-1-S.model --model polar1-s --structures ${STRUCTS:-NaCl NbCl5} --warmup ${WARMUP:-300} --blocks 12 \
  --order "$order" --pattern "$pattern" --b fx_e3nn_linear=on polar_graph_sums=on > "$out.log" 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) exit=$? $(grep -E 'b/a' "$out.log" | tr '\n' ' ')"
