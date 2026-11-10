#!/usr/bin/env bash
# Re-screen of module-constant switches against today's defaults (9 November 2026: BUTTERFLY, rejected on 6 November,
# won 0.986 against the later w' defaults). For each entry of SCREEN an MD-step A/B (tools/modal/round6_ab256.sh,
# both build orders, OMAT-0 medium, SrTiO3 10x10x10 unless STRUCTURES, 300 K, no gate): side a the GPU's defaults,
# side b with the entry's constants. Winners then need the gate and a second structure.
# Environment: OUT (the job sets it), SCREEN ("name|MODULE:NAME=VALUE ...;name|..."), STRUCTURES.
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
note "const_screen on $gpu; source $(cat .source_commit 2>/dev/null)"
IFS=';' read -r -a entries <<< "${SCREEN:?set SCREEN}"
for entry in "${entries[@]}"; do
  name=${entry%%|*}; consts=${entry#*|}
  mkdir -p "$out/$name"
  OUT=$out/$name NAME=$name B_CONSTANTS="$consts" \
    STRUCTURES="${STRUCTURES:-results/loup-a100-20261103/SrTiO3.cif@10x10x10}" bash tools/modal/round6_ab256.sh \
    > "$out/$name.log" 2>&1 < /dev/null
  note "end $name exit=$? $(grep -h 'end ab' "$out/$name/run.log" | grep -o 'b/a [0-9.]*' | tr '\n' ' ') $consts"
done
note done
