#!/usr/bin/env bash
# H200 w' instruction tuning (12 November 2026, results/wprime-h200-20261111): one H200 job.
#  1. tools/omat0_kernel_lab.py: the base w' and the seven h2-* variants (SKIP_OUTSIDE, HOIST_DX, CHANNEL_VECTOR and
#     their combinations) on the SrTiO3 10x10x10 capture.
#  2. tools/modal/wprime_h200_pick.py: the best valid variant under a lab ratio of 0.99 (none: stop).
#  3. tools/modal/round6_ab256.sh: its MD-step A/B in both orders, SrTiO3 10x10x10 and NbCl5 (OMAT-0 medium).
#  4. A rounding winner (CHANNEL_VECTOR): tools/force_stress_gate.py on SrTiO3 3x3x3 and NaCl.
# Exits 3 unless the GPU is an H200. Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 15 --tarball <tar> \
#     --command "bash tools/modal/wprime_h200_lab_ab.sh" --out ~/fm-runs/wprime-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
case "$gpu" in
  *H200*) ;;
  *) note "expected an H200, got $gpu"; echo "expected an H200, got $gpu"; exit 3 ;;
esac
note "wprime-h200 on $gpu; source $(cat .source_commit 2>/dev/null)"
python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --rounds 21 --output "$out/lab" \
  --variants base h2-skip h2-hoist h2-skip+hoist h2-vec2 h2-skip+vec2 h2-vec4 h2-skip+vec4 \
  > "$out/lab.log" 2>&1 < /dev/null
note "end lab exit=$?"
grep -h -E "^wprime +[A-Za-z0-9+-]+ +[0-9.]+ us" "$out/lab.log" | cut -c1-260 >> "$log"
pick=$(python tools/modal/wprime_h200_pick.py "$out/lab/lab.json" 2>> "$log")
note "pick: $pick"
eval "$pick"
if [ "${NAME:-none}" = none ]; then note "no variant under 0.99: no A/B"; note done; cat "$log"; exit 0; fi
OUT=$out NAME=$NAME B_CONSTANTS="$CONSTANTS" SKIP_GATE=1 bash tools/modal/round6_ab256.sh
if [ "$KIND" = rounding ]; then
  read -r -a bc <<< "$CONSTANTS"
  python tools/force_stress_gate.py --output "$out/gate-$NAME" --engine fastmace --model omat0-medium \
    --checkpoint "$ck" --structures-root "$FASTMACE_STRUCTURES" \
    --structures NaCl results/loup-a100-20261103/SrTiO3.cif@3x3x3 --production-constant "${bc[@]}" \
    > "$out/gate-$NAME.log" 2>&1 < /dev/null
  note "gate-$NAME exit=$? $(grep -h -iE 'verdict|pass|fail' "$out/gate-$NAME.log" | tail -n 4 | tr '\n' ' ' | cut -c1-400)"
fi
note done
cat "$log"
