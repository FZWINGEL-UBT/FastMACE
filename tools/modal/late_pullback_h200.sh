#!/usr/bin/env bash
# w' late pullback on the H200 (12 November 2026, results/wprime-late-pullback-20261112; owner's OK via the conductor):
#  1. tools/omat0_kernel_lab.py: today's H200 default (skip+hoist, "base") against "lp" (LATE_PULLBACK's transform of
#     the captured source; edge-vector comparison), SrTiO3 10x10x10, twice (fresh processes).
#  2. tools/force_stress_gate.py with LATE_PULLBACK=True on NaCl, NbCl5 and SrTiO3 3x3x3; proof counter APPLIED.
#  3. The MD-step A/B in both build orders (b = LATE_PULLBACK on, proof counter), SrTiO3 10x10x10 and NbCl5, as
#     round6_ab256.sh (1 fs, skin 0.5, 200 warm-up, 8 x 256 steps).
# Phases (each job must fit a profile's remaining budget): LABS (default 2; 0 skips), GATE (default 1),
# AB (default "SrTiO3 NbCl5"; "" skips). Exits 3 unless the GPU is an H200. Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu H200 --timeout-min 25 --tarball <slim tar> \
#     --command "bash tools/modal/late_pullback_h200.sh" --out ~/fm-runs/late-pullback-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif
K=fastmace.kernels.oeq_streaming.wprime_backward:LATE_PULLBACK
X=fastmace.kernels.oeq_streaming.wprime_pullback:APPLIED
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
case "$gpu" in
  *H200*) ;;
  *) note "expected an H200, got $gpu"; echo "expected an H200, got $gpu"; exit 3 ;;
esac
note "late-pullback on $gpu; source $(cat .source_commit 2>/dev/null)"
for t in $(seq 1 "${LABS:-2}"); do
  python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --rounds 21 --output "$out/lab-$t" \
    --structure "$st@10x10x10" --variants base lp > "$out/lab-$t.log" 2>&1 < /dev/null
  note "end lab-$t exit=$?"
  grep -h -E "^wprime +(base|lp) " "$out/lab-$t.log" | cut -c1-300 >> "$log"
  grep -h -E "dv |edge.vector" "$out/lab-$t.log" | head -n 4 | cut -c1-300 >> "$log"
done
if [ "${GATE:-1}" = 1 ]; then
python tools/force_stress_gate.py --output "$out/gate-lp" --engine fastmace --model omat0-medium --checkpoint "$ck" \
  --structures-root "$FASTMACE_STRUCTURES" --structures NaCl NbCl5 "$st@3x3x3" \
  --production-constant "$K=True" --production-expect "$X" > "$out/gate-lp.log" 2>&1 < /dev/null
note "gate-lp exit=$? expect failures $(grep -c 'expected counters' "$out/gate-lp.log")"
grep -h -iE "verdict|passed|fail" "$out/gate-lp.log" | tail -n 6 | cut -c1-400 >> "$log"
fi
for name in ${AB-SrTiO3 NbCl5}; do
  structure=$([ "$name" = SrTiO3 ] && echo "$st@10x10x10" || echo "$name")
  tag=$(echo "$structure" | sed 's#.*/##; s#\.cif##; s#@#-#')
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    python tools/ab_rewrites.py --output "$out/ab-lp-$tag-$order" --checkpoint "$ck" --model omat0-medium \
      --structures "$structure" --timestep-fs 1 --temperature-K 300 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
      --warmup 200 --block 256 --blocks 8 --order "$order" --pattern "$pattern" \
      --a-constant "$K=False" --b-constant "$K=True" --b-expect "$X" \
      > "$out/ab-lp-$tag-$order.log" 2>&1 < /dev/null
    note "end ab-lp-$tag-$order exit=$? $(grep -c 'EXPECT FAILED' "$out/ab-lp-$tag-$order.log") expect failures; $(grep -h -E "b/a|Error" "$out/ab-lp-$tag-$order.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
done
note done
cat "$log"
