#!/usr/bin/env bash
# Fastest Benchmark_loup time on the A100 (owner, 4 November 2026: "test on the a100 in modal and try to get the
# fastest time yet on SrTiO3 5000 atoms"), after the kernel-lab switches were screened on the RTX 4000 Ada and the
# RTX 5060 Ti (results/omat0-kernel-lab-20261103: radial tables at degree 5 0.908 / 0.935 there).
# 1. Screen (omat0_a100_launch_screen.sh): single switches in one order (SCREEN_ONE), the candidate sets in both
#    orders (CANDIDATES; deg5: DEGREE 5; dz5: and SKIP_ZERO_COLUMN; allns5: and STREAM_STORE_W too; all5, with
#    SLOPE_VECTOR 2 as well, is not a default candidate: 1.027 and 1.070 on top of degree 5 on the Ada and 5060 Ti).
# 2. The candidate with the lowest mean b/a over both orders (if below 1) and the default plan under the benchmark's
#    protocol (tools/bench/loup_md.py: 5 warm-up and 20 timed steps, 3 fresh processes, NVE and Langevin).
# 3. The step breakdown (tools/step_breakdown.py) with that set, then the kernel lab on this GPU (LAB=0 skips it):
#    the next targets, and the size of the 3xTF32 tensor-core forward idea.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 85 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_fastest.sh" --out ~/fm-runs/omat0-a100-fastest
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
rd=fastmace.rewrites.radial
wp=fastmace.kernels.oeq_streaming.wprime_backward
ff=fastmace.kernels.oeq_streaming.fused_forward
declare -A sets=(
  [deg5]="$rd:DEGREE=5"
  [all5]="$rd:DEGREE=5 $wp:SLOPE_VECTOR=2 $ff:SKIP_ZERO_COLUMN=True $ff:STREAM_STORE_W=True"
  [dz5]="$rd:DEGREE=5 $ff:SKIP_ZERO_COLUMN=True"
  [allns5]="$rd:DEGREE=5 $ff:SKIP_ZERO_COLUMN=True $ff:STREAM_STORE_W=True"
)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
SCREEN=${SCREEN_ONE:-"deg4 deg6 skipzero slopes2 sympairs d5-slopes2 d5-skipzero d5-stcs"} \
  bash tools/modal/omat0_a100_launch_screen.sh
SCREEN_BOTH=1 SCREEN=${CANDIDATES:-"deg5 dz5 allns5"} bash tools/modal/omat0_a100_launch_screen.sh
best="" best_ratio=1
for name in ${CANDIDATES:-deg5 dz5 allns5}; do
  ratio=$(grep -h -o "b/a [0-9.]* valid" "$out/ab-$name-ab.log" "$out/ab-$name-ba.log" 2>/dev/null |
          awk '{s += $2; n++} END {if (n == 2) printf "%.4f", s / n}')
  note "candidate $name mean b/a ${ratio:-missing}"
  [ -n "$ratio" ] && awk -v r="$ratio" -v b="$best_ratio" 'BEGIN {exit !(r < b)}' && best=$name best_ratio=$ratio
done
note "protocol sets: default ${best:-(no candidate below 1)} $best_ratio"
for set in default $best; do
  for ens in nve langevin; do
    for t in 1 2 3; do
      note "start loup-$set-$ens-$t"
      if [ "$set" = default ]; then constants=(); else read -r -a constants <<< "${sets[$set]}"; fi
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" \
        ${constants[@]:+--constant "${constants[@]}"} --output "$out/loup/$set-$ens-$t.json" \
        > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
if [ -n "$best" ]; then read -r -a constants <<< "${sets[$best]}"; else constants=(); fi
note "start breakdown ${best:-default}"  # the step's kernels by category with the fastest set (old copies gone?)
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-medium --structure "$st" \
  --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 --steps 200 ${constants[@]:+--constant "${constants[@]}"} \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$?"
rm -f "$out"/breakdown/trace.json
if [ "${LAB:-1}" = 1 ]; then
  note "start lab"
  python tools/omat0_kernel_lab.py --checkpoint "$ck" --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
  note "end lab exit=$?"
fi
note done
grep -h -E "candidate|end loup" "$log" | cut -c1-260
