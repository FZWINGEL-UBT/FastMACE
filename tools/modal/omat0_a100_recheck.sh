#!/usr/bin/env bash
# Benchmark_loup on the A100 with the defaults of 4 November 2026 (radial tables of degree 5, the skipped zero column
# on the A100-SXM4-80GB; owner "Yes"), and a screen of the fused folded forward off: the a.W product then runs as a
# separate GEMM (CUTLASS 3xTF32 on sm_80, about 40 TFLOPS) instead of the fused kernel's FP32 SIMT loop, which the
# A100 kernel lab found to be 56 % of the fused forward (results/omat0-kernel-lab-20261103, section 4).
# 1. SCREEN (omat0_a100_launch_screen.sh, default "ffoff"), one order, then both orders if below 0.99.
# 2. The default plan under the benchmark's protocol (tools/bench/loup_md.py: 5 warm-up and 20 timed steps, 3 fresh
#    processes, NVE and Langevin), and with fused_folded_forward=off too if it won both orders.
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_recheck.sh" --out ~/fm-runs/omat0-a100-recheck
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
screen=${SCREEN:-ffoff}
SCREEN=$screen bash tools/modal/omat0_a100_launch_screen.sh
sets="default"
for name in $screen; do
  r=$(grep -h -o "b/a [0-9.]* valid" "$out/ab-$name-ab.log" 2>/dev/null | awk '{print $2}')
  note "screen $name b/a ${r:-missing}"
  if [ -n "$r" ] && awk -v r="$r" 'BEGIN {exit !(r < 0.99)}'; then
    SCREEN_BOTH=1 SCREEN=$name bash tools/modal/omat0_a100_launch_screen.sh
    ba=$(grep -h -o "b/a [0-9.]* valid" "$out/ab-$name-ba.log" 2>/dev/null | awk '{print $2}')
    note "screen $name both orders ab ${r} ba ${ba:-missing}"
    [ -n "$ba" ] && awk -v r="$ba" 'BEGIN {exit !(r < 1)}' && [ "$name" = ffoff ] && sets="$sets ffoff"
  fi
done
for set in $sets; do
  opt=()
  [ "$set" = ffoff ] && opt=(--rewrites fused_folded_forward=off)
  for ens in nve langevin; do
    for t in 1 2 3; do
      note "start loup-$set-$ens-$t"
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" "${opt[@]}" \
        --output "$out/loup/$set-$ens-$t.json" > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
note done
grep -h -E "screen|end loup" "$log" | cut -c1-260
