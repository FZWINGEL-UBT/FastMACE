#!/usr/bin/env bash
# Fastest Benchmark_loup time on the A100 (owner, 4 November 2026: "test on the a100 in modal and try to get the
# fastest time yet on SrTiO3 5000 atoms"; "try out all of the ideas"), after the kernel-lab switches were screened on
# the RTX 4000 Ada and the RTX 5060 Ti (results/omat0-kernel-lab-20261103: radial tables at degree 5 0.908 / 0.935).
# 0. The fused forward's GPU test body on this GPU: TENSOR_CORES (3xTF32 a.W) against float64 on sm_80.
# 1. Screen (omat0_a100_launch_screen.sh): single switches in one order (SCREEN_ONE), the candidate sets in both
#    orders (CANDIDATES; deg5: DEGREE 5; dz5: and SKIP_ZERO_COLUMN; dztc5: and TENSOR_CORES; dztcre5: and
#    recompute_conv_weights=on, w rebuilt on tensor cores in the backward). Paired slope loads are not a candidate:
#    1.027 and 1.070 on top of degree 5 on the Ada and 5060 Ti.
# 2. The candidate with the lowest mean b/a over both orders (if below 1) and the default plan under the benchmark's
#    protocol (tools/bench/loup_md.py: 5 warm-up and 20 timed steps, 3 fresh processes, NVE and Langevin).
# 3. The step breakdown (tools/step_breakdown.py) with that set, then the kernel lab on this GPU (LAB=0 skips it).
# Usage (from WSL, Modal client venv):
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 90 --commit <sha> \
#     --command "bash tools/modal/omat0_a100_fastest.sh" --out ~/fm-runs/omat0-a100-fastest
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
rd=fastmace.rewrites.radial
ff=fastmace.kernels.oeq_streaming.fused_forward
declare -A sets=(  # module constants of each candidate
  [deg5]="$rd:DEGREE=5"
  [dz5]="$rd:DEGREE=5 $ff:SKIP_ZERO_COLUMN=True"
  [dztc5]="$rd:DEGREE=5 $ff:SKIP_ZERO_COLUMN=True $ff:TENSOR_CORES=True"
  [dztcre5]="$rd:DEGREE=5 $ff:SKIP_ZERO_COLUMN=True $ff:TENSOR_CORES=True"
)
declare -A fields=(  # RewriteConfig fields of each candidate
  [dztcre5]="recompute_conv_weights=on"
)
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "start tc-test"
python - > "$out/tc-test.log" 2>&1 <<"EOF"
import ast, textwrap
tree = ast.parse(open("tests/gpu/test_fused_folded_forward_cuda.py").read())
child = next(n.value.value for n in tree.body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CHILD")
exec(compile(textwrap.dedent(child), "test_fused_folded_forward_cuda.CHILD", "exec"), {"__name__": "__main__"})
EOF
note "end tc-test exit=$? $(tail -n 1 "$out/tc-test.log" | cut -c1-400)"
SCREEN=${SCREEN_ONE:-"tc tcre deg4 deg6 skipzero d5-tc d5-skipzero d5-slopes2"} bash tools/modal/omat0_a100_launch_screen.sh
candidates=${CANDIDATES:-"deg5 dz5 dztc5 dztcre5"}
SCREEN_BOTH=1 SCREEN=$candidates bash tools/modal/omat0_a100_launch_screen.sh
best="" best_ratio=1
for name in $candidates; do
  ratio=$(grep -h -o "b/a [0-9.]* valid" "$out/ab-$name-ab.log" "$out/ab-$name-ba.log" 2>/dev/null |
          awk '{s += $2; n++} END {if (n == 2) printf "%.4f", s / n}')
  note "candidate $name mean b/a ${ratio:-missing}"
  [ -n "$ratio" ] && awk -v r="$ratio" -v b="$best_ratio" 'BEGIN {exit !(r < b)}' && best=$name best_ratio=$ratio
done
note "protocol sets: default ${best:-(no candidate below 1)} $best_ratio"
options() {  # the loup_md / step_breakdown options of a set
  local set=$1 c r
  [ "$set" = default ] && return
  read -r -a c <<< "${sets[$set]:-}"
  read -r -a r <<< "${fields[$set]:-}"
  [ ${#c[@]} -gt 0 ] && echo "--constant ${c[*]}"
  [ ${#r[@]} -gt 0 ] && echo "--rewrites ${r[*]}"
}
for set in default $best; do
  for ens in nve langevin; do
    for t in 1 2 3; do
      note "start loup-$set-$ens-$t"
      # shellcheck disable=SC2046
      python tools/bench/loup_md.py --structure "$st" --checkpoint "$ck" --ensemble "$ens" $(options "$set") \
        --output "$out/loup/$set-$ens-$t.json" > "$out/loup-$set-$ens-$t.log" 2>&1 < /dev/null
      note "end loup-$set-$ens-$t exit=$? $(tail -n 1 "$out/loup-$set-$ens-$t.log" | cut -c1-300)"
    done
  done
done
note "start breakdown ${best:-default}"
# shellcheck disable=SC2046
python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-medium --structure "$st" \
  --temperature-K 300 --timestep-fs 1 --equilibrate-steps 100 --steps 200 $(options "${best:-default}") \
  > "$out/breakdown.log" 2>&1 < /dev/null
note "end breakdown exit=$?"
rm -f "$out"/breakdown/trace.json
if [ "${LAB:-1}" = 1 ]; then
  note "start lab"
  python tools/omat0_kernel_lab.py --checkpoint "$ck" --output "$out/lab" > "$out/lab.log" 2>&1 < /dev/null
  note "end lab exit=$?"
fi
note done
grep -h -E "tc-test|candidate|end loup" "$log" | cut -c1-260
