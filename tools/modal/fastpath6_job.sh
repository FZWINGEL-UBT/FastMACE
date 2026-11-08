#!/usr/bin/env bash
# Round 6 (8 November 2026) against the H200 defaults on main 1bd4f3be: per variant tools/spline_w_check.py (registers,
# local bytes; forces on a rattled copy against the defaults next to the same build's spread) and the MD-step A/B
# a = H200 defaults, b = the variant, both build orders; then all three together, and the step breakdown of the
# combination. Variants (Codex H200 round-6 reports; all bitwise by construction):
#   wse    = wprime_backward.STAGE_EDGES: w' stages its chunk's edges once per block (features, sender, receiver,
#            spline interval);
#   pf2    = spline_forward.PREFETCH_AHEAD 2: the forward hints edge k + 2's first four table rows into L1;
#   joined = compose_skip.JOINED_LAYOUT: the first interaction's sums d-major, the composed paths read in place
#            (no per-path transpose copies, one gradient tensor back).
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), BLOCKS (default 6), BREAKDOWN (default 1).
set -uo pipefail
out=${OUT:?the job sets OUT}
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
ck=${CHECKPOINT:-$FASTMACE_MODELS/mace-omat-0-medium.model}
st=results/loup-a100-20261103/SrTiO3.cif
s=fastmace.kernels.oeq_streaming.spline_forward
w=fastmace.kernels.oeq_streaming.wprime_backward
c=fastmace.kernels.triton.compose_skip
declare -A on=([wse]="$w:STAGE_EDGES=True" [pf2]="$s:PREFETCH_AHEAD=2" [joined]="$c:JOINED_LAYOUT=True")
declare -A off=([wse]="$w:STAGE_EDGES=False" [pf2]="$s:PREFETCH_AHEAD=0" [joined]="$c:JOINED_LAYOUT=False")
names="wse pf2 joined"
note "fastpath6_job on $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"

failed=""
for name in $names; do
  note "start check-$name"
  python tools/spline_w_check.py --checkpoint "$ck" --structure "$st@10x10x10" --output "$out/check-$name.json" \
    --constant ${on[$name]} --rattle 0.05 --sources "$out/sources-$name" --compare-constant ${off[$name]} \
    > "$out/check-$name.log" 2>&1 < /dev/null
  code=$?
  note "end check-$name exit=$code $(grep -hoE '"(compose_skip_in_use|joined_layout_in_use|wprime_stage_edges|forward_prefetch|forward_stage_edges|forward_min_blocks|wprime_stage_sh|forward_registers|forward_local_bytes|registers|local_bytes|max_abs|same_build_max_abs|energy_diff|same_build_energy_diff)": [^,]+' "$out/check-$name.log" | tr -d '\n ' | cut -c1-400)"
  if [ "$code" != 0 ]; then failed="$failed $name"; fi
done


ab() {  # name order constants...
  local name=$1 order=$2; shift 2
  local pattern; pattern=$([ "$order" = ab ] && echo abab || echo abba)
  note "start ab-$name-$order"
  python tools/ab_rewrites.py --output "$out/ab-$name-$order" --checkpoint "$ck" --model omat0-medium \
    --structures "$st@10x10x10" --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 \
    --warmup 200 --block 256 --blocks "${BLOCKS:-6}" --order "$order" --pattern "$pattern" \
    --b-constant "$@" > "$out/ab-$name-$order.log" 2>&1 < /dev/null
  note "end ab-$name-$order exit=$? $(grep -h -E 'b/a' "$out/ab-$name-$order.log" | tail -n 1 | cut -c1-240)"
}

for name in $names; do
  case " $failed " in *" $name "*) note "skip ab-$name: check or gate failed"; continue ;; esac
  for order in ab ba; do ab "$name" "$order" ${on[$name]}; done
done

all=""
for name in $names; do
  case " $failed " in *" $name "*) ;; *) all="$all ${on[$name]}" ;; esac
done
if [ -n "$all" ]; then for order in ab ba; do ab all "$order" $all; done; fi

if [ "${BREAKDOWN:-1}" != 0 ] && [ -n "$all" ]; then
  note "start breakdown"
  python tools/step_breakdown.py --output "$out/breakdown" --checkpoint "$ck" --model omat0-medium     --structure "$st@10x10x10" --temperature-K 300 --timestep-fs 1 --skin 0.5 --equilibrate-steps 100 --steps 200     --constant $all > "$out/breakdown.log" 2>&1 < /dev/null
  note "end breakdown exit=$?"
  rm -f "$out"/breakdown/trace.json
fi
note done
