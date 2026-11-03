#!/usr/bin/env bash
# The chunked conv backward on the A100 (3 November 2026): Benchmark_loup's follow-up found it 1.12 against the radial
# tables alone (A100 80GB PCIe, SrTiO3 5,000 atoms), its folded input gradient in many small chunks at ~2 % occupancy.
# Side a: the A100 default (radial_tables on, chunked_conv_backward off). Side b: chunked_conv_backward on with a
# chunk of CHUNK_L2_SHARE 0.25 (the rule), 1 and 4 of the L2, and unchunked inside the op (UNCHUNKED_L2_MULTIPLE
# large); both build orders each.
#   modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 75 --commit <sha> \
#     --command "bash tools/modal/a100_chunk_size.sh" --out ~/fm-runs/a100-chunk-size
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
rt=fastmace.kernels.oeq_streaming.runtime
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader); source $(cat .source_commit)"
for variant in share0.25 share1 share4 unchunked; do
  case $variant in
    share0.25) c="$rt:CHUNK_L2_SHARE=0.25";;
    share1) c="$rt:CHUNK_L2_SHARE=1.0";;
    share4) c="$rt:CHUNK_L2_SHARE=4.0";;
    unchunked) c="$rt:UNCHUNKED_L2_MULTIPLE=1000000";;
  esac
  for order in ab ba; do
    pattern=$([ "$order" = ab ] && echo abab || echo abba)
    name=ab-$variant-$order
    note "start $name"
    python tools/ab_rewrites.py --output "$out/$name" --checkpoint "$ck" --model omat0-medium --structures "$st" \
      --timestep-fs 1 --a-neighbours skin=0.5 --b-neighbours skin=0.5 --warmup 200 --block 200 --blocks 8 \
      --order "$order" --pattern "$pattern" --a radial_tables=on chunked_conv_backward=off \
      --b radial_tables=on chunked_conv_backward=on --b-constant "$c" > "$out/$name.log" 2>&1 < /dev/null
    note "end $name exit=$? $(grep -h "b/a" "$out/$name.log" | tail -n 1 | cut -c1-300)"
  done
done
note done
grep -h "b/a" "$out"/ab-*.log 2>/dev/null | cut -c1-300
