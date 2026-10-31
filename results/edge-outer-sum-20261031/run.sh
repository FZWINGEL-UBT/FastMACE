#!/usr/bin/env bash
# edge_outer_sum backward's superlinear growth (conductor, 31 October 2026): Nsight Compute on the kernels named
# *edge_outer_sum* (the custom backward and Inductor's pointwise consumer of its output) in one eager force call, at
# NbCl5 3x3x3 (1,944 atoms) and 4x4x8 (9,216), local RTX 5060 Ti. Holding the local GPU reservation:
#   python3 tools/ci/local_gpu.py run --purpose "fm edge_outer_sum profile" -- env FM=<source> bash run.sh ROOT
set -u -o pipefail
ROOT=$1
FM=${FM:?set FM to a source copy}
CACHE=$HOME/fastmace-cache/l2-gap-audit
mkdir -p "$ROOT/tmp"
export PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 FASTMACE_MODELS=$HOME/models FASTMACE_STRUCTURES=$HOME/fastmace-baseline/1a-T18 \
  TMPDIR=$ROOT/tmp TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_CACHE=$CACHE TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  TRITON_CACHE_DIR=$CACHE/triton MACE_COMPILE_CACHE=$CACHE/compile XDG_CACHE_HOME=$CACHE/xdg
PY=$HOME/environments/mace_cueq_oeq/bin/python
TOOL=$FM/tools/step_breakdown.py
NCU=/usr/local/cuda/bin/ncu
METRICS="gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size,launch__registers_per_thread"
for spec in NbCl5-1944:NbCl5 NbCl5-9216:NbCl5@4x4x8; do
  name=${spec%%:*} structure=${spec#*:}
  echo "$(date -u +%FT%TZ) start $name" >> "$ROOT/run.log"
  "$NCU" --profile-from-start off --target-processes all --cache-control none --clock-control none \
    --kernel-name regex:"edge_outer_sum|^kernel$" --metrics "$METRICS" --csv --page raw --print-units base \
    "$PY" "$TOOL" --pass ncu --output "$ROOT/$name" --checkpoint "$HOME/models/mace-mh-1.model" --model mh1-r2scan \
    --backend hybrid --temperature-K 1000 --timestep-fs 1 --equilibrate-steps 100 --structure "$structure" \
    > "$ROOT/$name.csv" 2> "$ROOT/$name.log"
  echo "$(date -u +%FT%TZ) end $name exit=$?" >> "$ROOT/run.log"
done
