#!/usr/bin/env bash
# What limits each part of the MD step (owner request, 9 November 2026: "profile the steps in depth and find out
# what limits each part of each step", H200 and A100). OMAT-0 medium, Benchmark_loup SrTiO3 10x10x10 (5,000 atoms),
# float32, the GPU's default plan, 300 K, 1 fs.
# 1. main: tools/step_breakdown.py's main pass (wall time, stages and idle time between them, the eager force call's
#    kernel sequence with operators and categories);
# 2. ncu: one eager force call (forward and backward) under Nsight Compute with the limiter metrics: speed of light
#    (SM, memory, DRAM, L2, L1), every pipe's utilisation, issue activity, every warp stall reason per issued
#    instruction, occupancy and its limits, global-load coalescing, shared-memory bank conflicts, branch divergence;
#    raw CSV in ncu_raw.csv, joined with the kernel sequence by step_breakdown --summarize-ncu (ncu_kernels.csv).
# The classification per kernel is done offline (tools/step_limiters.py).
# Modal does not let ncu lock the clocks, so ncu profiles at the running clocks (--clock-control none), and caches
# are not flushed between kernels (--cache-control none: the step's own L2 state).
# Usage (from WSL, Modal client venv):
#   FASTMACE_MODAL_NCU=1 modal run tools/modal/fastmace_modal.py::ncu_job --gpu H200 --timeout-min 40 --tarball <tar> \
#     --command "bash tools/modal/step_limiters.sh" --out ~/fm-runs/step-limiters-h200
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
st=results/loup-a100-20261103/SrTiO3.cif@10x10x10
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$gpu" > "$out/gpu.txt"
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); source $(cat .source_commit 2>/dev/null); $(ncu --version | tail -n 1)"
common=(--checkpoint "$ck" --model omat0-medium --structure "$st" --temperature-K 300 --timestep-fs 1)
python tools/step_breakdown.py --output "$out/main" "${common[@]}" --equilibrate-steps 100 --steps 300 --calls 5 \
  > "$out/main.log" 2>&1 < /dev/null
note "end main exit=$? $(tail -n 1 "$out/main.log" | cut -c1-300)"
rm -f "$out/main/trace.json"
metrics="$(python tools/step_breakdown.py --ncu-metrics)"
metrics+=,sm__cycles_active.avg,sm__cycles_elapsed.avg,smsp__issue_active.avg.pct_of_peak_sustained_active
metrics+=,smsp__inst_executed.sum,smsp__thread_inst_executed_per_inst_executed.ratio
metrics+=,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed
metrics+=,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct
metrics+=,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
metrics+=,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps
metrics+=,launch__occupancy_limit_blocks,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic
metrics+=,"regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio"
metrics+=,"regex:sm__inst_executed_pipe_.*\.avg\.pct_of_peak_sustained_active"
metrics+=,"regex:sm__pipe_.*_cycles_active\.avg\.pct_of_peak_sustained_active"
note "start ncu"
ncu --profile-from-start off --target-processes all --cache-control none --clock-control none \
  --metrics "$metrics" --export "$out/ncu" --force-overwrite \
  python tools/step_breakdown.py --pass ncu --output "$out/main" "${common[@]}" --equilibrate-steps 100 \
  > "$out/ncu.log" 2>&1 < /dev/null
note "end ncu exit=$? $(grep -c '==PROF== Profiling' "$out/ncu.log") kernels"
grep -q ERR_NVGPUCTRPERM "$out/ncu.log" && note "ERR_NVGPUCTRPERM, no access to the performance counters"
if [ -f "$out/ncu.ncu-rep" ]; then
  ncu --import "$out/ncu.ncu-rep" --csv --page raw --print-units base > "$out/ncu_raw.csv" 2>&1
  python tools/step_breakdown.py --output "$out/main" --summarize-ncu "$out/ncu_raw.csv" > "$out/join.log" 2>&1
  note "join exit=$? $(tail -n 1 "$out/join.log" | cut -c1-200)"
fi
note done
