#!/usr/bin/env bash
# Nsight Compute on the fused forward (H200 ideas plan, revision 7, "Next measurements" item 5; 6 November 2026).
# OMAT-0 medium, SrTiO3 10x10x10 (the kernel lab's capture). The lab times base and the variants, then launches each
# once between cudaProfilerStart/Stop (--profile-variants); ncu (--profile-from-start off) profiles only those
# launches, in the order the lab prints ("forward profiled launch i: NAME"): ID 0 is base.
# - pass 1, the plan's metrics: warp stall reasons (no instruction, MIO throttle, short and long scoreboard,
#   math-pipe throttle, per issued instruction), shared-memory wavefronts and bank conflicts, the tensor pipe's
#   active cycles; raw CSV in ncu-metrics.csv;
# - pass 2, ncu's sections (speed of light, warp state, scheduler, memory tables, compute, occupancy, launch,
#   instruction statistics) in ncu-sections.txt.
# Each pass has its own report (.ncu-rep, open with ncu-ui). If the container may not read the performance counters,
# ncu says ERR_NVGPUCTRPERM; run.log records it.
# Modal does not allow ncu to lock the clocks (LockingClocksFailed, 6 November 2026), so it profiles at the
# running clocks (--clock-control none); compare stall shares, not absolute times.
# Usage (from WSL, Modal client venv; the image is built on first use, about 1.5 GB more than the job image):
#   FASTMACE_MODAL_NCU=1 modal run tools/modal/fastmace_modal.py::ncu_job --gpu H200 --timeout-min 40 --commit <sha> \
#     --command "bash tools/modal/ncu_forward.sh" --out ~/fm-runs/ncu-forward-h200
# Environment: VARIANTS (default "F-skew-100ns"; space-separated lab variants), ALIAS (a plan GPU's name, e.g.
# "NVIDIA A100-SXM4-80GB" for the A100 plan on an H200; default none, the GPU's own plan), CONSTANTS (lab --constant
# words), EQUILIBRATE (lab --equilibrate-steps, default 300), SECTIONS=0 (skip pass 2).
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [ -n "${ALIAS:-}" ]; then
  export PYTHONPATH=$PWD/tools/bench/gpu_alias:${PYTHONPATH:-} FASTMACE_GPU_ALIAS="$real=$ALIAS"
fi
read -r -a variants <<< "${VARIANTS-F-skew-100ns}"
read -r -a constants <<< "${CONSTANTS:-}"
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
note "Modal $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader); alias ${FASTMACE_GPU_ALIAS:-none}; source $(cat .source_commit 2>/dev/null); $(ncu --version | tail -n 1)"
lab=(python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels forward --rounds 3 --warmup 1
     --equilibrate-steps "${EQUILIBRATE:-300}" --variants base "${variants[@]}" --profile-variants base "${variants[@]}")
[ ${#constants[@]} -eq 0 ] || lab+=(--constant "${constants[@]}")
metrics=smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
metrics+=,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
metrics+=,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
metrics+=,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.sum
metrics+=,smsp__inst_executed.sum,sm__cycles_elapsed.max,gpu__time_duration.sum
run_pass() {  # name, ncu options...
  local name=$1; shift
  note "start $name"
  ncu --profile-from-start off --clock-control none --kernel-name fused_folded_forward --target-processes all "$@" \
    --export "$out/$name" --force-overwrite "${lab[@]}" --output "$out/lab-$name" > "$out/$name.log" 2>&1 < /dev/null
  local code=$?
  note "end $name exit=$code $(grep -h -E 'profiled launch' "$out/$name.log" | tr '\n' ' ')"
  grep -q ERR_NVGPUCTRPERM "$out/$name.log" && note "$name: ERR_NVGPUCTRPERM, no access to the GPU's performance counters"
  grep -h -E "^forward " "$out/$name.log" | cut -c1-240 >> "$log"
  return 0
}
run_pass metrics --metrics "$metrics"
[ -f "$out/metrics.ncu-rep" ] && ncu --import "$out/metrics.ncu-rep" --csv --page raw --metrics "$metrics" \
  > "$out/ncu-metrics.csv" 2>&1
if [ "${SECTIONS:-1}" != 0 ]; then
  run_pass sections --section SpeedOfLight --section WarpStateStats --section SchedulerStats \
    --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section InstructionStats
  [ -f "$out/sections.ncu-rep" ] && ncu --import "$out/sections.ncu-rep" --page details > "$out/ncu-sections.txt" 2>&1
fi
[ -f "$out/ncu-metrics.csv" ] && cat "$out/ncu-metrics.csv" >> "$log"
note done
