#!/usr/bin/env bash
# The interaction-0 symmetric-contraction bound (tools/sc0_bound.py, 9 November 2026; Codex report v3 proposal 5,
# FUSED_LAYER0_SC) on the H200: the division, cuEq's uniform_1d forward and backward and the division's backward on
# interaction 0's real operands (OMAT-0 medium, SrTiO3 10x10x10) against Triton kernels with only the fused
# kernel's reads and writes. With NCU=1 (ncu_job image) a second process launches the proxy and the floor once
# under Nsight Compute: FP32 instruction counts, pipe use, DRAM bytes and stall reasons per kernel (ncu-metrics.csv).
# Usage (from WSL, Modal client venv, slim tarball):
#   FASTMACE_MODAL_NCU=1 modal run tools/modal/fastmace_modal.py::ncu_job --gpu H200 --timeout-min 20 \
#     --tarball <slim.tar> --command "NCU=1 bash tools/modal/sc0_bound_job.sh" --out <dir>
# Environment: OUT (the job sets it), EXPECT_GPU (default H200), ROUNDS (default 15), NCU (default 0).
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
real=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "$real" > "$out/gpu.txt"
if ! echo "$real" | grep -qE "${EXPECT_GPU:-H200}"; then note "WRONG-GPU $real"; exit 3; fi
note "sc0_bound_job on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); source $(cat .source_commit 2>/dev/null)"
python tools/sc0_bound.py --checkpoint "$ck" --rounds "${ROUNDS:-15}" --output "$out" > "$out/bound.log" 2>&1 < /dev/null
note "end bound exit=$? $(grep -E '^(proxy|floor|copy)' "$out/bound.log" | tr '\n' ' ' | cut -c1-1500)"
if [ "${NCU:-0}" = 1 ]; then
  metrics=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,launch__registers_per_thread
  metrics+=,smsp__inst_executed.sum,sm__inst_executed.avg.pct_of_peak_sustained_active
  metrics+=,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum
  metrics+=,sm__sass_thread_inst_executed_op_fadd_pred_on.sum
  metrics+=,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
  metrics+=,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
  metrics+=,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  metrics+=,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
  metrics+=,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active
  metrics+=,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
  metrics+=,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  note "start ncu $(ncu --version | tail -n 1)"
  ncu --profile-from-start off --clock-control none --target-processes all --metrics "$metrics" \
    --export "$out/ncu" --force-overwrite \
    python tools/sc0_bound.py --checkpoint "$ck" --profile-once --output "$out/ncu-run" > "$out/ncu.log" 2>&1 < /dev/null
  note "end ncu exit=$?"
  grep -q ERR_NVGPUCTRPERM "$out/ncu.log" && note "ncu: ERR_NVGPUCTRPERM"
  [ -f "$out/ncu.ncu-rep" ] && ncu --import "$out/ncu.ncu-rep" --csv --page raw --metrics "$metrics" \
    > "$out/ncu-metrics.csv" 2>&1
fi
note done
