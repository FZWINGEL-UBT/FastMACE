#!/usr/bin/env bash
# Round 2 follow-ups (9 November 2026), in the Nsight Compute image:
# 1. tools/fma_dual_pipe_bench.py: FFMA and DFMA chains at FP32:FP64 ratios (idea 1's gate on the A100: a 2:1 mix
#    must reach 85 FMA results per clock per SM, sustained);
# 2. Nsight Compute on today's w' (OMAT-0 medium, SrTiO3 10x10x10, the kernel lab's capture with this GPU's plan):
#    every stall reason per issued instruction and every pipe's utilisation. Answers: is w' FP32-pipe bound (idea 1's
#    second gate, FMA pipe active above 75 %), and are its stalls dependency waits (wait, short scoreboard,
#    math-pipe throttle) or memory (long scoreboard, MIO)?
# Modal does not let ncu lock clocks, so it profiles at the running clocks (--clock-control none).
# Usage (from WSL, Modal client venv):
#   FASTMACE_MODAL_NCU=1 modal run tools/modal/fastmace_modal.py::ncu_job --gpu A100-80GB --timeout-min 25 \
#     --tarball <slim tar> --command "bash tools/modal/ncu_wprime_dualpipe.sh" --out <dir>
# Environment: PARTS (default "bench ncu"), SECONDS_PER_MIX (default 10).
set -uo pipefail
out=${OUT:?the job sets OUT}
ck=$FASTMACE_MODELS/mace-omat-0-medium.model
log=$out/run.log
note() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 > "$out/gpu.txt"
note "Modal $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader); source $(cat .source_commit 2>/dev/null); $(ncu --version 2>/dev/null | tail -n 1)"
for part in ${PARTS:-bench ncu}; do
  case $part in
    bench)
      python tools/fma_dual_pipe_bench.py --output "$out/bench" --seconds "${SECONDS_PER_MIX:-10}" \
        > "$out/bench.log" 2>&1 < /dev/null
      note "end bench exit=$? $(grep -h '^mix' "$out/bench.log" | tr '\n' ' ')" ;;
    ncu)
      metrics="regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio"
      metrics+=",regex:sm__inst_executed_pipe_.*\.avg\.pct_of_peak_sustained_active"
      metrics+=",regex:sm__pipe_.*_cycles_active\.avg\.pct_of_peak_sustained_active"
      metrics+=",smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active"
      metrics+=",sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread"
      metrics+=",l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
      metrics+=",sm__cycles_elapsed.max,gpu__time_duration.sum"
      note "start ncu"
      ncu --profile-from-start off --clock-control none --kernel-name oeq_projected_backward_wprime \
        --target-processes all --metrics "$metrics" --export "$out/wprime" --force-overwrite \
        python tools/omat0_kernel_lab.py --checkpoint "$ck" --kernels wprime --variants base --rounds 3 --warmup 1 \
        --equilibrate-steps 100 --profile-variants base --output "$out/lab" > "$out/ncu.log" 2>&1 < /dev/null
      note "end ncu exit=$?"
      grep -q ERR_NVGPUCTRPERM "$out/ncu.log" && note "ncu: ERR_NVGPUCTRPERM"
      [ -f "$out/wprime.ncu-rep" ] && ncu --import "$out/wprime.ncu-rep" --csv --page raw > "$out/ncu-wprime.csv" 2>&1
      grep -h -E "^wprime " "$out/ncu.log" | cut -c1-240 >> "$log" ;;
  esac
done
note done
