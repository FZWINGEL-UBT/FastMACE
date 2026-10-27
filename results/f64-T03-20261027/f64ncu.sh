#!/usr/bin/env bash
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-f64b PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
M=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fp64.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,lts__t_bytes.sum
/usr/local/cuda-13.3/bin/ncu --profile-from-start off --clock-control none -k "regex:oeq_projected|segmented_polynomial|sigmoid|radial_table|silu|tanh|exp" --metrics $M --csv ~/environments/mace_cueq_oeq/bin/python f64_ncu_probe.py NbCl5 > f64ncu.csv 2> f64ncu.err
echo ALLDONE >> f64ncu.csv
