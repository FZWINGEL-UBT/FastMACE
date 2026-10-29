#!/usr/bin/env bash
# Nsight Compute: our edge-owned backward against LunaTP's default backward, NaCl and NbCl5 (MH-1 operands).
cd ~/tmp/opt7
M=gpu__time_duration.sum,lts__t_bytes.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__waves_per_multiprocessor,launch__grid_size,launch__block_size,smsp__inst_executed.sum,lts__t_requests_srcunit_tex_op_red.sum
rm -f ncu_cmp.txt
for f in ops-mh1-r2scan-NaCl.pt ops-mh1-r2scan-NbCl5.pt; do
  echo "== ours $f" >> ncu_cmp.txt
  PYTHONPATH=/mnt/d/Margraf/fm-f64:. PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.3/bin/ncu --profile-from-start off \
    --cache-control all --clock-control none --metrics $M --csv ~/environments/mace_cueq_oeq/bin/python ncu_probe.py $f \
    2>/dev/null | grep oeq_projected_backward >> ncu_cmp.txt
  echo "== luna $f" >> ncu_cmp.txt
  PATH=$HOME/environments/lunatp_screen/bin:/usr/local/cuda-13.3/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda-13.3 \
    TORCH_EXTENSIONS_DIR=$HOME/tmp/torch_ext_luna PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /usr/local/cuda-13.3/bin/ncu \
    --profile-from-start off --cache-control all --clock-control none --metrics $M --csv \
    ~/environments/lunatp_screen/bin/python ncu_luna.py $f 2>/dev/null | grep -v "^==" | grep -v '"ID"' >> ncu_cmp.txt
done
echo ALLDONE >> ncu_cmp.txt
