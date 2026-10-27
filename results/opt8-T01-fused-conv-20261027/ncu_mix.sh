#!/usr/bin/env bash
# usage: ncu_mix.sh OUT STRUCTURE CONFIG
cd ~/tmp/opt8
source ./env.sh
M=gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fmaheavy.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_cbu.sum,smsp__inst_executed_pipe_uniform.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_red.sum,smsp__inst_executed_op_global_atom.sum,smsp__inst_executed_op_local_ld.sum,smsp__inst_executed_op_local_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__inst_executed_pipe_xu.sum
/usr/local/cuda-13.3/bin/ncu --profile-from-start off --cache-control all --clock-control none -k regex:fused_ \
  --metrics $M $P ncu_fused.py ops-mh1-r2scan-$2.pt "$3" > $1 2>&1
echo done
