#!/usr/bin/env bash
cd ~/tmp/opt7
F=$(ls ops-*.pt | grep -v grads)
rm -f t00.jsonl
for round in 1 2; do
  PYTHONPATH=/mnt/d/Margraf/fm-f64:. PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_oeq/bin/python stream_bench.py $F 2>/dev/null | sed "s/^{/{\"side\": \"streaming\", \"round\": $round, /" >> t00.jsonl
  PATH=$HOME/environments/lunatp_screen/bin:/usr/local/cuda-13.3/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda-13.3 TORCH_EXTENSIONS_DIR=$HOME/tmp/torch_ext_luna PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 ~/environments/lunatp_screen/bin/python luna_bench.py $F 2>/dev/null | sed "s/^{/{\"side\": \"lunatp\", \"round\": $round, /" >> t00.jsonl
done
echo ALLDONE >> t00.jsonl
