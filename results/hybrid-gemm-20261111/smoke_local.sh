#!/usr/bin/env bash
# Local 5060 Ti correctness smoke of the hybrid kernel (sm_120 cubin; errors only, no timing claim). $1: extra args.
cd /mnt/d/Margraf/fm-a100-colab
CUDA_LAUNCH_BLOCKING=1 python3 tools/ci/local_gpu.py run --purpose hybrid-gemm-smoke --wait-minutes 30 -- \
  ~/environments/mace_cueq_oeq/bin/python results/hybrid-gemm-20261111/hybrid_lab.py --no-time \
  --output results/hybrid-gemm-20261111/smoke_5060ti.json "$@" 2>&1 | grep -v -i pynvml
