#!/usr/bin/env bash
# Local 5060 Ti correctness smoke of the hybrid kernel (sm_120 cubin; errors only, no timing claim).
cd /mnt/d/Margraf/fm-a100-colab
python3 tools/ci/local_gpu.py run --purpose hybrid-gemm-smoke --wait-minutes 120 -- \
  ~/environments/mace_cueq_oeq/bin/python results/hybrid-gemm-20261111/hybrid_lab.py --no-time \
  --output results/hybrid-gemm-20261111/smoke_5060ti.json > results/hybrid-gemm-20261111/smoke_5060ti.log 2>&1
echo "exit $?" >> results/hybrid-gemm-20261111/smoke_5060ti.log
