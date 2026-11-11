#!/bin/bash
# Colab H100 lab of the H200 w' variants (12 Nov 2026); the VM is stopped after the job.
cd /mnt/d/Margraf/FastMACE-wprime-h200
python3 tools/colab/fastmace_colab.py --gpu H100 --commit ${COMMIT:-59456497} --repo /mnt/d/Margraf/FastMACE \
  --models medium-omat-0 --out ~/fastmace-runs/wprime-h200-20261111/${RUN:-lab1} --timeout-min 40 \
  --command 'bash results/wprime-h200-20261111/colab_lab.sh' \
  > /mnt/d/Margraf/FastMACE-wprime-h200/results/wprime-h200-20261111/${RUN:-lab1}.driver.log 2>&1
echo "driver exit $?" >> /mnt/d/Margraf/FastMACE-wprime-h200/results/wprime-h200-20261111/${RUN:-lab1}.driver.log
