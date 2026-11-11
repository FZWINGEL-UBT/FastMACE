#!/bin/bash
cd /mnt/d/Margraf/fm-a100-colab
python3 tools/colab/fastmace_colab.py --gpu A100 --commit 014e6c3f --repo /mnt/d/Margraf/FastMACE --models medium-omat-0 \
  --out ~/fastmace-runs/a100-colab-ideas-20261111/half2 --timeout-min 40 --keep \
  --command 'bash results/a100-colab-ideas-20261111/colab_half2.sh' \
  > /mnt/d/Margraf/fm-a100-colab/results/a100-colab-ideas-20261111/half2.driver.log 2>&1
echo "driver exit $?" >> /mnt/d/Margraf/fm-a100-colab/results/a100-colab-ideas-20261111/half2.driver.log
