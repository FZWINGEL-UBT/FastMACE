#!/bin/bash
# Launch colab_ab.sh on a Colab A100 (WSL; run detached). $1: the FastMACE commit.
cd /mnt/d/Margraf/fm-a100-colab
L=/mnt/d/Margraf/fm-a100-colab/results/half-far-wprime-20261111/ab.driver.log
python3 tools/colab/fastmace_colab.py --gpu A100 --commit "$1" --repo /mnt/d/Margraf/FastMACE --models medium-omat-0 \
  --out ~/fastmace-runs/half-far-wprime-20261111/ab --timeout-min 240 \
  --command 'bash results/half-far-wprime-20261111/colab_ab.sh' > $L 2>&1
echo "driver exit $?" >> $L
