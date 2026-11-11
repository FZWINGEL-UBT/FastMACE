#!/bin/bash
# Local 5060 Ti smoke of the half2 w' lab (compile, launch, numbers), small sizes.
cd /mnt/d/Margraf/fm-a100-colab
export PYTHONPATH=$PWD
~/environments/mace_cueq_oeq/bin/python results/a100-colab-ideas-20261111/half2_wprime_lab.py \
  --output /home/fzwin/fastmace-runs/a100-colab-ideas-20261111/smoke-half2 --edges 32768 --reps 10
