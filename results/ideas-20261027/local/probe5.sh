#!/usr/bin/env bash
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE:/mnt/d/Margraf/FastMACE/tools PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
for s in NaNbO3 NbCl5; do timeout 900 ~/environments/mace_cueq_oeq/bin/python -c "import torch; torch.set_num_threads(4); import runpy, sys; sys.argv=[\"wrank.py\",\"$s\"]; runpy.run_path(\"wrank.py\", run_name=\"__main__\")" > wrank-$s.txt 2>&1; echo "wrank $s exit $?" >> progress.txt; done
