#!/usr/bin/env bash
# fm-ideas probe 4: D1 step A/B (one alternating pair per cell, 15 min timeout per process), then wrank (CPU, 4 threads)
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE:/mnt/d/Margraf/FastMACE/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> progress.txt; }
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
for s in NbCl5 NaNbO3 NaCl; do
  rm -rf p4-$s-pkg p4-$s-m56
  timeout 900 $P $SB --output p4-$s-pkg --structure $s > p4-$s-pkg.log 2>&1; log "p4 $s pkg exit $?"
  CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 timeout 900 $P $SB --output p4-$s-m56 --structure $s > p4-$s-m56.log 2>&1; log "p4 $s m56 exit $?"
done
for s in NbCl5 NaNbO3; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= timeout 1500 $P -c "import torch; torch.set_num_threads(4); import runpy, sys; sys.argv=[\"wrank.py\",\"$s\"]; runpy.run_path(\"wrank.py\", run_name=\"__main__\")" > wrank-$s.txt 2>&1; log "wrank $s exit $?"
done
log P4DONE
