#!/usr/bin/env bash
# fm-ideas probe 1: float64 emulation mantissa (fixed vs dynamic), forces and step; float32 step breakdown
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> progress.txt; }
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
for r in 1 2; do for s in NbCl5 NaNbO3 NaCl; do
  rm -rf sb-$s-pkg-r$r sb-$s-m56-r$r
  $P $SB --output sb-$s-pkg-r$r --structure $s > sb-$s-pkg-r$r.log 2>&1
  CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56 $P $SB --output sb-$s-m56-r$r --structure $s > sb-$s-m56-r$r.log 2>&1
done; done
log sb64 done
for s in NbCl5 NaNbO3 NaCl; do
  rm -rf sb32-$s
  $P /mnt/d/Margraf/FastMACE/tools/step_breakdown.py --output sb32-$s --checkpoint $HOME/models/mace-mh-1.model --structure $s --temperature-K 300 > sb32-$s.log 2>&1
done
log ALLDONE
