#!/usr/bin/env bash
# f64-T03: float64 package step time, main (768-edge chunks) against fm-f64c (4-L2 chunks), alternating processes.
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
SB="sb64b.py --checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
O=chunk-ab; rm -rf $O; mkdir -p $O
for round in 1 2; do for s in NbCl5 NaNbO3; do
  PYTHONPATH=/mnt/d/Margraf/FastMACE $P $SB --output $O/$s-main-$round --structure $s > $O/$s-main-$round.log 2>&1
  PYTHONPATH=/mnt/d/Margraf/fm-f64c $P $SB --output $O/$s-new-$round --structure $s > $O/$s-new-$round.log 2>&1
done; done
echo ALLDONE > $O/done
