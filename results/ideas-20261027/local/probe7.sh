#!/usr/bin/env bash
# ideas F0 ceiling: float32 default plan, folded input gradient stubbed vs base, ABAB separate processes
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-ideas:/mnt/d/Margraf/fm-ideas/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
for s in NbCl5 NaNbO3; do for r in 1 2; do for m in base stub; do
  rm -rf p7-$s-$m-$r
  F0_MODE=$m timeout 900 $P f0_stub.py --output p7-$s-$m-$r --checkpoint $HOME/models/mace-mh-1.model --structure $s --temperature-K 300 --calls 4 > p7-$s-$m-$r.log 2>&1
  echo "$(date -u +%FT%TZ) p7 $s $m $r exit $?" >> progress7.txt
done; done; done
echo P7DONE >> progress7.txt
