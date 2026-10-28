#!/usr/bin/env bash
# ideas D3: float64 package, transcendental recompute banned vs base, ABBA (separate processes, compile caches off)
cd ~/tmp/fm-ideas
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/home/fzwin/tmp/fm-ideas/src-6970b2c5:/home/fzwin/tmp/fm-ideas/src-6970b2c5/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
P=~/environments/mace_cueq_oeq/bin/python
W=/home/fzwin/tmp/fm-ideas/src-6970b2c5/tools/bench/ideas_d3_norecompute.py
A="--checkpoint $HOME/models/mace-mh-1.model --backend hybrid --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4"
run() { rm -rf p9-$1; D3_MODE=$2 timeout 1200 $P $W $A --output p9-$1 --structure $3 > p9-$1.log 2>&1; echo "$(date -u +%FT%TZ) $1 exit $?" >> progress9.txt; }
run NbCl5-b2 save NbCl5; run NbCl5-a2 base NbCl5
run NaCl-a1 base NaCl; run NaCl-b1 save NaCl; run NaCl-b2 save NaCl; run NaCl-a2 base NaCl
echo P9DONE >> progress9.txt
