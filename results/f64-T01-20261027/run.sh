#!/usr/bin/env bash
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/FastMACE PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
for s in NaCl NbCl5; do for b in hybrid cueq; do
  ~/environments/mace_cueq_oeq/bin/python sb64.py --output out-$s-$b-f64 --checkpoint ~/models/mace-mh-1.model --structure $s --backend $b --dtype float64 --equilibrate-steps 30 --steps 60 --calls 4 > log-$s-$b.txt 2>&1
  echo "$s $b exit=$?" >> done.txt
done; done
echo ALL >> done.txt
