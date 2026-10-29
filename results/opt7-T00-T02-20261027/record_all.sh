#!/usr/bin/env bash
cd ~/tmp/opt7
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule FASTMACE_MODELS=$HOME/models MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq PYTHONPATH=/mnt/d/Margraf/fm-f64 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1
P=~/environments/mace_cueq_oeq/bin/python
rm -f record.txt
for m in mh1-r2scan omat0-medium; do for s in NaCl NbCl5 NaNbO3 NbCl3O; do
  [ -f ops-$m-$s.pt ] || $P record.py $m - $s 2>&1 | grep -E "calls|Error" >> record.txt
done; done
echo ALLDONE >> record.txt
