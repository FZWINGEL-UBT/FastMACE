#!/usr/bin/env bash
# Float32 production defaults (fm-f64 = opt6-T01 + float64 work; float32 unaffected): where a step goes, MH-1 with
# and without radial tables, and OMAT-0 medium (generic MACE), on uniform and heterogeneous cells.
cd ~/tmp/f64probe
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=/mnt/d/Margraf/fm-f64 PYTHONDONTWRITEBYTECODE=1 MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
T=/mnt/d/Margraf/fm-f64/tools/step_breakdown.py
rm -rf share-* share.txt
for s in NaCl NbCl5 NaNbO3 NbCl3O; do
  $P $T --output share-mh1-$s --checkpoint $HOME/models/mace-mh-1.model --model mh1-r2scan --structure $s --steps 300 --calls 8 > share-mh1-$s.log 2>&1
  $P $T --output share-mh1tab-$s --checkpoint $HOME/models/mace-mh-1.model --model mh1-r2scan --structure $s --steps 300 --calls 8 --rewrites radial_tables=on > share-mh1tab-$s.log 2>&1
  $P $T --output share-omat-$s --checkpoint $HOME/models/mace-omat-0-medium.model --model omat0-medium --structure $s --steps 300 --calls 8 > share-omat-$s.log 2>&1
  echo "$s done" >> share.txt
done
echo ALLDONE >> share.txt
