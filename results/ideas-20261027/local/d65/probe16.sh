#!/usr/bin/env bash
# Decision 65 local check (RTX 5060 Ti): the default float64 plan has D1, D2 and D3 on; its forces against the
# decision-63 package (all three off) and stock, NbCl5 and NaNbO3, one process per side.
cd ~/tmp/fm-ideas
S=/home/fzwin/tmp/fm-ideas/src-d65
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=$S:$S/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models
P=~/environments/mace_cueq_oeq/bin/python
mkdir -p d65 && cd d65
for s in NbCl5 NaNbO3; do
  timeout 1800 $P $S/tools/bench/f64pkg_acc.py d65 $s > acc-$s-d65.log 2>&1
  echo "$(date -u +%FT%TZ) $s d65 exit $?" >> ../progress16.txt
  timeout 1800 $P $S/tools/bench/f64pkg_acc.py pkg63 $s engine.fp64_emulation_mantissa_bits=dynamic \
    rewrite.w_tables=off rewrite.save_transcendentals=off > acc-$s-pkg63.log 2>&1
  echo "$(date -u +%FT%TZ) $s pkg63 exit $?" >> ../progress16.txt
done
echo P16DONE >> ../progress16.txt
