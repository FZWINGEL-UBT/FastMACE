#!/usr/bin/env bash
# fp32 idea 1 prototype (local RTX 5060 Ti): FASTMACE_ATOM_CHUNKS on the gated chains, ABCCBA per cell, caches off.
cd ~/tmp/fm-fp32
S=/mnt/d/Margraf/fm-fp32
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=$S:$S/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models \
  TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
P=~/environments/mace_cueq_oeq/bin/python
run() {  # cell chunks tag
  out=ch-$1-c$2-$3
  FASTMACE_ATOM_CHUNKS=$2 timeout 1800 $P $S/tools/step_breakdown.py --structure $1 \
    --checkpoint ~/models/mace-mh-1.model --temperature-K 300 --steps 500 --calls 3 --output $out > $out.log 2>&1
  echo "$(date -u +%FT%TZ) $out exit $?" >> progress-ch.txt
}
for s in NbCl5 NaNbO3; do
  run $s 1 r1; run $s 2 r1; run $s 4 r1; run $s 4 r2; run $s 2 r2; run $s 1 r2
done
echo CHDONE >> progress-ch.txt
