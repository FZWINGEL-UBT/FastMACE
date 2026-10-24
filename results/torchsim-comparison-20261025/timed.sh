#!/usr/bin/env bash
# FastMACE (main 5a3166c) vs TorchSim 0.6.2, local RTX 5060 Ti, one quiet reservation. Conductor, 25 Oct 2026.
B=/mnt/d/Margraf/conductor-tools/torchsim
SRC=~/fastmace-torchsim-src
OUT=~/fastmace-torchsim-results; mkdir -p $OUT
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule FASTMACE_MODELS=$HOME/models PYTHONDONTWRITEBYTECODE=1
FM=~/environments/mace_cueq_oeq/bin/python; TS=~/environments/torchsim_eval/bin/python
cd $SRC
PYTHONPATH=$SRC $FM tools/ci/local_gpu.py run --quiet --purpose "conductor: FastMACE vs TorchSim TIMED (quiet, ~90 min)" \
  --wait-minutes 600 -- bash -c "
for round in 1 2; do
 for label in mh1-r2scan omat0-medium; do for s in NaCl NbCl5; do
  PYTHONPATH=$SRC $FM $B/bench_fastmace.py \$label \$s 300 1000 3 2>&1 | grep RESULT >> $OUT/results.jsonl
  for c in none default reduce-overhead; do
   timeout 1800 $TS $B/bench_torchsim.py \$label \$s 50 200 3 cueq \$c 2>&1 | grep -E 'RESULT' >> $OUT/results.jsonl || echo \"FAIL torchsim \$label \$s \$c\" >> $OUT/results.jsonl
  done
 done; done
done
for c in none default; do timeout 1800 $TS $B/bench_torchsim.py omat0-medium NaCl 50 200 3 cueq \$c 8 2>&1 | grep RESULT >> $OUT/results.jsonl; done
echo DONE >> $OUT/results.jsonl"
