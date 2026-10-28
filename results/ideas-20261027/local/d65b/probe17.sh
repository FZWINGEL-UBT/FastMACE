#!/usr/bin/env bash
# Decision 65: the save_transcendentals partitioner picklable; two NbCl5 d65 runs with a fresh Inductor cache.
cd ~/tmp/fm-ideas && mkdir -p d65b && cd d65b
S=/home/fzwin/tmp/fm-ideas/src-d65b
export FASTMACE_STRUCTURES=/mnt/d/Margraf/ResearchModule PYTHONPATH=$S:$S/tools PYTHONDONTWRITEBYTECODE=1 \
  MACE_COMPILE_CACHE=$HOME/compile-cache/mace_cueq_oeq FASTMACE_MODELS=$HOME/models TORCHINDUCTOR_CACHE_DIR=$HOME/tmp/fm-ideas/d65b/inductor
P=~/environments/mace_cueq_oeq/bin/python
for run in 1 2; do
  start=$(date +%s); timeout 1800 $P $S/tools/bench/f64pkg_acc.py d65-r$run NbCl5 > acc-r$run.log 2>&1
  echo "run $run exit $? $(( $(date +%s) - start )) s pickle-warnings $(grep -c "pickle" acc-r$run.log)" >> ../progress17.txt
done
echo P17DONE >> ../progress17.txt
