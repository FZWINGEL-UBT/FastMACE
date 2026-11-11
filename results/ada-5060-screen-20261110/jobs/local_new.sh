#!/usr/bin/env bash
# The RTX 5060 Ti part of main 4e9388c8's switches (conductor, 11 November 2026), after local_all.sh: L10 (SC_FACTORED
# on top of SC_GENERATED, base = the spline stack, tf32x3 and compose) through local_gpu.py, default allocator, on
# src-screen. In WSL: nohup setsid bash local_new.sh &
J=$HOME/fm-ada5060/jobs S=$HOME/fm-ada5060/src-screen R=$HOME/fm-ada5060/runs/local-new
mkdir -p $R
unset PYTORCH_CUDA_ALLOC_CONF
while pgrep -f "[l]ocal_all.sh" > /dev/null; do sleep 60; done
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen L10 (OMAT-0 medium, d66)" -- \
  env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$R bash $J/ab3.sh $S L10
echo "$(date -u +%FT%TZ) L10 exit=$?" >> $R/chain.log
