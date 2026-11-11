#!/usr/bin/env bash
# The RTX 5060 Ti work after the 10 November restarts: one local_gpu.py reservation per group, default CUDA allocator
# (PYTORCH_CUDA_ALLOC_CONF unset; conductor 10 Nov), groups in order. In WSL: nohup setsid bash local_all.sh &
J=$HOME/fm-ada5060/jobs S=$HOME/fm-ada5060/src-7d18028d R=$HOME/fm-ada5060/runs/local
mkdir -p $R
unset PYTORCH_CUDA_ALLOC_CONF
for group in "L1 L5 L6" "L2 L3 L4" "L7 L8 L9" polar; do
  if [ "$group" = polar ]; then cmd=(env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$HOME/fm-ada5060/runs/local-polar bash $J/polar.sh $S pair-M pair-L epi-S epi-M epi-L)
  else cmd=(env -u PYTORCH_CUDA_ALLOC_CONF HOST=local ROOT=$R bash $J/ab2.sh $S $group); fi
  python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen $group (OMAT-0 medium / POLAR-1, d66)" -- "${cmd[@]}"
  echo "$(date -u +%FT%TZ) group '$group' exit=$?" >> $R/chain.log
done
