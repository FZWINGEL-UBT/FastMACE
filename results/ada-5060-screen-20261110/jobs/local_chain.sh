#!/usr/bin/env bash
# The RTX 5060 Ti packs, one local_gpu.py reservation per group so other lanes can interleave.
#   bash local_chain.sh   (in WSL)
J=$HOME/fm-ada5060/jobs
R=$HOME/fm-ada5060/runs/local
mkdir -p $R
for group in "L1 L5 L6" "L2 L3 L4"; do
  python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen A/B packs $group (OMAT-0 medium, d66)" -- \
    env HOST=local ROOT=$R bash $J/ab.sh $HOME/fm-ada5060/src-7d18028d $group
  echo "$(date -u +%FT%TZ) group '$group' exit=$?" >> $R/chain.log
done
