#!/usr/bin/env bash
# The RTX 5060 Ti POLAR-1 cells (after local_chain.sh's packs; their own reservation).
R=$HOME/fm-ada5060/runs/local-polar
mkdir -p $R
until grep -q "group 'L2 L3 L4'" $HOME/fm-ada5060/runs/local/chain.log 2>/dev/null; do sleep 120; done
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen POLAR-1 A/Bs (d66)" -- \
  env HOST=local ROOT=$R bash $HOME/fm-ada5060/jobs/polar.sh $HOME/fm-ada5060/src-7d18028d pair-M pair-L epi-S epi-M epi-L
echo "$(date -u +%FT%TZ) polar exit=$?" >> $R/chain.log
