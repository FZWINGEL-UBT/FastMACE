#!/usr/bin/env bash
# RTX 5060 Ti packs L7-L9 (ab2.sh), after local_chain.sh's second group; own reservation.
R=$HOME/fm-ada5060/runs/local
until grep -q "group 'L2 L3 L4'" $R/chain.log 2>/dev/null; do sleep 120; done
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --purpose "ada-5060-screen A/B packs L7 L8 L9 (d66)" -- \
  env HOST=local ROOT=$R bash $HOME/fm-ada5060/jobs/ab2.sh $HOME/fm-ada5060/src-7d18028d L7 L8 L9
echo "$(date -u +%FT%TZ) group 'L7 L8 L9' exit=$?" >> $R/chain.log
