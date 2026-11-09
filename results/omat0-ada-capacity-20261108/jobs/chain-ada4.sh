#!/usr/bin/env bash
# leto, after chain-ada3: NaCl from 35^3 up (34^3 passed alone), SrTiO3 from 33^3 up if the 32^3 rerun passed.
J=/mnt/homes/zwingel/fm-ada-capacity
while pgrep -f "^bash jobs3/chain-ada3" > /dev/null; do sleep 60; done
bash $J/jobs3/ladder.sh $J/runs/edge-ada3 NaCl 35 37
st=$(python3 -c "import json; print(json.load(open('$J/runs/edge-ada2/points/point-SrTiO3-float32-163840.json'))['status'])" 2>/dev/null)
echo "$(date -u +%FT%TZ) SrTiO3 32^3 rerun status=$st"
[ "$st" = ok ] && bash $J/jobs3/ladder.sh $J/runs/edge-ada3 SrTiO3 33 35
touch $J/runs/ada4.done
