#!/usr/bin/env bash
# Step one cubic repeat up at a time (edge2.sh: idle wait, 40-minute limit, GPU log) until a point fails or MAX.
#   bash ladder.sh ROOT STRUCTURE FIRST_N MAX_N
J=/mnt/homes/zwingel/fm-ada-capacity
ROOT=$1 S=$2 n=$3 max=$4
while [ $n -le $max ]; do
  ROOT=$ROOT bash $J/jobs3/edge2.sh $J/src-d3d459ef $S:${n}x${n}x${n}
  f=$(ls $ROOT/points/point-$S-float32-$((n*n*n*$([ $S = SrTiO3 ] && echo 5 || echo 8))).json 2>/dev/null)
  st=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$f" 2>/dev/null)
  echo "$(date -u +%FT%TZ) ladder $S ${n}^3 status=$st"
  [ "$st" = ok ] || break
  n=$((n + 1))
done
