#!/usr/bin/env bash
# leto chain 2 for item 2: the plain plan (the rung forced off) and the rung forced on at SrTiO3 32^3 and 36^3
# (measured peak and step time: what rule 12 should predict, and the rung's cost), the memory probe with both stages
# ("always") at 32^3 and 36^3 (the plain stage's measured scale), then the speed A/B.
#   ROOT=<dir> bash chain2.sh NEW_SRC
set -u
NEW=$1
J=$(dirname "$0")
export NOSTOP=1
ROOT=$ROOT/plain TAG=-plain EXTRA="--steps 100 --rewrites recompute_first_table=off" bash $J/points2.sh "$NEW" SrTiO3:32x32x32,36x36x36
ROOT=$ROOT/rung TAG=-rung EXTRA="--steps 100 --rewrites recompute_first_table=on" bash $J/points2.sh "$NEW" SrTiO3:32x32x32,36x36x36
ROOT=$ROOT/always TAG=-always EXTRA="--steps 20 --engine memory_probe=always" bash $J/points2.sh "$NEW" SrTiO3:32x32x32,36x36x36
ROOT=$ROOT/ab bash $J/abregion.sh "$NEW"
echo "$(date -u +%FT%TZ) CHAIN2 DONE" >> $ROOT/run.log
