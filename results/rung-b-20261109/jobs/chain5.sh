#!/usr/bin/env bash
# leto chain 5 (conductor review, item 2): the default plan of a7a8211e (OLD) and the two-size probe (NEW) in the
# reverse order of chain 4 (OLD first): SrTiO3 35^3-37^3 on OLD, then 35^3-36^3 on NEW; ms/step, peak, rungs taken.
#   ROOT=<dir> bash chain5.sh NEW_SRC OLD_SRC
set -u
NEW=$1; OLD=$2
J=$(dirname "$0")
export NOSTOP=1
ROOT=$ROOT/old TAG=-old EXTRA="--steps 100" bash $J/points2.sh "$OLD" SrTiO3:35x35x35,36x36x36,37x37x37
ROOT=$ROOT/new TAG=-new EXTRA="--steps 100" bash $J/points2.sh "$NEW" SrTiO3:35x35x35,36x36x36
echo "$(date -u +%FT%TZ) CHAIN5 DONE" >> $ROOT/run.log
