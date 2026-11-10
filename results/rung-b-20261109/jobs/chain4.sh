#!/usr/bin/env bash
# leto chain 4 for item 2 (conductor review): the default plan at SrTiO3 33^3-37^3 on the two-size probe (NEW) and
# 34^3-36^3 on a7a8211e (OLD): rungs taken, ms/step, peak reserved; then the force/stress gate of the new rung.
#   ROOT=<dir> bash chain4.sh NEW_SRC OLD_SRC
set -u
NEW=$1; OLD=$2
J=$(dirname "$0")
export NOSTOP=1
ROOT=$ROOT/new TAG=-new EXTRA="--steps 100" bash $J/points2.sh "$NEW" SrTiO3:33x33x33,34x34x34,35x35x35,36x36x36,37x37x37
ROOT=$ROOT/old TAG=-old EXTRA="--steps 100" bash $J/points2.sh "$OLD" SrTiO3:34x34x34,35x35x35,36x36x36
ROOT=$ROOT bash $J/gate3.sh "$NEW"
echo "$(date -u +%FT%TZ) CHAIN4 DONE" >> $ROOT/run.log
