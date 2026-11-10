#!/usr/bin/env bash
# leto chain 6: the Ada limits again on the two-size probe (1522aef2): SrTiO3 39^3 and NaCl 37^3 with their rungs.
#   ROOT=<dir> bash chain6.sh NEW_SRC
set -u
NEW=$1
J=$(dirname "$0")
export NOSTOP=1
ROOT=$ROOT/limits TAG=-limit bash $J/points2.sh "$NEW" SrTiO3:39x39x39 NaCl:37x37x37
echo "$(date -u +%FT%TZ) CHAIN6 DONE" >> $ROOT/run.log
