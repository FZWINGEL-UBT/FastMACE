#!/usr/bin/env bash
# leto chain 3 for item 2 on cd56800c (the layer region as its own rung, generic_recompute_first_layer, after the
# first table's; the probe measures the cheaper stage): the default plan at SrTiO3 32^3 (plan text against
# rungb1/plan-old, a7a8211e) and 36^3 (which rungs the planner takes), then the limits.
#   ROOT=<dir> bash chain3.sh NEW_SRC
set -u
NEW=$1
J=$(dirname "$0")
export NOSTOP=1
ROOT=$ROOT/plan TAG=-default EXTRA="--steps 100" bash $J/points2.sh "$NEW" SrTiO3:32x32x32,36x36x36
unset NOSTOP
ROOT=$ROOT/points bash $J/points2.sh "$NEW" SrTiO3:38x38x38,39x39x39,40x40x40,41x41x41 NaCl:37x37x37,38x38x38,39x39x39
echo "$(date -u +%FT%TZ) CHAIN3 DONE" >> $ROOT/run.log
