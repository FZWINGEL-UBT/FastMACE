#!/usr/bin/env bash
# leto chain for item 2 (rung B, the first-layer region): plan text below the limit on both trees, the 36^3 MD profile
# with FIRST_LAYER_REGION on and off, the capacity points, then the speed A/B. Each GPU job waits for an idle GPU.
#   ROOT=<dir> bash chain.sh NEW_SRC OLD_SRC
set -u
NEW=$1; OLD=$2
J=$(dirname "$0")
export ROOT
g=fastmace.rewrites.generic.modules
# 1. the default plan below the limit: SrTiO3 32^3 on both trees (plan text in the point records)
ROOT=$ROOT/plan-new bash $J/points.sh "$NEW" SrTiO3:32x32x32
ROOT=$ROOT/plan-old bash $J/points.sh "$OLD" SrTiO3:32x32x32
# 2. the profile at 36^3 (the plan takes generic_recompute_first_table there): region on (default) and off
mkdir -p $ROOT/prof
ROOT=$ROOT/prof FM=$NEW TAG=-region bash $J/leto.sh md:36x36x36
ROOT=$ROOT/prof FM=$NEW TAG=-noregion EXTRA="--constant $g:FIRST_LAYER_REGION=False" bash $J/leto.sh md:36x36x36
# 3. the limits
ROOT=$ROOT/points bash $J/points.sh "$NEW" SrTiO3:38x38x38,39x39x39,40x40x40,41x41x41 NaCl:36x36x36,37x37x37,38x38x38
# 4. speed: the rung forced on both sides, region off (a) / on (b)
ROOT=$ROOT/ab bash $J/abregion.sh "$NEW"
echo "$(date -u +%FT%TZ) CHAIN DONE" >> $ROOT/run.log
