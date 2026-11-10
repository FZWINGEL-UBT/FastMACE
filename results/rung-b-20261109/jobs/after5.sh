#!/usr/bin/env bash
# leto: wait for chain 5, then run chain 6.
#   bash after5.sh
R5=/mnt/homes/zwingel/fm-memprofile/runs/rungb5
until grep -q "CHAIN5 DONE" $R5/run.log 2>/dev/null; do sleep 60; done
R6=/mnt/homes/zwingel/fm-memprofile/runs/rungb6; mkdir -p $R6/limits
ROOT=$R6 bash /mnt/homes/zwingel/fm-memprofile/rungb-jobs/chain6.sh /mnt/homes/zwingel/fm-memprofile/rungb-c4 > $R6/chain.out 2>&1
