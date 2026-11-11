#!/usr/bin/env bash
# murphy, one job at a time (11 November 2026): the X4/X2 packs on the fixed tree, then ada_new.sh.
B=/mnt/homes/zwingel/fm-ada5060
FIXTREE=src-regfix bash $B/jobs/ada_fixverify.sh
bash $B/jobs/ada_new.sh
