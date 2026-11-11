#!/usr/bin/env bash
# murphy, one job at a time (11 November 2026; leto busy with another user): regfix.sh, then the X4/X2 packs on the
# fixed tree (ada_fixverify.sh, FIXTREE=src-regfix), then ada_new.sh (GEOM, SC_FACTORED, VIRIAL_ELEMENTWISE).
B=/mnt/homes/zwingel/fm-ada5060
bash $B/jobs/regfix.sh
FIXTREE=src-regfix bash $B/jobs/ada_fixverify.sh
bash $B/jobs/ada_new.sh
