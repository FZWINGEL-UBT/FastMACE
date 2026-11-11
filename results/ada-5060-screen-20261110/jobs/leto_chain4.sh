#!/usr/bin/env bash
# leto, one job at a time (11 November 2026): regfix.sh (the spline-registry-fix tests, memcheck, CPU suite), then the
# X4 and X2 packs on the same fixed tree (ada_fixverify.sh with FM=src-regfix). Runs on this host only (no claim race
# with murphy's POLAR gate chain).
B=/mnt/homes/zwingel/fm-ada5060
bash $B/jobs/regfix.sh
until grep -qE "DONE|STOP" $B/runs/regfix/run.log 2>/dev/null; do sleep 60; done
FIXTREE=src-regfix bash $B/jobs/ada_fixverify.sh
