#!/usr/bin/env bash
# 4-T09 GPU smoke on Modal (L4), the command of a tools/modal/fastmace_modal.py job, run from the source root
# of results/4-T09-smoke/tarball.py's tarball: the commit plus inputs/ (the protocol as YAML, since the image's
# Python 3.10 has no tomllib, and 3b-T07's short run without its quench).
#
# 1. npt_anneal_NbCl5_short end to end with gpu-LBFGS for the relaxation and the quench, into $OUT/e2e;
# 2. fastmace quench --into gpu (gpu-LBFGS) on a copy of 3b-T07's short run, into $OUT/short-copy/quench-gpu,
#    with max_steps 2600: every frame that converged in 3b-T07's ASE quench did so within 2,523 steps, and the
#    vapour frames that stopped at 5000 there are not compared.
#
# Everything is written into $OUT as it runs, so a timeout still returns the finished part.
set -uo pipefail
export FASTMACE_COMMIT="$(cat .source_commit)"
echo "=== $(date -u +%FT%TZ) e2e (FastMACE $FASTMACE_COMMIT)" >>"$OUT/progress.log"
python -m fastmace.cli run inputs/npt_anneal_NbCl5_short_gpu.yaml --output "$OUT/e2e" >"$OUT/e2e.log" 2>&1
echo "=== $(date -u +%FT%TZ) e2e exit $?; copy" >>"$OUT/progress.log"
cp -r inputs/3b-T07-short "$OUT/short-copy"
python -m fastmace.cli quench "$OUT/short-copy" --into gpu --optimizer gpu-LBFGS --max-steps 2600 \
    >"$OUT/copy.log" 2>&1
echo "=== $(date -u +%FT%TZ) copy exit $?" >>"$OUT/progress.log"
rm -f "$OUT"/short-copy/trajectory.*.traj "$OUT"/short-copy/restart.npz  # inputs, kept locally
tail -n 3 "$OUT/e2e.log" "$OUT/copy.log"
