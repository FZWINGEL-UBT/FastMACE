#!/usr/bin/env bash
# One H200 container for the bspline-lab lane's night batch (9 November 2026): the SPLINE_W forward table-read lab
# (packed8 against real and one-slab, plus w' locality probes local-sender / local-dout / combined;
# tools/modal/spline_read_lab_job.sh), then the compiler and shared-spill MD-step A/Bs (tools/modal/compiler_spill_ab.sh).
set -uo pipefail
out=${OUT:?the job sets OUT}
mkdir -p "$out/lab" "$out/ab"
OUT=$out/lab bash tools/modal/spline_read_lab_job.sh
OUT=$out/ab bash tools/modal/compiler_spill_ab.sh
cat "$out/lab/run.log" "$out/ab/run.log" > "$out/run.log"
