#!/usr/bin/env bash
# sx-layout lab, follow-up (7 November 2026): the spline on piecewise-uniform knots (tools/sx_layout_lab.py GRIDS:
# a695 dense at short r and certified over the whole domain, p399 certified at r >= 1 A only) against uniform 256
# and 2,048 knots and production, forward and w'. Same job as sx_layout_lab.sh with these variants.
set -uo pipefail
VARIANTS="base sx-spline256 sx-spline2048 sx-splinea695 sx-splinep399 B-spline256 B-spline2048 B-splinea695 B-splinep399" \
  exec bash tools/modal/sx_layout_lab.sh
