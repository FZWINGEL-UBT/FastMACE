#!/usr/bin/env bash
# leto: SrTiO3 32^3 with --diagnose (the fallback refuses, so the first failure is raised): why the Ada's new plan
# fell through to the terminal sequence instead of a rescue rung. After tiled-speed's suite (QUIET 5 minutes).
J=/mnt/homes/zwingel/fm-ada-capacity
QUIET=5 bash $J/jobs/queue.sh true
ROOT=$J/runs/diag-ada bash $J/jobs3/diag.sh $J/src-d3d459ef SrTiO3:32x32x32
