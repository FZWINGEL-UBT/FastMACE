#!/usr/bin/env bash
# leto: the prerequisite probes (spline / compose in use on the Ada), the force/stress gate of the Ada default plan,
# then the capacity scan of the plan before this change (73e421ea; the Ada plan as at eced7d75) from the last maxima.
J=/mnt/homes/zwingel/fm-ada-capacity
SRC=$J/src-73e421ea
ROOT=$J/runs/probe-leto bash $J/jobs/probe.sh $SRC 4
# (the default gate ran in the first pass of this chain)
ROOT=$J/runs/scan-old-leto bash $J/jobs/scan.sh $SRC SrTiO3:69120 NaCl:125000
