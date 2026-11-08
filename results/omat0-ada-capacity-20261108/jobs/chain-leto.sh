#!/usr/bin/env bash
# leto: the force/stress gate with the ten switches and with the Ada default plan, then the capacity scan of the plan
# before this change (source 73e421ea; the Ada plan is the same as at eced7d75) from the last scan's maxima.
J=/mnt/homes/zwingel/fm-ada-capacity
SRC=$J/src-73e421ea
ROOT=$J/runs/gate-leto bash $J/jobs/gate.sh $SRC all
ROOT=$J/runs/gate-leto bash $J/jobs/gate.sh $SRC default
ROOT=$J/runs/scan-old-leto bash $J/jobs/scan.sh $SRC SrTiO3:69120 NaCl:125000
