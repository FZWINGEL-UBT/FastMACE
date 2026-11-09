#!/usr/bin/env bash
# leto, source 969bfc4d (probe fix): NaCl 36^3 with --diagnose (it still fell to the terminal sequence), then SrTiO3
# from 36^3 up (35^3 passed). Holds runs/ada-claim (host leto) until done.
J=/mnt/homes/zwingel/fm-ada-capacity
mkdir -p $J/runs/ada-claim && hostname > $J/runs/ada-claim/host
ROOT=$J/runs/diag-ada6 bash $J/jobs3/diag.sh $J/src-969bfc4d NaCl:36x36x36
SRC=$J/src-969bfc4d bash $J/jobs3/ladder2.sh $J/runs/edge-ada6 SrTiO3 36 39
rm -rf $J/runs/ada-claim.done6; mv $J/runs/ada-claim $J/runs/ada-claim.done6
touch $J/runs/ada6.done
