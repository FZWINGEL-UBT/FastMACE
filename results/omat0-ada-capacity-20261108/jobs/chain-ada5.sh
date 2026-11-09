#!/usr/bin/env bash
# leto, with the probe fix (969bfc4d): SrTiO3 from 32^3 and NaCl from 36^3 up, one repeat at a time until a failure.
# Holds runs/ada-claim (host leto; tiled-speed's suite queue waits on it) until done.
J=/mnt/homes/zwingel/fm-ada-capacity
mkdir -p $J/runs/ada-claim && hostname > $J/runs/ada-claim/host
export SRC=$J/src-969bfc4d
bash $J/jobs3/ladder2.sh $J/runs/edge-ada5 SrTiO3 32 35
bash $J/jobs3/ladder2.sh $J/runs/edge-ada5 NaCl 36 38
rm -rf $J/runs/ada-claim.done5; mv $J/runs/ada-claim $J/runs/ada-claim.done5
touch $J/runs/ada5.done
