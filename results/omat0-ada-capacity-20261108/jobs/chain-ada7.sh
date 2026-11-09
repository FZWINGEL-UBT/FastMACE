#!/usr/bin/env bash
# leto, after chain-ada6: NaCl 30^3 and 35^3 on 969bfc4d (the probe now measures the rung correctly: does it still
# choose the default plan where it fits, and at what speed). Holds runs/ada-claim (host leto) until done.
J=/mnt/homes/zwingel/fm-ada-capacity
until [ -f $J/runs/ada6.done ]; do sleep 60; done
mkdir -p $J/runs/ada-claim && hostname > $J/runs/ada-claim/host
ROOT=$J/runs/edge-ada7 bash $J/jobs3/edge2.sh $J/src-969bfc4d NaCl:30x30x30 NaCl:35x35x35
rm -rf $J/runs/ada-claim.done7; mv $J/runs/ada-claim $J/runs/ada-claim.done7
touch $J/runs/ada7.done
