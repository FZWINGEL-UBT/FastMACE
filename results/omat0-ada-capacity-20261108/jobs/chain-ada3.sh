#!/usr/bin/env bash
# leto: after chain-ada2, SrTiO3 32^3 once more (the rerun in chain-ada2 shared the GPU with another user's job from
# 22:41 UTC, so it is void), then mark the Ada work done (release2.sh frees the claims).
J=/mnt/homes/zwingel/fm-ada-capacity
while pgrep -f "^bash jobs3/chain-ada2" > /dev/null; do sleep 60; done
ROOT=$J/runs/edge-ada2 bash $J/jobs3/edge2.sh $J/src-d3d459ef SrTiO3:32x32x32
touch $J/runs/ada3.done
