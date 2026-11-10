#!/usr/bin/env bash
# Holds leto's claims (runs/ada-claim watched by tiled-speed's suite) until chain-ada4 is done, then renames them.
R=/mnt/homes/zwingel/fm-ada-capacity/runs
until [ -f $R/ada4.done ]; do sleep 60; done
mv $R/ada-claim $R/ada-claim.done; mv $R/ada-claim2 $R/ada-claim2.done
echo "$(date -u +%FT%TZ) released ada-claim and ada-claim2"
