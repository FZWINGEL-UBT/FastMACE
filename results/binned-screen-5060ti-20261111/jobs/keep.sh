#!/bin/bash
# WSL: stay open while the unit runs (keeps the VM alive and signals the end), then print the job log.
while systemctl --user is-active -q fm-binned-5060; do sleep 20; done
cat $HOME/fm-runs/binned-5060-20261111/out/job.log | cut -c1-320
