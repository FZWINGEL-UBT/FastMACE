#!/usr/bin/env bash
# murphy, one job at a time (11 November 2026): memcheck2.sh on the unfixed tree (the X4 layout; its first attempt
# exited at once: common.sh sourced before ROOT was set), then ada_new.sh.
B=/mnt/homes/zwingel/fm-ada5060
rm -rf $B/runs/memcheck
bash $B/jobs/memcheck2.sh none
bash $B/jobs/ada_new.sh
