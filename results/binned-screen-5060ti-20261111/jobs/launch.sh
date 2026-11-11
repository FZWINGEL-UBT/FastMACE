#!/bin/bash
# WSL: job.sh as a user systemd unit inside one local_gpu reservation (quiet: a timing run).
J=/mnt/d/Margraf/FastMACE/results/binned-screen-5060ti-20261111/jobs
RUN=$HOME/fm-runs/binned-5060-20261111
mkdir -p "$RUN/out"
systemd-run --user --collect --unit=fm-binned-5060 \
  --setenv=PATH=/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin --setenv=LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  bash -lc "python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose 'BINNED_SCREEN rebuild parts and MD A/B, SrTiO3 16^3 and 20^3 (both orders)' --quiet --wait-minutes 120 -- bash $J/job.sh > $RUN/out/job.log 2>&1; echo \"launch exit=\$?\" >> $RUN/out/job.log"
sleep 3
systemctl --user is-active fm-binned-5060
