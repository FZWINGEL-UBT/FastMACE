#!/usr/bin/env bash
set -eu
cd /mnt/d/Margraf/fm-dmqe-ideas
mkdir -p results/dmqe-ideas-20261109/local-profile
setsid nohup bash tools/local/dmqe_profile_local.sh > results/dmqe-ideas-20261109/local-profile/launcher.log 2>&1 < /dev/null &
printf 'PID %s\n' "$!"
