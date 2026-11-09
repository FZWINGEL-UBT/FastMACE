#!/usr/bin/env bash
set -eu
sudo -n systemd-run --unit=fm-dmqe-force-20261109 --collect --uid="$(id -u)" --gid="$(id -g)" \
 --setenv=HOME="$HOME" --setenv=PATH="$PATH" -p WorkingDirectory=/mnt/d/Margraf/fm-dmqe-ideas \
 bash /mnt/d/Margraf/fm-dmqe-ideas/tools/local/dmqe_force_local.sh
