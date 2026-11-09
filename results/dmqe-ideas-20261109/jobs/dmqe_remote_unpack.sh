#!/usr/bin/env bash
set -eu
run=/mnt/homes/zwingel/fm-dmqe-20261109
tar -xf "$run/source.tar" -C "$run/src"
printf 'a7a8211e2f740c5b95ebcdcd6ed2beee76098718\n' > "$run/src/.source_commit"
exit
