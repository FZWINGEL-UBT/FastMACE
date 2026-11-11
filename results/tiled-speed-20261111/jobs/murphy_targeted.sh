#!/bin/bash
# murphy_targeted.sh SHA NAME TESTS...: ws_suite.sh's idle check, then cpu_targeted.sh on the NAS tree src-SHA
set -u
SHA=$1; NAME=$2; shift 2
R=/mnt/homes/zwingel/fm-tiled-speed
busy=$(ps -eo user:20,pcpu --no-headers | awk '$1!="zwingel" && $1!="root" && $2>20' | wc -l)
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[c]apacity_scan|[l]aunch_|[p]ytest')
claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
echo "$(hostname) busy=$busy gpu=$gpu ours=$ours claims=$claims $(cat /proc/loadavg)"
if [ "$busy" != 0 ] || [ "$gpu" != 0 ] || [ "$ours" != 0 ] || [ "$claims" != 0 ]; then echo "NOT IDLE"; exit 3; fi
rm -rf $R/tmp/pytest-$NAME
setsid nohup bash $R/cpu_targeted.sh $SHA $NAME "$@" > /dev/null 2>&1 < /dev/null &
echo started
