#!/bin/bash
# murphy_suite.sh SHA: idle check (no other user's process or GPU job, none of our capacity/profile jobs), then the
# full CPU suite (-n 8, not acceptance) on the NAS tree src-SHA
set -u
SHA=$1
R=/mnt/homes/zwingel/fm-tiled-speed
busy=$(ps -eo user:20,pcpu --no-headers | awk '$1!="zwingel" && $1!="root" && $2>20' | wc -l)
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
ours=$(pgrep -u zwingel -fc '[f]m-ada-capacity|[f]m-memprofile|[c]apacity_scan|[l]aunch_|[p]ytest')
claims=$(ls -d /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim /mnt/homes/zwingel/fm-ada-capacity/runs/ada-claim2 2>/dev/null | wc -l)
echo "$(hostname) busy=$busy gpu=$gpu ours=$ours claims=$claims $(cat /proc/loadavg)"
if [ "$busy" != 0 ] || [ "$gpu" != 0 ] || [ "$ours" != 0 ] || [ "$claims" != 0 ]; then echo "NOT IDLE"; exit 3; fi
rm -rf $R/tmp/pytest-t11-$SHA
setsid nohup bash $R/cpu_targeted.sh $SHA t11-$SHA tests/cpu/test_whole_system_float64_edges.py tests/cpu/test_strain_per_model.py tests/cpu/test_strain_rebinding_dynamo.py tests/cpu/test_layering.py > /dev/null 2>&1 < /dev/null &
echo started
