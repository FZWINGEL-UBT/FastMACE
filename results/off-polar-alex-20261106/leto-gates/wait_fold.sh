# Waits up to 4 h for leto to be idle (no other user's GPU process, CPU job or pytest), then runs the fold gate.
B=/mnt/homes/zwingel/fm-offpolar-gates
for i in $(seq 1 48); do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
  busy=$(ps -eo user,pcpu,comm --no-headers | awk '$1 != "root" && $1 != "zwingel" && $2 > 20' | paste -sd, -)
  suite=$(pgrep -f "[p]ytest" | paste -sd, -)
  if [ -z "$gpu$busy$suite" ]; then echo "$(date -u +%FT%TZ) idle, starting"; exec bash $B/run.sh $B/leto4 fold; fi
  echo "$(date -u +%FT%TZ) busy gpu=[$gpu] cpu=[$busy] pytest=[$suite]"
  sleep 300
done
echo "$(date -u +%FT%TZ) gave up after 4 h"
