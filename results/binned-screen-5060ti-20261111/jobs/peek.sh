#!/bin/bash
# WSL: a look at the running job (read only).
O=$HOME/fm-runs/binned-5060-20261111/out
cat $O/job.log 2>/dev/null | cut -c1-300 | tail -30
echo "-- files"; ls $O | head -30
for f in $O/parts.log; do [ -f "$f" ] && { echo "-- $f"; tail -12 "$f" | cut -c1-260; }; done
