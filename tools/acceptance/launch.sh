#!/usr/bin/env bash
# Start tools/acceptance/run.py detached, so that it survives the shell (and the WSL session or
# SSH connection) that started it (ResearchModule AGENTS.md: "Long runs must be launched detached").
#
#   launch.sh SUITE [SUITE ...] --machine NAME --run-id ID [run.py options]
#
# Its own output goes to <root>/<run id>/run-<UTC time>.log; the PID is printed. Check the run
# afterwards in <root>/<run id>/manifest.json. Uses the system python3 (run.py needs only the
# standard library); the jobs choose their own environments.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(python3 "$here/run.py" "$@" --dry-run > /dev/null && python3 - "$here" "$@" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import suites
args = sys.argv[2:]
machine = suites.MACHINES[args[args.index("--machine") + 1]]
run_id = args[args.index("--run-id") + 1]
print(suites.check_location(machine, machine.path(machine.root) / run_id))
PY
)
mkdir -p "$root"
log="$root/run-$(date -u +%Y%m%dT%H%M%SZ).log"
setsid nohup python3 "$here/run.py" "$@" > "$log" 2>&1 < /dev/null &
echo "started PID $! ; log $log"
