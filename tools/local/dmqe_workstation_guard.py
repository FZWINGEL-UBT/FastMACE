"""Run one workstation lab only while the shared workstation is idle."""
from __future__ import annotations

import argparse
import json
import os
import pwd
import signal
import subprocess
import time
from pathlib import Path


def compute_processes():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    processes = []
    for line in raw.splitlines():
        if not line.strip().isdigit():
            raise RuntimeError(f"unidentified GPU process: {line}")
        pid = int(line)
        try:
            uid = Path(f"/proc/{pid}").stat().st_uid
            owner = pwd.getpwuid(uid).pw_name
        except (OSError, KeyError):
            owner = "unknown"
        processes.append({"pid": pid, "owner": owner})
    return processes


def other_cpu_ticks():
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            uid = path.stat().st_uid
            if uid < 1000 or uid == os.getuid() or uid == 65534:
                continue
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(path.name)] = (uid, int(fields[11]) + int(fields[12]))
        except (OSError, IndexError, ValueError):
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    if not args.record.resolve().is_relative_to(Path("/mnt/homes/zwingel")):
        parser.error("record must stay under /mnt/homes/zwingel")
    first = other_cpu_ticks()
    time.sleep(5)
    second = other_cpu_ticks()
    # Ignore idle login daemons and heartbeat threads; reject >=1% of a core over the sample.
    limit = max(1, int(os.sysconf("SC_CLK_TCK") * 0.05))
    busy = [{"pid": pid, "owner": pwd.getpwuid(uid).pw_name, "ticks": ticks - first[pid][1]}
            for pid, (uid, ticks) in second.items()
            if pid in first and ticks - first[pid][1] >= limit]
    gpu = compute_processes()
    record = {"host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "cpu_sample_seconds": 5, "other_active_cpu": busy, "gpu_processes": gpu,
              "idle": not busy and not gpu, "command": command}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if busy or gpu:
        return 3
    child = subprocess.Popen(command, start_new_session=True)
    while child.poll() is None:
        time.sleep(2)
        others = [item for item in compute_processes() if item["owner"] != pwd.getpwuid(os.getuid()).pw_name]
        if others:
            record["stopped_for_other_gpu_user"] = others
            args.record.write_text(json.dumps(record, indent=2) + "\n")
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
            return 4
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
