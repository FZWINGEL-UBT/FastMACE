"""Idle check of a self-hosted GPU CI runner (plan task 2-T18; docs/development.md, "GPU CI").

    python3 tools/ci/gpu_idle.py check [--wait-minutes 30] [--interval-seconds 30] [--idle-samples 1]
    python3 tools/ci/gpu_idle.py watch --out FILE [--interval-seconds 60]
    python3 tools/ci/gpu_idle.py summarize FILE
    python3 tools/ci/gpu_idle.py marks fast|full|acceptance

``check`` decides whether the GPU is free for a CI job, waiting up to a limit. The GPU is busy while any
compute process is on it: another user's (the runner never shares a GPU with another user), one of the
runner user's own from another task (one GPU per task), or one whose owner ``ps`` cannot name (counted as
another user's). If nvidia-smi fails or lists no GPU, the GPU counts as busy too. A busy GPU is not an error:
``check`` writes ``idle=false`` to ``$GITHUB_OUTPUT``, the reasons to the job summary, and exits 0, so the
workflow skips the tests job.

Sources of the processes (``--source``, default ``$FASTMACE_CI_GPU_SOURCE``, else ``auto``):

- ``nvidia-smi`` (workstations): ``nvidia-smi --query-compute-apps`` and ``ps`` for their owners;
- ``wsl`` (the PC): WSL's nvidia-smi lists no processes, so the processes are those holding ``/dev/dxg``.
  Windows programs and the desktop are not visible by pid, so the local runner also sets
  ``--max-memory-mib`` (the desktop's own use plus a margin) and ``--max-utilization``, with
  ``--idle-samples`` consecutive idle samples, and ``--min-available-gib`` (WSL's 15 GB: 10 GB free before a
  GPU suite); and a live local GPU reservation of someone else (``--reservation``, tools/ci/local_gpu.py) is
  busy;
- ``auto``: ``wsl`` when the kernel is WSL's, else ``nvidia-smi``.

``watch`` runs beside the tests and appends a JSON line whenever another user's compute process is on the
GPU. It only records (conductor answer to 2-T18 Q3); ``summarize`` turns the file into the job summary.
``marks`` prints a tier's ``-m`` expression from tests/_support/tiers.py.

Standard library only: it runs with the system ``python3`` before any environment is involved.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_gpu  # noqa: E402

GPU_QUERY = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
DXG = "/dev/dxg"


def _run(argv):
    """stdout of ``argv``, or None if it could not run or failed."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_compute_apps(text):
    """Process ids of ``nvidia-smi --query-compute-apps=pid,...`` (``[Insufficient Permissions]`` fields allowed)."""
    pids = []
    for line in text.splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit():
            pids.append(int(first))
    return pids


def parse_ps(text):
    """{pid: user} from ``ps -o pid=,user=``."""
    users = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            users[int(parts[0])] = parts[1]
    return users


def parse_gpus(text):
    """Rows of ``nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total``."""
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, utilization, used, total = parts
        rows.append(dict(name=name, utilization_percent=_number(utilization), memory_used_MiB=_number(used),
                         memory_total_MiB=_number(total)))
    return rows


def dxg_holders(proc_root="/proc"):
    """Process ids with ``/dev/dxg`` open (WSL's GPU device; every CUDA process in WSL holds it)."""
    pids = []
    root = Path(proc_root)
    try:
        entries = list(root.iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:  # gone, or another user's process
            continue
        for fd in fds:
            try:
                if os.readlink(fd) == DXG:
                    pids.append(int(entry.name))
                    break
            except OSError:
                continue
    return sorted(pids)


def available_gib(meminfo="/proc/meminfo"):
    """MemAvailable in GiB, or None."""
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


def decide(pids, users, gpus, own_user, ignore_pids=(), max_memory_mib=None, max_utilization=None,
           available_gib=None, min_available_gib=None):
    """(idle, reasons, processes) for the compute processes ``pids`` with owners ``users`` ({pid: user})."""
    reasons, processes = [], []
    ignore = set(ignore_pids)
    for pid in pids:
        if pid in ignore:
            continue
        user = users.get(pid)
        if user is None:
            owner = "other user"
            reasons.append(f"pid {pid}: unknown owner (ps cannot name it), counted as another user's")
        elif user == own_user:
            owner = "own"
            reasons.append(f"pid {pid}: own process ({user}) from another task")
        else:
            owner = "other user"
            reasons.append(f"pid {pid}: another user's process ({user})")
        processes.append(dict(pid=pid, user=user, owner=owner))
    if not gpus:
        reasons.append("nvidia-smi listed no GPU")
    for gpu in gpus:
        used, utilization = gpu["memory_used_MiB"], gpu["utilization_percent"]
        if max_memory_mib is not None and (used is None or used > max_memory_mib):
            reasons.append(f"{gpu['name']}: memory used {used} MiB > {max_memory_mib:g} MiB")
        if max_utilization is not None and (utilization is None or utilization > max_utilization):
            reasons.append(f"{gpu['name']}: utilization {utilization} % > {max_utilization:g} %")
    if min_available_gib is not None and (available_gib is None or available_gib < min_available_gib):
        shown = "unknown" if available_gib is None else f"{available_gib:.1f}"
        reasons.append(f"RAM available {shown} GiB < {min_available_gib:g} GiB")
    return not reasons, reasons, processes


def _pids(run, source, proc_root):
    """(compute process ids, failure reason or None) from the source."""
    if source == "wsl":
        return dxg_holders(proc_root), None
    text = run(APPS_QUERY)
    if text is None:
        return [], "nvidia-smi --query-compute-apps failed"
    return parse_compute_apps(text), None


def _users(run, pids):
    return parse_ps(run(["ps", "-o", "pid=,user=", "-p", ",".join(map(str, pids))]) or "") if pids else {}


def sample(run, own_user, source, proc_root="/proc", max_memory_mib=None, max_utilization=None,
           min_available_gib=None, reservation=None):
    """One idle record of this moment. ``reservation``: the local GPU marker (tools/ci/local_gpu.py), whose live
    holder makes the GPU busy."""
    pids, failure = _pids(run, source, proc_root)
    gpus = parse_gpus(run(GPU_QUERY) or "")
    idle, reasons, processes = decide(pids, _users(run, pids), gpus, own_user, ignore_pids={os.getpid()},
                                      max_memory_mib=max_memory_mib, max_utilization=max_utilization,
                                      available_gib=available_gib() if min_available_gib is not None else None,
                                      min_available_gib=min_available_gib)
    if failure:
        idle, reasons = False, [failure] + reasons
    holder = local_gpu.live_holder(reservation, proc_root) if reservation else None
    if holder is not None:
        idle, reasons = False, reasons + [local_gpu.describe(holder)]
    return dict(utc=_now(), source=source, idle=idle, reasons=reasons, processes=processes, gpus=gpus)


def wait_for_idle(sample_once, wait_s, interval_s, idle_samples=1, clock=time.monotonic, sleep=time.sleep):
    """Sample every ``interval_s`` until ``idle_samples`` consecutive samples are idle, for at most ``wait_s``.

    Returns the last record with ``idle`` (the verdict), ``waited_s``, ``samples`` and ``busy_reasons_seen``.
    """
    start, count, streak, seen = clock(), 0, 0, []
    while True:
        record = sample_once()
        count += 1
        if record["idle"]:
            streak += 1
        else:
            streak = 0
            seen += [reason for reason in record["reasons"] if reason not in seen]
        waited = clock() - start
        if streak >= idle_samples or waited >= wait_s:
            break
        sleep(min(interval_s, wait_s - waited))
    result = dict(record, idle=streak >= idle_samples, waited_s=waited, samples=count, busy_reasons_seen=seen)
    if record["idle"] and not result["idle"]:
        result["reasons"] = [f"idle in only {streak} of the {idle_samples} consecutive samples needed"]
    return result


def watch(out, own_user, source, interval_s=60.0, rounds=None, sleep=time.sleep, proc_root="/proc"):
    """Append a JSON line to ``out`` for every sample with another user's compute process (record only)."""
    done = 0
    while rounds is None or done < rounds:
        pids, _ = _pids(_run, source, proc_root)
        users = _users(_run, pids)
        others = [dict(pid=pid, user=users.get(pid), owner="other user") for pid in pids
                  if pid != os.getpid() and users.get(pid) != own_user]
        if others:
            with open(out, "a") as handle:
                handle.write(json.dumps(dict(utc=_now(), processes=others)) + "\n")
        done += 1
        if rounds is None or done < rounds:
            sleep(interval_s)


def summarize_watch(path):
    """Markdown for the job summary from a ``watch`` file."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.is_file() else []
    if not lines:
        return "No other user's process appeared on the GPU during the run.\n"
    seen = {}
    for line in lines:
        entry = json.loads(line)
        for process in entry["processes"]:
            key = (process["pid"], process["user"])
            first, _ = seen.get(key, (entry["utc"], None))
            seen[key] = (first, entry["utc"])
    rows = "".join(f"| {pid} | {user or 'unknown'} | {first} | {last} |\n"
                   for (pid, user), (first, last) in seen.items())
    return ("**Another user's process appeared on the GPU during the run.** It is recorded only; the run was "
            "not stopped, so its timings are not clean.\n\n| pid | user | first seen (UTC) | last seen (UTC) |\n"
            "| ---: | --- | --- | --- |\n" + rows)


def tier_expression(tier):
    """The ``-m`` expression of a GPU tier, from tests/_support/tiers.py."""
    spec = importlib.util.spec_from_file_location("_ci_tiers", REPOSITORY_ROOT / "tests" / "_support" / "tiers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"fast": module.FAST, "full": module.FULL, "acceptance": module.ACCEPTANCE}[tier]


def _source(name):
    if name != "auto":
        return name
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        release = ""
    return "wsl" if "microsoft" in release else "nvidia-smi"


def _optional_float(name):
    value = os.environ.get(name)
    return float(value) if value else None


def _append(variable, text):
    path = os.environ.get(variable)
    if path:
        with open(path, "a") as handle:
            handle.write(text)


def _summary(result):
    head = "GPU idle" if result["idle"] else "GPU busy: tests skipped"
    lines = [f"### {head} ({result['source']}, {result['samples']} samples, waited {result['waited_s']:.0f} s)\n"]
    for reason in result["reasons"] or result["busy_reasons_seen"]:
        lines.append(f"- {reason}\n")
    for gpu in result["gpus"]:
        lines.append(f"- {gpu['name']}: {gpu['utilization_percent']} %, {gpu['memory_used_MiB']} of "
                     f"{gpu['memory_total_MiB']} MiB\n")
    return "".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="wait until the GPU is idle, up to a limit")
    check.add_argument("--wait-minutes", type=float, default=30.0)
    check.add_argument("--interval-seconds", type=float, default=30.0)
    check.add_argument("--idle-samples", type=int, default=int(os.environ.get("FASTMACE_CI_IDLE_SAMPLES", "1")))
    check.add_argument("--max-memory-mib", type=float, default=_optional_float("FASTMACE_CI_MAX_MEMORY_MIB"))
    check.add_argument("--max-utilization", type=float, default=_optional_float("FASTMACE_CI_MAX_UTILIZATION"))
    check.add_argument("--min-available-gib", type=float, default=_optional_float("FASTMACE_CI_MIN_AVAILABLE_GIB"))
    check.add_argument("--reservation", default=os.environ.get("FASTMACE_LOCAL_GPU_MARKER"),
                       help="the local GPU reservation's marker (tools/ci/local_gpu.py); a live one is busy")
    check.add_argument("--record", help="write the final record as JSON here")
    watcher = commands.add_parser("watch", help="record other users' compute processes during the run")
    watcher.add_argument("--out", required=True)
    watcher.add_argument("--interval-seconds", type=float, default=60.0)
    for sub in (check, watcher):
        sub.add_argument("--user", default=getpass.getuser(), help="the runner's user (default: this one)")
        sub.add_argument("--source", choices=("auto", "nvidia-smi", "wsl"),
                         default=os.environ.get("FASTMACE_CI_GPU_SOURCE", "auto"))
    summarize = commands.add_parser("summarize", help="the job summary of a watch file")
    summarize.add_argument("file")
    marks = commands.add_parser("marks", help="the -m expression of a GPU tier")
    marks.add_argument("tier", choices=("fast", "full", "acceptance"))
    args = parser.parse_args(argv)

    if args.command == "marks":
        print(tier_expression(args.tier))
        return 0
    if args.command == "summarize":
        print(summarize_watch(args.file), end="")
        return 0
    source = _source(args.source)
    if args.command == "watch":
        watch(args.out, args.user, source, args.interval_seconds)
        return 0
    result = wait_for_idle(
        lambda: sample(_run, args.user, source, max_memory_mib=args.max_memory_mib,
                       max_utilization=args.max_utilization, min_available_gib=args.min_available_gib,
                       reservation=args.reservation),
        wait_s=args.wait_minutes * 60.0, interval_s=args.interval_seconds, idle_samples=args.idle_samples)
    if args.record:
        Path(args.record).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    _append("GITHUB_OUTPUT", f"idle={'true' if result['idle'] else 'false'}\n")
    _append("GITHUB_STEP_SUMMARY", _summary(result))
    if not result["idle"]:
        reasons = "; ".join(result["reasons"])
        print(f"::notice title=GPU busy, tests skipped::{reasons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
