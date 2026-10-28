"""The local GPU reservation on the PC (plan task 2-T18; docs/development.md, "The local GPU reservation").

    python3 tools/ci/local_gpu.py run --purpose TEXT [--quiet] [--wait-minutes 60] -- COMMAND ...
    python3 tools/ci/local_gpu.py reserve --purpose TEXT --pid PID [--quiet] [--wait-minutes 60]
    python3 tools/ci/local_gpu.py release --pid PID
    python3 tools/ci/local_gpu.py check [--cpu-only] [--min-available-gib 6]
    python3 tools/ci/local_gpu.py status   # the holder and the queue

A GPU suite started while sibling CPU suites ran has crashed WSL (19 October 2026, when WSL had 15 GB of
RAM; it has 40 GB since 21 October), and WSL's nvidia-smi does not show who is on the GPU. So every local
GPU run (the GPU CI runner, a worker's GPU suite, a timing run) holds this reservation, and every local CPU
suite checks it first:

- **reserve** creates the marker directory atomically (``mkdir``; default ``~/.fastmace-local-gpu``, or
  ``$FASTMACE_LOCAL_GPU_MARKER``) with ``holder.json``: owner, purpose, pid, the pid's start time and the
  start of the reservation. While another live holder has it, reserve waits, up to ``--wait-minutes``.
- Waiters are served **first come, first served**: each reserve or run first puts a ticket into the queue
  next to the marker (``~/.fastmace-local-gpu-queue/<arrival ns>-<pid>.json``, written to a temp file and
  renamed) and takes the marker only while its ticket is the oldest live one. Without the queue, waiters
  raced at each release, and one worker's back-to-back jobs kept four short runs waiting 37 minutes
  (22 October 2026). A ticket whose waiting process is gone is stale and is dropped; a waiter that gives
  up removes its own ticket.
- Holding it, reserve **waits for the machine**: until no other ``pytest`` process that can use the GPU
  runs, and at least ``--min-available-gib`` (default 10) GiB of RAM are available. A pytest process whose
  environment sets ``CUDA_VISIBLE_DEVICES`` empty (or ``-1``) cannot use the GPU and does not count
  (phase 3, 22 October 2026: five workers' CPU suites kept GPU reservations waiting). With ``--quiet``
  (timing runs) every other pytest process counts, and the reservation is marked quiet. If the machine is
  not free in time, reserve releases the marker again and fails with status 75 (busy).
- **release** removes the marker, only for its own pid.
- A marker whose holder process is gone (or whose pid now belongs to another process) is **stale**: it is
  reported and removed by the next reserve or check.
- **check** exits 1 while a live marker exists, or with ``--min-available-gib`` while less RAM is
  available. Workers run it before every local CPU suite. ``check --cpu-only`` is for a suite run with
  ``CUDA_VISIBLE_DEVICES=``: only a quiet (timing) reservation blocks it.
- **run** reserves for itself, runs the command, releases on any exit and returns the command's status
  (75 if it could not reserve in time).

Standard library only.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BUSY = 75  # EX_TEMPFAIL: not reserved in time
DEFAULT_MARKER = "~/.fastmace-local-gpu"
HOLDER = "holder.json"
UNWRITTEN_GRACE_S = 60.0  # a marker without holder.json is a reserver between mkdir and write, this long
QUEUE_SUFFIX = "-queue"


def default_marker():
    return Path(os.path.expanduser(os.environ.get("FASTMACE_LOCAL_GPU_MARKER", DEFAULT_MARKER)))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stat_fields(pid, proc_root):
    """The fields of /proc/<pid>/stat after ``(comm)``, or None."""
    try:
        text = (Path(proc_root) / str(pid) / "stat").read_text()
    except OSError:
        return None
    return text[text.rfind(")") + 2:].split()


def start_ticks(pid, proc_root="/proc"):
    """The process's start time in clock ticks since boot (field 22 of /proc/<pid>/stat), or None."""
    fields = _stat_fields(pid, proc_root)
    try:
        return int(fields[19])
    except (TypeError, IndexError, ValueError):
        return None


def parent_pid(pid, proc_root="/proc"):
    fields = _stat_fields(pid, proc_root)
    try:
        return int(fields[1])
    except (TypeError, IndexError, ValueError):
        return None


def available_gib(proc_root="/proc"):
    """MemAvailable in GiB, or None."""
    try:
        for line in (Path(proc_root) / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


def _ancestors(pid, proc_root):
    seen = set()
    while pid and pid not in seen:
        seen.add(pid)
        pid = parent_pid(pid, proc_root)
    return seen


def _is_pytest(argv):
    """``pytest ...``, ``.../pytest ...`` or ``python -m pytest ...`` (not a shell whose script mentions it)."""
    if not argv:
        return False
    # A queued ``local_gpu.py run -- python -m pytest ...`` wrapper is not a running pytest: counting it
    # made the holder wait for every queued reservation (deadlock, round 2, 21 October 2026). Prefixes such
    # as ``timeout 900 python3 local_gpu.py ...`` or ``env ... python3 local_gpu.py ...`` are wrappers too
    # (phase 3, 22 October 2026: 3-T04's holder waited for 3-T03's ``timeout`` wrapper).
    if any(Path(arg).name == "local_gpu.py" for arg in argv):
        return False
    if Path(argv[0]).name == "pytest":
        return True
    return any(argv[i] == "-m" and argv[i + 1] == "pytest" for i in range(len(argv) - 1))


def _cpu_only(pid, proc_root):
    """True if the process's environment hides every GPU (``CUDA_VISIBLE_DEVICES`` empty or ``-1``)."""
    try:
        environ = (Path(proc_root) / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return False
    for entry in environ:
        if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
            return entry.split(b"=", 1)[1].strip() in (b"", b"-1")
    return False


def pytest_processes(proc_root="/proc", own_pid=None, gpu_only=False):
    """Pids of pytest processes other than this one and its ancestors; with ``gpu_only`` only those that
    can use the GPU (not ``_cpu_only``)."""
    own = _ancestors(own_pid if own_pid is not None else os.getpid(), proc_root)
    pids = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            argv = [arg for arg in entry.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0") if arg]
        except OSError:
            continue
        if _is_pytest(argv) and not (gpu_only and _cpu_only(int(entry.name), proc_root)):
            pids.append(int(entry.name))
    return sorted(pids)


def read_holder(marker):
    try:
        return json.loads((Path(marker) / HOLDER).read_text())
    except (OSError, ValueError):
        return None


def live_holder(marker, proc_root="/proc", now=None):
    """The holder of a live marker, or None when there is none or it is stale."""
    marker = Path(marker)
    if not marker.is_dir():
        return None
    holder = read_holder(marker)
    if holder is None:
        try:
            age = (time.time() if now is None else now) - marker.stat().st_mtime
        except OSError:
            return None
        return dict(owner=None, purpose="being reserved", pid=None) if age < UNWRITTEN_GRACE_S else None
    ticks = start_ticks(holder.get("pid"), proc_root)
    if ticks is None:
        return None
    if holder.get("start_ticks") is not None and ticks != holder["start_ticks"]:
        return None  # the pid now belongs to another process
    return holder


def describe(holder):
    quiet = " (quiet)" if holder.get("quiet") else ""
    return (f"local GPU reserved by {holder.get('owner')} for {holder.get('purpose')!r}{quiet} "
            f"(pid {holder.get('pid')}, since {holder.get('started')})")


def remove_stale(marker, proc_root="/proc"):
    """Remove the marker if it exists and is stale; report it on stderr."""
    marker = Path(marker)
    if marker.is_dir() and live_holder(marker, proc_root) is None:
        holder = read_holder(marker) or {}
        print(f"local_gpu: removing a stale reservation ({describe(holder)}; holder process gone)", file=sys.stderr)
        _remove_marker(marker)


def _remove_marker(marker):
    """Free the marker's path at once: rename the directory aside, then delete it.

    Deleting in place could leave the directory behind: on NFS a ``holder.json`` that another process has
    open is renamed to ``.nfsXXXX`` instead of deleted, ``rmdir`` then fails, and the empty marker read as
    a reserver between mkdir and write for ``UNWRITTEN_GRACE_S``. That stalled every waiter for 60 s (the
    queue test with its temporary directory on murphy's NFS, suite-speed, 27 October 2026). A directory
    renamed aside is out of the way however long its deletion takes; later removals retry the leftovers.
    """
    marker = Path(marker)
    aside = marker.with_name(f".{marker.name}.removed-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}")
    try:
        marker.rename(aside)
    except FileNotFoundError:
        return
    except OSError:  # not renamable here: delete in place, as before
        shutil.rmtree(marker, ignore_errors=True)
        return
    for leftover in [aside, *marker.parent.glob(f".{marker.name}.removed-*")]:
        shutil.rmtree(leftover, ignore_errors=True)


def reserve(marker, purpose, pid, proc_root="/proc", user=None, now=None, quiet=False):
    """(True, holder) if this call created the marker; (False, live holder) if someone else holds it.
    ``quiet`` (a timing run) is recorded in the holder; it then blocks CPU-only suites too."""
    marker = Path(marker)
    for _ in range(2):
        try:
            marker.mkdir()
        except FileExistsError:
            holder = live_holder(marker, proc_root)
            if holder is not None:
                return False, holder
            remove_stale(marker, proc_root)
            continue
        holder = dict(owner=user or getpass.getuser(), purpose=purpose, pid=pid,
                      start_ticks=start_ticks(pid, proc_root), started=now or _now())
        if quiet:
            holder["quiet"] = True
        partial = marker / (HOLDER + ".partial")
        partial.write_text(json.dumps(holder) + "\n")
        partial.replace(marker / HOLDER)
        return True, holder
    return False, live_holder(marker, proc_root) or dict(purpose="unknown (raced)")


def release(marker, pid):
    """Remove the marker if ``pid`` holds it. True if it was removed."""
    marker = Path(marker)
    holder = read_holder(marker)
    if holder is None or holder.get("pid") != pid:
        return False
    _remove_marker(marker)
    return True


def queue_dir(marker):
    """The waiters' queue, next to the marker: ``~/.fastmace-local-gpu-queue``."""
    marker = Path(marker)
    return marker.with_name(marker.name + QUEUE_SUFFIX)


def _write_atomically(path, data):
    partial = path.with_name("." + path.name + ".partial")
    partial.write_text(json.dumps(data) + "\n")
    partial.replace(path)


def enqueue(marker, purpose, pid, waiter, proc_root="/proc", user=None, arrival_ns=None):
    """Append a ticket for the waiting process ``waiter`` (reserving for ``pid``). (path, ticket)."""
    queue = queue_dir(marker)
    queue.mkdir(parents=True, exist_ok=True)
    arrival_ns = time.time_ns() if arrival_ns is None else arrival_ns
    ticket = dict(owner=user or getpass.getuser(), purpose=purpose, pid=pid, waiter=waiter,
                  start_ticks=start_ticks(waiter, proc_root), arrival_ns=arrival_ns)
    # The name sorts in arrival order; the pid breaks ties.
    path = queue / f"{arrival_ns:020d}-{waiter}.json"
    _write_atomically(path, ticket)
    return path, ticket


def dequeue(path):
    Path(path).unlink(missing_ok=True)


def _alive(pid, ticks, proc_root):
    now_ticks = start_ticks(pid, proc_root)
    return now_ticks is not None and (ticks is None or now_ticks == ticks)


def live_tickets(marker, proc_root="/proc"):
    """The live tickets, oldest first, as (path, ticket). A stale ticket (its waiter gone, or its pid now
    another process's) is reported and removed."""
    try:
        paths = sorted(p for p in queue_dir(marker).iterdir() if p.suffix == ".json" and not p.name.startswith("."))
    except OSError:
        return []
    live = []
    for path in paths:
        try:
            ticket = json.loads(path.read_text())
        except FileNotFoundError:
            continue  # its waiter left meanwhile
        except (OSError, ValueError):
            ticket = None
        if ticket is not None and _alive(ticket.get("waiter"), ticket.get("start_ticks"), proc_root):
            live.append((path, ticket))
            continue
        print(f"local_gpu: removing a stale queue ticket {path.name} ({describe_ticket(ticket or {})}; "
              "waiter process gone)", file=sys.stderr)
        dequeue(path)
    return live


def describe_ticket(ticket):
    return f"{ticket.get('purpose')!r} by {ticket.get('owner')}, pid {ticket.get('pid')}"


def take_turn(marker, path, ticket, proc_root="/proc", quiet=False):
    """One poll of a queued waiter: reserve only if its ticket is the oldest live one. (True, holder) on
    success (the ticket is then removed); else (False, what it waits for)."""
    if not path.exists():
        _write_atomically(path, ticket)  # removed by someone else: back at its place in the queue
    tickets = live_tickets(marker, proc_root)
    ahead = [t for p, t in tickets if p.name < path.name]
    if ahead:
        return False, f"{len(ahead)} waiter(s) ahead in the queue, first {describe_ticket(ahead[0])}"
    ok, holder = reserve(marker, ticket["purpose"], ticket["pid"], proc_root, quiet=quiet)
    if not ok:
        return False, describe(holder)
    dequeue(path)
    return True, holder


def queue_lines(marker, proc_root="/proc"):
    """The queue for ``status``: a header and one line per live ticket, oldest first, with its age."""
    tickets = live_tickets(marker, proc_root)
    if not tickets:
        return ["queue empty"]
    lines = [f"queue ({len(tickets)} waiting, first served first):"]
    for i, (_, ticket) in enumerate(tickets, 1):
        age = max(0, int(time.time() - ticket.get("arrival_ns", 0) / 1e9))
        lines.append(f"  {i}. {describe_ticket(ticket)}, waiting {age // 60}m{age % 60:02d}s")
    return lines


def machine_reasons(proc_root="/proc", own_pid=None, min_available_gib=None, quiet=False):
    """Why the machine is not free: other pytest processes (only those that can use the GPU, unless
    ``quiet``), and less RAM than ``min_available_gib``."""
    reasons = []
    others = pytest_processes(proc_root, own_pid, gpu_only=not quiet)
    if others:
        kind = "other pytest process(es)" if quiet else "other pytest process(es) that can use the GPU"
        reasons.append(f"{len(others)} {kind} run: pids {others}")
    if min_available_gib is not None:
        available = available_gib(proc_root)
        if available is None or available < min_available_gib:
            shown = "unknown" if available is None else f"{available:.1f}"
            reasons.append(f"RAM available {shown} GiB < {min_available_gib:g} GiB")
    return reasons


def wait_for_machine(proc_root="/proc", own_pid=None, min_available_gib=10.0, wait_s=3600.0, interval_s=30.0,
                     clock=time.monotonic, sleep=time.sleep, quiet=False):
    """Wait until no other pytest runs (that can use the GPU, unless ``quiet``) and enough RAM is
    available. (ok, the last reasons)."""
    start = clock()
    while True:
        reasons = machine_reasons(proc_root, own_pid, min_available_gib, quiet)
        waited = clock() - start
        if not reasons or waited >= wait_s:
            return not reasons, reasons
        sleep(min(interval_s, wait_s - waited))


def acquire(marker, purpose, pid, proc_root="/proc", wait_s=3600.0, interval_s=30.0, min_available_gib=10.0,
            clock=time.monotonic, sleep=time.sleep, quiet=False, waiter=None):
    """Queue, reserve in arrival order (waiting for the holder and earlier waiters), then wait for the
    machine; release again on failure. ``waiter`` is the waiting process (default ``pid``): its ticket is
    stale once it is gone."""
    start = clock()
    path, ticket = enqueue(marker, purpose, pid, pid if waiter is None else waiter, proc_root)
    try:
        while True:
            ok, reason = take_turn(marker, path, ticket, proc_root, quiet=quiet)
            if ok:
                break
            waited = clock() - start
            if waited >= wait_s:
                return False, [reason]
            sleep(min(interval_s, wait_s - waited))
    finally:
        dequeue(path)
    left = max(0.0, wait_s - (clock() - start))
    ok, reasons = wait_for_machine(proc_root, pid, min_available_gib, left, interval_s, clock, sleep, quiet=quiet)
    if not ok:
        release(marker, pid)
    return ok, reasons


def check(marker, proc_root="/proc", min_available_gib=None, cpu_only=False):
    """(free, reasons): no live reservation (a stale one is removed), and enough RAM if asked. With
    ``cpu_only`` (a suite run with ``CUDA_VISIBLE_DEVICES=``) only a quiet reservation blocks."""
    remove_stale(marker, proc_root)
    reasons = []
    holder = live_holder(marker, proc_root)
    if holder is not None and (not cpu_only or holder.get("quiet")):
        reasons.append(describe(holder))
    if min_available_gib is not None:
        available = available_gib(proc_root)
        if available is None or available < min_available_gib:
            shown = "unknown" if available is None else f"{available:.1f}"
            reasons.append(f"RAM available {shown} GiB < {min_available_gib:g} GiB")
    return not reasons, reasons


def _run_command(marker, argv, pid):
    """Run ``argv``; release the marker on any exit, including SIGTERM from a cancelled CI job."""
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
        release(marker, pid)
        sys.exit(128 + signum)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        child = subprocess.Popen(argv)
        return child.wait()
    finally:
        release(marker, pid)
        for s, handler in previous.items():
            signal.signal(s, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--marker", type=Path, default=None,
                        help=f"the marker directory (default $FASTMACE_LOCAL_GPU_MARKER, else {DEFAULT_MARKER})")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="reserve, run COMMAND, release")
    reserve_cmd = commands.add_parser("reserve", help="reserve for PID (release it yourself, e.g. trap EXIT)")
    for sub in (run, reserve_cmd):
        sub.add_argument("--purpose", required=True)
        sub.add_argument("--wait-minutes", type=float, default=60.0)
        sub.add_argument("--min-available-gib", type=float, default=10.0)
        sub.add_argument("--interval-seconds", type=float, default=30.0)
        sub.add_argument("--quiet", action="store_true",
                         help="a timing run: wait until no other pytest runs at all, and block CPU-only suites")
    reserve_cmd.add_argument("--pid", type=int, required=True, help="the holder, usually the calling shell ($$)")
    run.add_argument("cmd", nargs=argparse.REMAINDER, help="-- COMMAND ...")
    release_cmd = commands.add_parser("release", help="remove the marker held by PID")
    release_cmd.add_argument("--pid", type=int, required=True)
    check_cmd = commands.add_parser("check", help="exit 1 while the local GPU is reserved (or RAM is short)")
    check_cmd.add_argument("--min-available-gib", type=float, default=None)
    check_cmd.add_argument("--cpu-only", action="store_true",
                           help="for a suite run with CUDA_VISIBLE_DEVICES=: only a quiet reservation blocks it")
    commands.add_parser("status", help="print the holder, if any, and the queue")
    # Accept --marker after the subcommand as well.
    for sub in (run, reserve_cmd, release_cmd, check_cmd):
        sub.add_argument("--marker", type=Path, default=None, dest="sub_marker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    marker = getattr(args, "sub_marker", None) or args.marker or default_marker()

    if args.command == "status":
        holder = live_holder(marker)
        print(describe(holder) if holder else "local GPU free")
        print("\n".join(queue_lines(marker)))
        return 0
    if args.command == "check":
        free, reasons = check(marker, min_available_gib=args.min_available_gib, cpu_only=args.cpu_only)
        for reason in reasons:
            print(f"local_gpu: {reason}", file=sys.stderr)
        return 0 if free else 1
    if args.command == "release":
        if not release(marker, args.pid):
            print(f"local_gpu: pid {args.pid} holds no reservation at {marker}", file=sys.stderr)
        return 0
    pid = os.getpid() if args.command == "run" else args.pid
    ok, reasons = acquire(marker, args.purpose, pid, wait_s=args.wait_minutes * 60.0,
                          interval_s=args.interval_seconds, min_available_gib=args.min_available_gib,
                          quiet=args.quiet, waiter=os.getpid())
    if not ok:
        print("local_gpu: not reserved in time: " + "; ".join(reasons), file=sys.stderr)
        return BUSY
    print(f"local_gpu: reserved {marker} for {args.purpose!r} (pid {pid})", file=sys.stderr)
    if args.command == "reserve":
        return 0
    command = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not command:
        release(marker, pid)
        parser.error("run needs -- COMMAND ...")
    return _run_command(marker, command, pid)


if __name__ == "__main__":
    sys.exit(main())
