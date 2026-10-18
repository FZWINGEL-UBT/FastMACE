"""Run phase 1a acceptance suites on this machine, one job after another (plan task 1a-T19).

    python3 tools/acceptance/run.py SUITE [SUITE ...] --machine NAME --run-id ID [--only JOB ...] [--dry-run]

Launch it detached with ``launch.sh`` (ResearchModule's AGENTS.md: long runs survive the shell that
started them). Before every GPU job it checks that no compute process is on the GPU; before a timing
job also that the GPU is idle (utilization at most --max-utilization percent; on the PC also through
Windows' nvidia-smi.exe, which sees desktop programs that WSL's does not) and records the machine's
load. A busy GPU stops the run: the job is recorded as "gpu-busy", and nothing later starts.

``<root>/<run id>/manifest.json`` records the machine, GPU, driver, environments, the FastMACE and
baseline commits and, per job, its command, environment additions, idle check, exit code and times.
Every job writes its own output directory beside it; its stdout and stderr go to ``<job>.log``.

Standard library only; it imports neither FastMACE nor torch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suites  # noqa: E402

SMI_QUERY = ["--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"]


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_gpus(text):
    """Rows of ``nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total``."""
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, driver, utilization, used, total = parts
        rows.append(dict(name=name, driver=driver, utilization_percent=_number(utilization),
                         memory_used_MiB=_number(used), memory_total_MiB=_number(total)))
    return rows


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_pids(text):
    return [int(word) for word in text.split() if word.strip().isdigit()]


def idle_verdict(compute_pids, gpus, windows_gpus=None, timing=False, max_utilization=5.0):
    """(idle, reasons): no compute process; for timing also utilization <= max_utilization (both views)."""
    reasons = []
    if compute_pids:
        reasons.append(f"compute processes on the GPU: {compute_pids}")
    if timing:
        for view, rows in (("nvidia-smi", gpus), ("Windows nvidia-smi.exe", windows_gpus or [])):
            for row in rows:
                utilization = row["utilization_percent"]
                if utilization is None or utilization > max_utilization:
                    reasons.append(f"{view}: {row['name']} utilization {utilization} % > {max_utilization:g} %")
    return not reasons, reasons


def _run(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def settled_idle_check(check, settle_s=60.0, interval_s=5.0, sleep=time.sleep):
    """``check()`` until the GPU is idle, for at most ``settle_s``: utilization is averaged over the last
    sample period, so right after the previous job exits it can still read high with no process on the
    GPU. A compute process is never waited out. Returns the last record, with the attempts made."""
    record, attempts = check(), 1
    waited = 0.0
    while not record["idle"] and not record.get("compute_pids") and waited < settle_s:
        sleep(interval_s)
        waited += interval_s
        record, attempts = check(), attempts + 1
    record["attempts"] = attempts
    return record


def idle_check(machine, timing, max_utilization):
    """The idle record of this moment (nvidia-smi, and on the PC Windows' view too) and its verdict."""
    pids = parse_pids(_run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]) or "")
    gpus = parse_gpus(_run(["nvidia-smi", *SMI_QUERY]) or "")
    windows = None
    if machine.windows_smi and os.path.exists(machine.windows_smi):
        windows = parse_gpus(_run([machine.windows_smi, *SMI_QUERY]) or "")
    idle, reasons = idle_verdict(pids, gpus, windows, timing, max_utilization)
    if not gpus:
        idle, reasons = False, reasons + ["nvidia-smi listed no GPU"]
    return dict(utc=now(), compute_pids=pids, gpus=gpus, windows_gpus=windows, load_average=_load(),
                users=(_run(["who"]) or "").split("\n")[:-1], idle=idle, reasons=reasons)


def _load():
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def commit_of(path):
    """The git commit of a checkout, else its .source_commit (NAS copy), else a COMMIT file (baseline export)."""
    path = Path(path)
    out = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    if out and out.strip():
        return out.strip()
    for marker in (".source_commit", "COMMIT"):
        if (path / marker).is_file():
            return (path / marker).read_text().strip()
    return None


def build_jobs(machine, run_root, names, research_module=None, gate_labels=None, gate_engine="fastmace"):
    jobs = []
    for name in names:
        if name == "gate":
            jobs += suites.gate_jobs(machine, run_root, gate_labels or suites.GATE_LABELS, gate_engine)
        elif name == "workflow":
            jobs += suites.workflow_jobs(machine, run_root, research_module)
        else:
            jobs += suites.SUITES[name](machine, run_root)
    return jobs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="+", choices=sorted(suites.SUITES))
    p.add_argument("--machine", required=True, choices=sorted(suites.MACHINES))
    p.add_argument("--run-id", required=True, help="Directory name under the machine's acceptance root")
    p.add_argument("--only", nargs="+", help="Run only these jobs (names as --dry-run lists them)")
    p.add_argument("--gate-labels", nargs="+", choices=suites.GATE_LABELS)
    p.add_argument("--gate-engine", choices=("fastmace", "baseline"), default="fastmace")
    p.add_argument("--research-module", help="workflow: the ResearchModule tree with 1a-T17's main.py")
    p.add_argument("--fastmace", help="The FastMACE checkout the jobs use (default: the machine's)")
    p.add_argument("--max-utilization", type=float, default=5.0, help="Idle limit for timing jobs, percent")
    p.add_argument("--dry-run", action="store_true", help="Print the jobs and write nothing")
    return p.parse_args(argv)


def main(argv=None, execute=None, check=None):
    """``execute(job, env, log_path)`` runs a job and returns its exit code; ``check(job)`` returns an idle record."""
    args = parse_args(argv)
    machine = suites.MACHINES[args.machine]
    if args.fastmace:
        machine = dataclasses.replace(machine, fastmace=args.fastmace)
    run_root = suites.check_location(machine, machine.path(machine.root) / args.run_id)
    jobs = build_jobs(machine, run_root, args.suites, args.research_module, args.gate_labels, args.gate_engine)
    if args.only:
        unknown = sorted(set(args.only) - {job.name for job in jobs})
        if unknown:
            raise SystemExit(f"unknown jobs {unknown}")
        jobs = [job for job in jobs if job.name in args.only]
    if args.dry_run:
        for job in jobs:
            print(f"{job.name}{'  [timing]' if job.timing else ''}: {job.command_line()}")
        return None
    env = suites.environment(machine, run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "FASTMACE_CACHE"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(runs=[])
    record = dict(started_utc=now(), host=socket.gethostname(), machine=machine.name, suites=args.suites,
                  run_root=str(run_root), environment=env, python=sys.executable,
                  fastmace_commit=commit_of(machine.path(machine.fastmace)),
                  baseline_commit=commit_of(machine.path(machine.baseline)),
                  environments={name: machine.python(name) for name in ("mace_cueq_oeq", "mace_cueq_latest")},
                  jobs=[])
    manifest["runs"].append(record)

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    execute = execute or _execute
    check = check or (lambda job: settled_idle_check(lambda: idle_check(machine, job.timing, args.max_utilization)))
    stopped, failures = None, []
    for job in jobs:
        entry = dict(name=job.name, argv=list(job.argv), output=job.output, timing=job.timing, note=job.note)
        record["jobs"].append(entry)
        if job.gpu:
            entry["idle_check"] = check(job)
            if not entry["idle_check"]["idle"]:
                entry["status"] = "gpu-busy"
                stopped = "GPU busy before " + job.name
                save()
                break
        entry["started_utc"] = now()
        save()
        started = time.perf_counter()
        entry["exit_code"] = execute(job, {**env, **job.env}, run_root / f"{job.name}.log")
        entry.update(finished_utc=now(), wall_s=time.perf_counter() - started,
                     status="complete" if entry["exit_code"] == 0 else "failed")
        save()
        if entry["exit_code"] != 0:
            failures.append(job.name)  # later jobs still run: each one is independent
    status = "stopped" if stopped else "failed" if failures else "complete"
    record.update(status=status, stopped=stopped, failed=failures, finished_utc=now())
    save()
    return record


def _execute(job, env, log_path):
    with open(log_path, "w") as log:
        return subprocess.run(list(job.argv), env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    main()
