"""The GPU compatibility check: ``pytest tests/gpu -m compat`` with the GPU and provider recorded (plan task
4-T13; SPEC decision 39, amended 23 October 2026).

The tier (``tests/_support/tiers.py``) is the fast GPU tier plus the slow tests that check what differs between
GPU architectures: kernels compile and attach, the accuracy gate on MH-1 and a few other models, CUDA-graph
replay against eager with the kernel counts recorded, reproducible mode, a short MD smoke and a restart round
trip. About 30-60 minutes. It runs on every production GPU type each phase, and once when a type is added.

    FASTMACE_MODELS=~/models python tools/acceptance/compat.py --provider local --output <dir>
    ... --provider colab --label "Colab L4" --output <dir>

From the checkout's root, with the Python whose environment is being checked (``--python``, default this one;
a hybrid environment for the OpenEquivariance paths). Writes ``<dir>/compat.json``: provider, label, host,
GPU name, compute capability, driver, the versions of torch, CUDA, Triton, cuEquivariance, OpenEquivariance,
MACE and ASE, the FastMACE commit, the pytest command, its outcome counts, failures and durations; next to it
``junit.xml``, the tests' own records (``results/``, ``FASTMACE_RESULTS``) and, when
``tools/acceptance/resultlog.py`` is there, ``results.jsonl`` with one line per finished test. The exit code
is pytest's.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ("local", "workstation", "colab", "modal", "other")
PACKAGES = ("torch", "triton", "cuequivariance", "cuequivariance_torch", "openequivariance", "mace", "ase", "numpy")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=PROVIDERS, required=True, help="where the GPU is")
    p.add_argument("--label", default=None, help="free text for the record, e.g. 'Colab L4' (default: the GPU name)")
    p.add_argument("--output", required=True, type=Path, help="directory for the records (created)")
    p.add_argument("--python", default=sys.executable, help="the environment's Python (default: this one)")
    p.add_argument("--workers", type=int, default=0, help="pytest-xdist workers (0: none)")
    p.add_argument("pytest_args", nargs="*", help="extra pytest arguments, after --")
    return p.parse_args(argv)


def environment(python):
    """The GPU and the package versions as ``python`` sees them."""
    probe = (
        "import json, importlib.metadata as md, torch\n"
        f"packages = {list(PACKAGES)!r}\n"
        "def version(name):\n"
        "    for candidate in (name, name.replace('_', '-')):\n"
        "        try:\n"
        "            return md.version(candidate)\n"
        "        except md.PackageNotFoundError:\n"
        "            pass\n"
        "out = {'versions': {name: version(name) for name in packages}, 'cuda': torch.version.cuda,\n"
        "       'cuda_available': torch.cuda.is_available()}\n"
        "if torch.cuda.is_available():\n"
        "    out['gpu'] = torch.cuda.get_device_name(0)\n"
        "    out['capability'] = list(torch.cuda.get_device_capability(0))\n"
        "print(json.dumps(out))\n")
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=True)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    try:
        out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        out["driver"] = None
    return out


def fastmace_commit():
    if os.environ.get("FASTMACE_COMMIT"):
        return os.environ["FASTMACE_COMMIT"]
    source = ROOT / ".source_commit"
    if source.is_file():
        return source.read_text().strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pytest_command(args):
    command = [args.python, "-m", "pytest", "tests/gpu", "-m", "compat", "-p", "no:cacheprovider", "-rfEs",
               "--durations=30", f"--junitxml={args.output / 'junit.xml'}"]
    if (ROOT / "tools" / "acceptance" / "resultlog.py").is_file():
        command += ["-p", "resultlog", f"--result-log={args.output / 'results.jsonl'}"]
    if args.workers:
        command += ["-n", str(args.workers)]
    return command + list(args.pytest_args)


def summarize(junit):
    """Outcome counts, the failed and errored tests with their messages' last lines, and the 15 slowest."""
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    failures, durations = [], []
    if not Path(junit).is_file():
        return {"counts": counts, "failures": failures, "slowest": durations, "junit": None}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        seconds = float(case.get("time") or 0.0)
        durations.append((seconds, name))
        outcome = "passed"
        for kind in ("failure", "error", "skipped"):
            element = case.find(kind)
            if element is not None:
                outcome = {"failure": "failed"}.get(kind, kind)
                if kind != "skipped":
                    message = (element.get("message") or element.text or "").strip().splitlines()
                    failures.append({"test": name, "outcome": outcome, "message": message[-1] if message else ""})
                break
        counts[outcome] += 1
    durations.sort(reverse=True)
    return {"counts": counts, "failures": failures,
            "slowest": [{"test": name, "seconds": seconds} for seconds, name in durations[:15]],
            "test_seconds": sum(seconds for seconds, _ in durations), "junit": str(junit)}


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    env = environment(args.python)
    if not env.get("cuda_available"):
        raise SystemExit(f"{args.python}: CUDA is not available, nothing to check")
    if not os.environ.get("FASTMACE_MODELS"):
        raise SystemExit("set FASTMACE_MODELS to the released checkpoints (MH-1, MP-0, POLAR-1-S)")
    record = {"tool": "compat", "tier": "-m compat (tests/_support/tiers.py)", "provider": args.provider,
              "label": args.label or env.get("gpu"), "host": socket.gethostname(), "gpu": env.get("gpu"),
              "capability": env.get("capability"), "driver": env.get("driver"), "cuda": env.get("cuda"),
              "versions": env["versions"], "python": args.python, "fastmace_commit": fastmace_commit(),
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    command = pytest_command(args)
    record["command"] = command
    path = args.output / "compat.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    child_env = dict(os.environ, FASTMACE_RESULTS=str(args.output / "results"),
                     PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(ROOT / "tools" / "acceptance"),
                                                              os.environ.get("PYTHONPATH")])))
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, env=child_env).returncode
    record.update(exit_code=status, wall_s=time.perf_counter() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  **summarize(args.output / "junit.xml"))
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    counts = record["counts"]
    print(f"compat on {record['label']} ({args.provider}): {counts['passed']} passed, {counts['failed']} failed, "
          f"{counts['error']} errors, {counts['skipped']} skipped in {record['wall_s'] / 60:.1f} min ({path})")
    return status


if __name__ == "__main__":
    sys.exit(main())
