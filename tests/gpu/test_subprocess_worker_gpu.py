"""The persistent subprocess worker on the GPU (plan task OPT-T07; docs/design/opt-T07-persistent-worker.md).

MH-1 (``mh1-r2scan``) on NaCl 3x3x3, float64, Langevin at 300 K, 2 fs. The parent initialises CUDA
first, so a ``reproducible=True`` plan runs in a child (2-T16's trigger):

- **worker against fresh children, bitwise:** three ``run(100)`` calls served by one worker, and the
  same three calls with ``RunnerConfig(persistent=False)`` (a fresh child per call, 2-T07's path):
  positions and momenta bitwise equal. The plan is reproducible, so §10.6 promises it;
- **memory release:** the worker is listed by ``nvidia-smi`` while it runs and gone after
  ``close()``, and the device's free memory seen from the parent comes back within 64 MiB; when the
  parent is killed with SIGKILL while its worker is idle, the worker leaves the GPU within 30 s.
  Under WSL2, ``nvidia-smi`` reports PIDs from outside WSL's namespace, so no worker PID matches
  (4-T13): there a worker holds the GPU while its process is alive and the GPU's used memory is more
  than 64 MiB above the reading before it started, and has left it once the process is gone and the
  memory is back within 64 MiB.

Marked ``slow`` and ``acceptance``, untimed (the wall times are recorded, not judged).
"""

import importlib.util
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_worker_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]


def _wsl():
    try:
        with open("/proc/version") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def used_MiB():
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return float(output.split()[0])


def holds_gpu(pid, baseline_MiB):
    """``pid`` is listed by ``nvidia-smi``; under WSL2 (whose PIDs ``nvidia-smi`` cannot see), it is alive and
    the GPU's used memory is more than WSL_MARGIN_MIB above ``baseline_MiB``."""
    if not _wsl():
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout
        return str(pid) in output.split()
    return os.path.exists("/proc/%d" % pid) and used_MiB() > baseline_MiB + WSL_MARGIN_MIB


def released(pid, baseline_MiB):
    """``pid`` holds no GPU memory any more: not listed; under WSL2, gone and the used memory back within
    WSL_MARGIN_MIB of ``baseline_MiB``."""
    if not _wsl():
        return not holds_gpu(pid, baseline_MiB)
    return not os.path.exists("/proc/%d" % pid) and used_MiB() <= baseline_MiB + WSL_MARGIN_MIB


# Under WSL2 the used memory is the whole GPU's, the Windows desktop's included, so the margin is wider than the
# parent's 64 MiB check; a worker's CUDA context alone is several hundred MiB.
WSL_MARGIN_MIB = 128
HOLDS_GPU = "\n".join([f"WSL_MARGIN_MIB = {WSL_MARGIN_MIB}", ""] + [inspect.getsource(function) for function in
                                                                   (_wsl, used_MiB, holds_gpu, released)])

DRIVER = """
import json, os, subprocess, sys, time
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import RunnerConfig

SPEC = {spec!r}
torch.zeros(1, device="cuda")  # CUDA started before the plan: the runner trigger (2-T16)
torch.cuda.synchronize()
free_before = torch.cuda.mem_get_info()[0]
{holds_gpu}
baseline_MiB = used_MiB()

integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
ff = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64")
started = time.perf_counter()
sim = fm.Simulation(atoms, ff, integrator=integrator, seed=20261020,
                    workload=fm.Workload(max_temperature_K=300.0, reproducible=True),
                    runner=RunnerConfig(persistent=SPEC["persistent"]))
setup_s = time.perf_counter() - started
seconds = []
for _ in range(3):
    started = time.perf_counter()
    sim.run(steps=100)
    seconds.append(time.perf_counter() - started)
committed = sim.committed()
block = sim.metadata()["subprocess"]
pid = block["last"]["pid"]
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "runner": sim.plan.runner, "subprocess": block,
        "setup_s": setup_s, "seconds": seconds, "gpu": torch.cuda.get_device_name(),
        "listed_while_running": holds_gpu(pid, baseline_MiB) if SPEC["persistent"] else None}}
if SPEC["mode"] == "idle":  # the test kills this parent now
    print(json.dumps(out), flush=True)
    time.sleep(3600)
sim.close()
time.sleep(1.0)
out["listed_after_close"] = not released(pid, baseline_MiB)
out["free_change_MiB"] = (torch.cuda.mem_get_info()[0] - free_before) / 2**20
print(json.dumps(out))
"""


def driver(*, persistent, mode="close"):
    return ENGINE.run_fresh(DRIVER.format(spec=dict(persistent=persistent, mode=mode), holds_gpu=HOLDS_GPU))


def test_the_worker_equals_fresh_children_bitwise_and_releases_the_gpu(tmp_path):
    worker = driver(persistent=True)
    fresh = driver(persistent=False)
    for run in (worker, fresh):
        assert run["runner"] == "subprocess" and run["step"] == 300
    assert (worker["subprocess"]["children"], worker["subprocess"]["calls"]) == (1, 4)
    assert (fresh["subprocess"]["children"], fresh["subprocess"]["calls"]) == (4, 4)
    print(f"worker: setup {worker['setup_s']:.1f} s, calls {[round(s, 1) for s in worker['seconds']]} s; "
          f"fresh: setup {fresh['setup_s']:.1f} s, calls {[round(s, 1) for s in fresh['seconds']]} s; "
          f"free memory after close {worker['free_change_MiB']:+.0f} MiB; {worker['gpu']}")
    assert (worker["positions"], worker["momenta"]) == (fresh["positions"], fresh["momenta"]), \
        "the worker's trajectory differs from fresh children's"
    assert worker["listed_while_running"] is True and worker["listed_after_close"] is False
    assert abs(worker["free_change_MiB"]) <= 64


def _result_line(stream):
    """The driver's JSON line: libraries may print to stdout first (cuEquivariance's Triton cache
    manager prints "GPU information: ..." where pynvml is not supported, as in WSL). ``None`` at EOF."""
    for line in stream:
        if line.startswith("{"):
            return json.loads(line)
    return None


def test_a_killed_parent_releases_the_workers_gpu_memory(tmp_path):
    root = ENGINE.REPOSITORY_ROOT
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(root / "tests")]),
                       PYTHONDONTWRITEBYTECODE="1")
    code = textwrap.dedent(DRIVER.format(spec=dict(persistent=True, mode="idle"), holds_gpu=HOLDS_GPU))
    baseline_MiB = used_MiB()
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=environment, stdout=subprocess.PIPE,
                              text=True)
    try:
        out = _result_line(parent.stdout)
        assert out is not None, f"the parent exited ({parent.poll()}) without its result line"
        pid = out["subprocess"]["last"]["pid"]
        assert holds_gpu(pid, baseline_MiB)
    finally:
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=60)
    deadline = time.monotonic() + 30
    while not released(pid, baseline_MiB) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert released(pid, baseline_MiB), "the worker kept the GPU after its parent was killed"
    # The killed parent could not remove its hand-off directory (kept, as after an error); the test does.
    from fastmace import _cache
    from fastmace.engine import subprocess as runner

    for directory in (_cache.cache_root() / runner.CACHE_NAME).glob(f"{parent.pid}-*"):
        shutil.rmtree(directory, ignore_errors=True)
