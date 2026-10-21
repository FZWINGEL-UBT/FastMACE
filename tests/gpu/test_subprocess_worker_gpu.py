"""The persistent subprocess worker on the GPU (plan task OPT-T07; docs/design/opt-T07-persistent-worker.md).

MH-1 (``mh1-r2scan``) on NaCl 3x3x3, float64, Langevin at 300 K, 2 fs. The parent initialises CUDA
first, so a ``reproducible=True`` plan runs in a child (2-T16's trigger):

- **worker against fresh children, bitwise:** three ``run(100)`` calls served by one worker, and the
  same three calls with ``RunnerConfig(persistent=False)`` (a fresh child per call, 2-T07's path):
  positions and momenta bitwise equal. The plan is reproducible, so §10.6 promises it;
- **memory release:** the worker is listed by ``nvidia-smi`` while it runs and gone after
  ``close()``, and the device's free memory seen from the parent comes back within 64 MiB; when the
  parent is killed with SIGKILL while its worker is idle, the worker leaves the GPU within 30 s.

Marked ``slow`` and ``acceptance``, untimed (the wall times are recorded, not judged).
"""

import importlib.util
import json
import os
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


def listed(pid):
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout
    return str(pid) in output.split()


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
        "listed_while_running": listed(pid) if SPEC["persistent"] else None}}
if SPEC["mode"] == "idle":  # the test kills this parent now
    print(json.dumps(out), flush=True)
    time.sleep(3600)
sim.close()
time.sleep(1.0)
out["listed_after_close"] = listed(pid)
out["free_change_MiB"] = (torch.cuda.mem_get_info()[0] - free_before) / 2**20
print(json.dumps(out))
"""


def driver(*, persistent, mode="close"):
    return ENGINE.run_fresh(DRIVER.format(spec=dict(persistent=persistent, mode=mode)))


def _listed(pid):
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout
    return str(pid) in output.split()


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
    code = textwrap.dedent(DRIVER.format(spec=dict(persistent=True, mode="idle")))
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=environment, stdout=subprocess.PIPE,
                              text=True)
    try:
        out = _result_line(parent.stdout)
        assert out is not None, f"the parent exited ({parent.poll()}) without its result line"
        pid = out["subprocess"]["last"]["pid"]
        assert _listed(pid)
    finally:
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=60)
    deadline = time.monotonic() + 30
    while _listed(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not _listed(pid), "the worker kept the GPU after its parent was killed"
