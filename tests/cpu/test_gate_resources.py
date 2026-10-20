"""The gate's resource policy and statuses (plan task 2-T05; SPEC §12.3 "Status" and "Resources").

- The reduced cell: the largest diagonal supercell of the structure's primitive cell (its largest
  translational repeat) that is smaller than the production cell and has every height at least
  ``reduced_cell_min_height_cutoffs`` x the cutoff; on cubic, triclinic, rattled, small and
  non-periodic structures. Smaller candidates follow, for the walk after an out-of-memory.
- The CPU estimate: a per-edge rate (plus an atoms x volume term for PolarMACE's k-space) fitted to
  two calibration boxes, never stored; the boxes are periodic cubes cut from the structure.
- The ladder, with simulated out-of-memory at each step: plan device on the production cell (on a
  CUDA plan always in a child process: a GPU out-of-memory can leave the CUDA context unusable), CPU
  float64 on the production cell within the time and memory estimate (in a child process), the
  reduced cell on the plan device (walking down in size), then on the CPU; a non-memory failure is
  ``unavailable`` at once (decision 3); a cell is never larger than one that ran out of memory.
- End to end on the CPU with a tiny model: a production-cell reference out of memory gives
  ``passed-reduced``, with the replay check (captured replay against a direct evaluation of the
  same force field) on the production cell; a replay that differs fails.
- The CPU step's own decision (calibrate, estimate, run or decline) on a tiny model.
- T15's reduced gate: one copy, a subset of the quantities.
"""

import dataclasses

import numpy as np
import pytest
import torch

from _support.structures import nacl, nbcl5
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace.config import EngineConfig, GateConfig, IntegratorConfig
from fastmace.engine import base, constraints
from fastmace.errors import GateFailed, GateUnavailable
from fastmace.planner import selfcheck
from fastmace.planner.workload import Workload

NA_CL = (11, 17)
WORKLOAD = Workload(max_temperature_K=300.0)
NVT = IntegratorConfig(temperature_K=300.0)
NPT = IntegratorConfig(ensemble="npt", pressure_GPa=0.5, temperature_K=300.0)
OOM = "CUDA out of memory. Tried to allocate 18.60 GiB"


# --------------------------------------------------------------------------- the reduced cell


def test_reduced_cell_of_cubic_nacl():
    atoms = nacl(repeat=3)
    assert selfcheck.primitive_repeat(atoms) == (3, 3, 3)
    cells = selfcheck.reduced_cells(atoms, cutoff=4.0)
    first = cells[0]
    assert first.repeat == (3, 3, 2) and first.primitive_repeat == (3, 3, 3)
    assert len(first.atoms) == 144
    np.testing.assert_allclose(first.atoms.cell.array, np.diag([16.92, 16.92, 11.28]), atol=1e-12)
    assert selfcheck.reduced_cell(atoms, cutoff=4.0) == first
    # The same crystal: the reduced cell equals the conventional cell repeated 3x3x2.
    expected = nacl(repeat=1).repeat((3, 3, 2))
    assert sorted(first.atoms.get_chemical_symbols()) == sorted(expected.get_chemical_symbols())
    assert _same_crystal(first.atoms, expected)
    # Smaller candidates, largest first, each one valid.
    sizes = [len(cell.atoms) for cell in cells]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
    assert sizes[:3] == [144, 96, 64]
    for cell in cells:
        assert min(selfcheck.heights(cell.atoms.cell.array)) >= 2 * 4.0


def test_reduced_cell_of_triclinic_nbcl5():
    atoms = nbcl5(repeat=1).repeat((2, 2, 3))
    assert selfcheck.primitive_repeat(atoms) == (2, 2, 3)
    first = selfcheck.reduced_cell(atoms, cutoff=6.0)
    assert first.repeat == (2, 2, 2)  # the c height is 6.4 Å: two of them reach 2 x 6 Å
    assert len(first.atoms) == 36 * 8
    assert _same_crystal(first.atoms, nbcl5(repeat=1).repeat((2, 2, 2)))
    assert min(selfcheck.heights(first.atoms.cell.array)) >= 12.0


def test_reduced_cell_of_a_rattled_structure():
    atoms = nacl(repeat=3, rattle=0.05, seed=4)
    assert selfcheck.primitive_repeat(atoms) == (3, 3, 3)
    first = selfcheck.reduced_cell(atoms, cutoff=4.0)
    assert first.repeat == (3, 3, 2) and len(first.atoms) == 144
    assert first.atoms.get_chemical_symbols().count("Na") == 72


def test_no_reduced_cell_when_none_is_high_enough_or_smaller():
    assert selfcheck.reduced_cells(nacl(repeat=2), cutoff=6.0) == []  # needs 3 x 5.64 Å, has 2
    assert selfcheck.reduced_cells(nacl(repeat=1), cutoff=1.0) == []  # nothing smaller than one cell


def test_no_reduced_cell_without_a_repeat():
    rng = np.random.default_rng(0)
    amorphous = nacl(repeat=3)
    amorphous.positions += rng.uniform(-1.0, 1.0, amorphous.positions.shape)
    assert selfcheck.primitive_repeat(amorphous) == (1, 1, 1)
    assert selfcheck.reduced_cells(amorphous, cutoff=4.0) == []


def test_minimum_height_in_cutoffs():
    assert selfcheck.reduced_cell(nacl(repeat=2), cutoff=5.0, min_height_cutoffs=1.0).repeat == (2, 2, 1)
    assert selfcheck.reduced_cell(nacl(repeat=3), cutoff=4.0, min_height_cutoffs=3.0) is None


def _same_crystal(one, two, tolerance=1e-6):
    """Same cell and the same atoms up to order and lattice translations."""
    if not np.allclose(one.cell.array, two.cell.array, atol=1e-9) or len(one) != len(two):
        return False
    inverse = np.linalg.inv(two.cell.array)
    for z, position in zip(one.numbers, one.positions, strict=True):
        candidates = two.positions[two.numbers == z]
        diff = (candidates - position) @ inverse
        diff -= np.round(diff)
        if np.min(np.linalg.norm(diff @ two.cell.array, axis=1)) > tolerance:
            return False
    return True


# --------------------------------------------------------------------------- the CPU estimate


def test_edge_count_matches_the_neighbour_list():
    atoms = nacl(repeat=2)
    # Rock salt within 4 Å: 6 neighbours at 2.82 Å and 12 at 3.99 Å.
    assert selfcheck.edge_count(atoms, cutoff=4.0) == 64 * 18
    assert selfcheck.edge_count(atoms, cutoff=3.0) == 64 * 6


def test_calibration_boxes_are_periodic_cubes_cut_from_the_structure():
    atoms = nbcl5(repeat=1).repeat((2, 2, 4))
    small, large = selfcheck.calibration_boxes(atoms, cutoff=5.0)
    for box, side in ((small, 10.0), (large, 10.0 * 2 ** (1 / 3))):
        np.testing.assert_allclose(box.cell.array, np.identity(3) * side)
        assert all(box.pbc)
        assert set(box.numbers) <= set(atoms.numbers)
        assert int(box.numbers.sum()) % 2 == 0  # an even electron count for a neutral singlet
    assert 1.5 < len(large) / len(small) < 3.0  # twice the volume; few atoms, so not exactly twice


def test_estimate_takes_the_largest_term_and_is_conservative():
    points = [{"edges": 1000, "atoms": 50, "atoms_volume": 1e5, "seconds": 0.002, "bytes": 1e6},
              {"edges": 2000, "atoms": 100, "atoms_volume": 4e5, "seconds": 0.005, "bytes": 2e6}]
    target = {"edges": 100_000, "atoms": 5000, "atoms_volume": 1e9}
    estimate = selfcheck.estimate_cost(points, target, polar=False)
    assert estimate["seconds"] == pytest.approx(0.25)  # the larger rate, 2.5e-6 s per edge (5e-5 per atom)
    assert estimate["bytes"] == pytest.approx(1e8)
    assert selfcheck.estimate_cost([{**p, "bytes": None} for p in points], target, polar=False)["bytes"] is None
    # A cost that grows per atom faster than per edge is not hidden by the per-edge term.
    heavy = {**target, "atoms": 20_000}
    assert selfcheck.estimate_cost(points, heavy, polar=False)["bytes"] == pytest.approx(4e8)


def test_estimate_has_an_atoms_volume_term_for_polar_models():
    a, b = 1e-6, 1e-8  # s per edge, s per atom x Å^3
    points = [{"edges": e, "atoms": n, "atoms_volume": nv, "seconds": a * e + b * nv, "bytes": None}
              for e, n, nv in ((1000, 50, 1e5), (2000, 100, 4e5))]
    target = {"edges": 50_000, "atoms": 2500, "atoms_volume": 2.5e8}
    polar = selfcheck.estimate_cost(points, target, polar=True)
    exact = a * 50_000 + b * 2.5e8
    assert polar["seconds"] >= exact  # the fit gives it exactly; the per atom x volume rate more
    assert polar["seconds"] > selfcheck.estimate_cost(points, target, polar=False)["seconds"]


@pytest.mark.skipif(__import__("os").name != "posix", reason="RLIMIT_DATA is POSIX")
def test_a_capped_process_cannot_exhaust_the_host():
    import subprocess
    import sys

    script = ("from fastmace.planner import selfcheck; import numpy; "
              "limit = selfcheck.cap_memory(256 * 2**20); assert limit is not None; "
              "numpy.ones(2**28)")  # 2 GiB past a 256 MiB cap
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode != 0 and "MemoryError" in done.stderr, done.stderr[-2000:]


# --------------------------------------------------------------------------- the ladder (simulated)


@pytest.fixture
def no_engines_yet():
    saved = (base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first)
    base._forget_production_engines()
    yield
    base._PRODUCTION_ENGINES.count, base._PRODUCTION_ENGINES.first = saved


@pytest.fixture
def pristine(tmp_path, monkeypatch):
    pytest.importorskip("mace")
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    torch.serialization.add_safe_globals([slice])

    def make(family="mh1"):
        model = tiny_mace_model(list(NA_CL), **FAMILIES[family])
        path = tmp_path / f"tiny-{family}.model"
        torch.save(model, path)
        return loading.load_pristine(str(path), trusted=True)

    return make


def cpu_planner():
    from fastmace.planner.minimal import MinimalPlanner

    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                          process=constraints.ProcessRecord())


def setup(pristine, integrator=NVT, atoms=None, family="mh1"):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.plan import Hardware

    atoms = nacl(repeat=3, rattle=0.02, seed=1) if atoms is None else atoms
    ff = ForceField(pristine(family), atoms, precision="float64")
    planner = cpu_planner()
    plan = planner.plan(ff, atoms, integrator, WORKLOAD, Hardware("cpu", (0, 0), 0))
    return planner, plan, ff, atoms


def on_gpu(plan):
    """The plan as if it ran on a GPU: only the ladder reads the device (the steps are simulated)."""
    return dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, device="cuda:0"))


class FakeSteps:
    """Stands in for ``selfcheck._run_step``: each step either runs (zeros), runs out of memory, fails,
    or (a CPU estimate step) declines. ``outcomes`` maps (device, atoms) to the outcome; default runs."""

    def __init__(self, outcomes=None, cpu="run"):
        self.outcomes = dict(outcomes or {})
        self.cpu = cpu
        self.calls = []

    def __call__(self, job, *, child):
        if "candidates" in job:  # the CPU step: calibrate, estimate, choose, run
            sizes = [len(c["copies"][0]["numbers"]) for c in job["candidates"]]
            self.calls.append(("cpu-estimate", tuple(sizes), child))
            if self.cpu == "decline":
                return [], {"declined": "estimated 2400 s above the limit of 300 s", "cpu": {"estimates": []}}
            if self.cpu == "oom":
                raise MemoryError("killed: out of memory")
            chosen = job["candidates"][0]
            return _zeros(chosen["copies"]), {"cpu": {"chosen": 0}, "reference": {"record_id": "fake"}}
        natoms = len(job["copies"][0]["numbers"])
        self.calls.append((job["device"], natoms, child))
        outcome = self.outcomes.get((job["device"], natoms), "run")
        if outcome == "oom":
            raise torch.cuda.OutOfMemoryError(OOM)
        if outcome == "fail":
            raise RuntimeError("a reference that does not run")
        return _zeros(job["copies"]), {"reference": {"record_id": "fake"}}


def _zeros(copies):
    n = len(copies[0]["numbers"])
    return [{role: (0.0, np.zeros((n, 3)), np.zeros((3, 3))) for role, _ in selfcheck.ROLES} for _ in copies]


def ladder(pristine, monkeypatch, fake, gate=None, check="gate", gpu=True):
    planner, plan, ff, atoms = setup(pristine)
    monkeypatch.setattr(selfcheck, "_run_step", fake)
    plan = on_gpu(plan) if gpu else plan
    return selfcheck.references(plan, ff, atoms, NVT, check=check, seed=0, gate=gate)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_first_step_runs_and_records_its_device(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps()
    refs = ladder(pristine, monkeypatch, fake)
    assert fake.calls == [("cuda:0", 216, True)]
    assert refs.device == "cuda:0" and refs.reduced is None
    assert [step["outcome"] for step in refs.steps] == ["ran"]
    assert refs.steps[0]["cell"] == "production" and refs.steps[0]["atoms"] == 216


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gpu_out_of_memory_goes_to_the_cpu_in_a_child(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "oom"})
    refs = ladder(pristine, monkeypatch, fake)
    assert fake.calls == [("cuda:0", 216, True), ("cpu-estimate", (216,), True)]
    assert refs.device == "cpu" and refs.reduced is None
    assert [(s["device"], s["cell"], s["outcome"]) for s in refs.steps] == [
        ("cuda:0", "production", "out of memory"), ("cpu", "production", "ran")]
    assert "out of memory" in refs.steps[0]["reason"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cpu_over_its_limit_goes_to_the_reduced_cell_on_the_gpu(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "oom"}, cpu="decline")
    refs = ladder(pristine, monkeypatch, fake)
    assert fake.calls[-1] == ("cuda:0", 144, True)
    assert refs.device == "cuda:0" and refs.reduced.repeat == (3, 3, 2)
    assert [s["outcome"] for s in refs.steps] == ["out of memory", "declined", "ran"]
    assert "2400 s" in refs.steps[1]["reason"]
    assert len(refs.reference_copies[0].atoms) == 144 and len(refs.copies[0].atoms) == 216


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_every_step_out_of_memory_is_unavailable_and_memory_never_grows(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "oom", ("cuda:0", 144): "oom", ("cuda:0", 96): "oom",
                      ("cuda:0", 64): "oom"}, cpu="oom")
    with pytest.raises(GateUnavailable) as raised:
        ladder(pristine, monkeypatch, fake)
    # Production on the GPU, then the CPU; the reduced cells on the GPU, each at most 3/4 of the last
    # one out of memory; then the CPU with only cells no larger than the last one out of memory.
    assert fake.calls == [("cuda:0", 216, True), ("cpu-estimate", (216,), True), ("cuda:0", 144, True),
                          ("cuda:0", 96, True), ("cuda:0", 64, True), ("cpu-estimate", (64,), True)]
    record = raised.value.gate_record
    assert record.status == "unavailable"
    steps = record.to_metadata()["steps"]
    assert [s["outcome"] for s in steps] == ["out of memory"] * 6
    assert "out of memory" in record.reason and "reduced" in record.reason


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_failure_that_is_not_memory_is_unavailable_at_once(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "fail"})
    refs = ladder(pristine, monkeypatch, fake, check="allow_unavailable")
    assert fake.calls == [("cuda:0", 216, True)]
    assert refs.unavailable.status == "unavailable"
    assert refs.unavailable.reason.startswith("reference failed: RuntimeError")
    assert [s["outcome"] for s in refs.unavailable.steps] == ["failed"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_zero_time_limit_never_tries_the_cpu(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "oom"})
    refs = ladder(pristine, monkeypatch, fake, gate=GateConfig(cpu_float64_time_limit_s=0.0))
    assert [call[0] for call in fake.calls] == ["cuda:0", "cuda:0"]
    assert [s["outcome"] for s in refs.steps] == ["out of memory", "skipped", "ran"]
    assert "cpu_float64_time_limit_s=0" in refs.steps[1]["reason"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reduced_cell_off_ends_after_the_production_cell(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cuda:0", 216): "oom"}, cpu="decline")
    refs = ladder(pristine, monkeypatch, fake, gate=GateConfig(reduced_cell="off"), check="allow_unavailable")
    assert len(fake.calls) == 2
    assert [s["outcome"] for s in refs.unavailable.steps] == ["out of memory", "declined", "skipped"]
    assert 'reduced_cell="off"' in refs.unavailable.steps[2]["reason"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_reduced_cell_is_recorded(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine, atoms=nacl(repeat=2, rattle=0.02, seed=1))
    fake = FakeSteps({("cuda:0", 64): "oom"}, cpu="decline")
    monkeypatch.setattr(selfcheck, "_run_step", fake)
    refs = selfcheck.references(on_gpu(plan), ff, atoms, NVT, check="allow_unavailable", seed=0)
    assert [s["outcome"] for s in refs.unavailable.steps] == ["out of memory", "declined", "skipped"]
    assert "no supercell" in refs.unavailable.steps[2]["reason"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_cpu_plan_walks_production_then_reduced_on_the_cpu(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cpu", 216): "oom"})
    refs = ladder(pristine, monkeypatch, fake, gpu=False)
    assert fake.calls == [("cpu", 216, False), ("cpu", 144, False)]
    assert refs.device == "cpu" and refs.reduced.repeat == (3, 3, 2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_steps_run_in_a_child_after_a_production_engine(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    planner.build(plan, ff, atoms)
    fake = FakeSteps({("cuda:0", 216): "oom"}, cpu="decline")
    monkeypatch.setattr(selfcheck, "_run_step", fake)
    refs = selfcheck.references(on_gpu(plan), ff, atoms, NVT, check="gate", seed=0)
    assert all(call[-1] for call in fake.calls)
    assert refs.ran == "child process"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("error", [
    RuntimeError("CUDA driver error: device not ready"),
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    selfcheck.GateJobError("the reference child process failed (exit code 1)", returncode=1,
                           why="torch.AcceleratorError: CUDA error: out of memory\nSticky error detected"),
    selfcheck.GateJobError("the reference child process failed (exit code 1)", returncode=1,
                           why="ValueError: something else"),
    selfcheck.GateJobError("the reference child process failed (exit code -9)", returncode=-9, why=None),
], ids=["driver", "cuda", "accelerator", "exit", "kill"])
def test_a_gpu_reference_child_failure_is_a_device_failure_and_the_ladder_goes_on(
        pristine, no_engines_yet, monkeypatch, error):
    def step(job, *, child):
        if "candidates" not in job and job["device"].startswith("cuda") and job["cell"] == "production":
            raise error
        return FakeSteps()(job, child=child)

    refs = ladder(pristine, monkeypatch, step)
    first = refs.steps[0]
    assert first["outcome"] in ("device failure", "out of memory")
    assert first["outcome"] == ("device failure" if "out of memory" not in str(getattr(error, "why", "")) and
                                "exit code -9" not in str(error) else "out of memory")
    text = str(getattr(error, "what", None) or error)
    assert text in first["reason"]
    if isinstance(error, selfcheck.GateJobError):
        assert first["exit_code"] == int(text.split("exit code ")[1].rstrip(")"))
        if error.why:
            assert error.why.splitlines()[0] in first["reason"]
    assert refs.steps[1]["device"] == "cpu" and refs.device == "cpu"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_failure_in_a_cpu_plan_is_still_unavailable_at_once(pristine, no_engines_yet, monkeypatch):
    fake = FakeSteps({("cpu", 216): "fail"})
    refs = ladder(pristine, monkeypatch, fake, gpu=False, check="allow_unavailable")
    assert [s["outcome"] for s in refs.unavailable.steps] == ["failed"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_reference_order_error_in_a_child_still_raises(pristine, no_engines_yet, monkeypatch):
    from fastmace.reference.calculators import ReferenceOrderError

    def step(job, *, child):
        raise selfcheck.GateJobError(
            "the reference child process failed (exit code 1)",
            why="fastmace.reference.calculators.ReferenceOrderError: the one-graph strain is off")

    with pytest.raises(ReferenceOrderError):
        ladder(pristine, monkeypatch, step)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("script, outcome", [
    ("import sys; sys.stderr.write('RuntimeError: CUDA driver error: device not ready\\n'); sys.exit(1)",
     "device failure"),
    ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", "out of memory"),
], ids=["exit", "kill"])
def test_real_child_processes_that_fail(pristine, no_engines_yet, monkeypatch, script, outcome):
    import os
    import sys

    if outcome == "out of memory" and os.name != "posix":
        pytest.skip("SIGKILL is POSIX")
    calls = []

    def command(job_path, result_path):
        calls.append(job_path)
        return [sys.executable, "-c", script]

    monkeypatch.setattr(selfcheck, "_child_command", command)
    refs = ladder(pristine, monkeypatch, selfcheck._run_step, gate=GateConfig(reduced_cell="off"),
                  check="allow_unavailable")
    steps = refs.unavailable.steps
    assert steps[0]["device"] == "cuda:0" and steps[0]["outcome"] == outcome and steps[0]["process"] == "child"
    assert steps[1]["device"] == "cpu" and len(calls) == 2  # the CPU step ran a child too
    assert refs.unavailable.seconds["step_1_s"] > 0


CAPPED_CHILD = """
import sys
from fastmace.planner import selfcheck

def allocate_past_the_cap(job):
    selfcheck.cap_memory(256 * 2**20)
    import numpy
    return numpy.ones(2**28)  # 2 GiB

selfcheck.evaluate_references = allocate_past_the_cap
sys.exit(selfcheck._child_main(sys.argv[1:]))
"""


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(__import__("os").name != "posix", reason="RLIMIT_DATA is POSIX")
def test_a_reference_child_past_its_cap_is_out_of_memory_and_the_ladder_goes_on(pristine, no_engines_yet,
                                                                                  monkeypatch):
    import sys

    monkeypatch.setattr(selfcheck, "_child_command", lambda job, result: [sys.executable, "-c", CAPPED_CHILD,
                                                                          str(job), str(result)])
    refs = ladder(pristine, monkeypatch, selfcheck._run_step, gate=GateConfig(reduced_cell="off"),
                  check="allow_unavailable")
    steps = refs.unavailable.steps
    assert [(s["device"], s["outcome"]) for s in steps[:2]] == [("cuda:0", "out of memory"), ("cpu", "out of memory")]
    assert "out of memory (capped at" in steps[1]["reason"] and steps[1]["exit_code"] == 3


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_child_out_of_memory_is_recognised(pristine, no_engines_yet, monkeypatch):
    error = selfcheck.GateJobError("the reference child process failed (exit code 1)", returncode=1,
                                   why="torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert selfcheck.is_out_of_memory(error)
    killed = selfcheck.GateJobError("the reference child process failed (exit code -9)", returncode=-9, why=None)
    assert selfcheck.is_out_of_memory(killed)
    assert not selfcheck.is_out_of_memory(RuntimeError("shape mismatch"))
    assert selfcheck.is_out_of_memory(MemoryError())


# --------------------------------------------------------------------------- end to end on the CPU


def _production_cell_out_of_memory(monkeypatch):
    original = selfcheck._run_step

    def step(job, *, child):
        if job.get("cell") == "production":
            raise torch.cuda.OutOfMemoryError(OOM)
        return original(job, child=child)

    monkeypatch.setattr(selfcheck, "_run_step", step)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("integrator", [NVT, NPT], ids=["nvt", "npt"])
def test_passed_reduced_on_the_cpu(pristine, no_engines_yet, monkeypatch, integrator):
    from fastmace import report

    planner, plan, ff, atoms = setup(pristine, integrator)
    _production_cell_out_of_memory(monkeypatch)
    refs = selfcheck.references(plan, ff, atoms, integrator, check="gate", seed=3)
    engine = planner.build(plan, ff, atoms)
    before = {name: tensor.detach().clone() for name, tensor in engine.registry.tensors().items()}
    record = selfcheck.production(engine, refs, check="gate")
    assert record.status == "passed-reduced", record.to_metadata()
    assert record.passed
    assert record.reduced_cell == (3, 3, 2) and record.device == "cpu"
    metadata = record.to_metadata()
    assert [s["outcome"] for s in metadata["steps"]] == ["out of memory", "ran"]
    assert "seconds" not in metadata["steps"][0] and set(metadata["seconds"]) >= {"step_1_s", "step_2_s"}
    assert metadata["replay"]["passed"] and metadata["replay"]["cell"] == "production"
    assert metadata["replay"]["copies"] == len(refs.copies)
    assert metadata["force_eV_per_A"] is not None
    for name, tensor in engine.registry.tensors().items():
        assert torch.equal(before[name], tensor), name
    line = report.gate_line(metadata)
    assert line.startswith("passed-reduced (") and "reduced cell 3×3×2" in line


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_trajectory_after_a_reduced_gate_equals_one_without(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    _production_cell_out_of_memory(monkeypatch)
    refs = selfcheck.references(plan, ff, atoms, NVT, check="gate", seed=3)
    engine = planner.build(plan, ff, atoms)
    selfcheck.production(engine, refs, check="gate")
    plain = planner.build(plan, ff, atoms)
    for target in (engine, plain):
        target.run(6)
        target.flush()
    np.testing.assert_array_equal(engine.state.positions.numpy(), plain.state.positions.numpy())
    assert engine.potential_energy() == plain.potential_energy()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_replay_that_differs_from_direct_evaluation_fails(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    _production_cell_out_of_memory(monkeypatch)
    refs = selfcheck.references(plan, ff, atoms, NVT, check="gate", seed=3)
    engine = planner.build(plan, ff, atoms)
    original = selfcheck._read_outputs

    def drifted(engine_, stress):  # a replayed stage whose forces are off by 1e-4 eV/Å
        energy, forces, values = original(engine_, stress)
        return energy, forces + 1e-4, values

    monkeypatch.setattr(selfcheck, "_read_outputs", drifted)
    with pytest.raises(GateFailed) as raised:
        selfcheck.production(engine, refs, check="gate")
    record = raised.value.gate_record
    assert record.status == "failed" and "replay" in record.reason
    assert not record.to_metadata()["replay"]["passed"]
    force = record.to_metadata()["replay"]["quantities"]["force_eV_per_A"]
    assert force["limit"] == record.to_metadata()["quantities"]["force_eV_per_A"]["limit"]  # the gate's own limit


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_replay_within_the_gates_limit_passes(pristine, no_engines_yet, monkeypatch):
    planner, plan, ff, atoms = setup(pristine)
    _production_cell_out_of_memory(monkeypatch)
    refs = selfcheck.references(plan, ff, atoms, NVT, check="gate", seed=3)
    engine = planner.build(plan, ff, atoms)
    original = selfcheck._read_outputs

    def rounded(engine_, stress):  # a replay off by float32-like rounding, below the gate's limit
        energy, forces, values = original(engine_, stress)
        return energy, forces + 4e-6, values

    monkeypatch.setattr(selfcheck, "_read_outputs", rounded)
    record = selfcheck.production(engine, refs, check="gate")
    replay = record.to_metadata()["replay"]
    assert record.status == "passed-reduced" and replay["passed"]
    assert replay["quantities"]["force_eV_per_A"]["difference"] == pytest.approx(4e-6, rel=1e-6)
    assert "max(2 x float32 yardstick, floor)" in replay["bound"]


# --------------------------------------------------------------------------- the CPU step itself


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("limit", [1e6, 1e-9], ids=["within", "over"])
@pytest.mark.parametrize("family", ["mh1", "polar"])
def test_cpu_step_calibrates_estimates_and_runs_or_declines(pristine, no_engines_yet, limit, family):
    from fastmace.errors import PlanRefused

    if family == "polar":
        pytest.importorskip("graph_longrange")  # PolarMACE's long-range package (not in the CPU CI venv)
    try:
        planner, plan, ff, atoms = setup(pristine, family=family)
    except PlanRefused as error:  # PolarMACE has no CPU production route; the reference has one
        pytest.skip(str(error))
    job = selfcheck.cpu_job(plan, ff, [("production", None, atoms, selfcheck.make_copies(atoms, NVT, 0))],
                            atoms, GateConfig(cpu_float64_time_limit_s=limit))
    values, info = selfcheck.evaluate_references(job)
    estimate = info["cpu"]["estimates"][0]
    assert estimate["seconds"] > 0 and estimate["edges"] == selfcheck.edge_count(atoms, 4.0)
    assert len(info["cpu"]["calibration"]) == 2
    if limit > 1:
        assert info["cpu"]["chosen"] == 0 and len(values) == 2
        assert set(values[0]) == {"reference", "yardstick"}
    else:
        assert values == [] and "above the limit" in info["declined"]


# --------------------------------------------------------------------------- the report line


def test_gate_lines_name_the_reduced_cell_and_a_later_device():
    from fastmace import report

    numbers = {"copies": "2 rattled (0.05 Å)", "energy_meV_per_atom": 0.004, "force_eV_per_A": 2e-6}
    first = {"status": "passed", "device": "cuda:0", "steps": [{"outcome": "ran"}], **numbers}
    assert report.gate_line(first) == "passed (2 rattled (0.05 Å); 0.004 meV/atom, 2e-06 eV/Å; floors §12.3)"
    cpu = {**first, "device": "cpu", "steps": [{"outcome": "out of memory"}, {"outcome": "ran"}]}
    assert report.gate_line(cpu) == ("passed (reference on cpu after out of memory; 2 rattled (0.05 Å); "
                                     "0.004 meV/atom, 2e-06 eV/Å; floors §12.3)")
    reduced = {**cpu, "status": "passed-reduced", "device": "cuda:0", "reduced_cell": [3, 3, 2]}
    assert report.gate_line(reduced) == (
        "passed-reduced (reduced cell 3×3×2 of the primitive cell on cuda:0, replay checked on the production "
        "cell; 2 rattled (0.05 Å); 0.004 meV/atom, 2e-06 eV/Å; floors §12.3)")


# --------------------------------------------------------------------------- T15's reduced gate


def test_one_copy_and_a_subset_of_quantities():
    atoms = nacl(repeat=2)
    assert [c.kind for c in selfcheck.make_copies(atoms, NVT, seed=1, count=1)] == ["rattled"]
    assert [c.kind for c in selfcheck.make_copies(atoms, NPT, seed=1, count=1)] == ["rattled", "strained"]
    rows = [{"production": {"energy_meV_per_atom": 5.0, "force_eV_per_A": 1e-4, "stress_eV_per_A3": None},
             "yardstick": {"energy_meV_per_atom": 0.1, "force_eV_per_A": 1e-4, "stress_eV_per_A3": None}}]
    assert not selfcheck.verdict(rows)["passed"]
    only_forces = selfcheck.verdict(rows, quantities=("force_eV_per_A",))
    assert only_forces["passed"] and set(only_forces["quantities"]) == {"force_eV_per_A"}
