"""Lean neighbour buffers on CUDA (cap-nlist): bitwise the layout before, and a lower build peak.

1. List level: the Triton build, chunked (``BUILD_CHUNK``) and lean, against the layout before in one chunk, on the
   test structures; the compacted edges after several builds and compactions, byte for byte. The build's peak
   allocation above the list's own buffers is below 64 B per screened pair.
2. MD (MH-1, NaCl 3x3x3 at 1000 K, 400 steps with rebuilds, each in a fresh process; ``reproducible=True`` and
   float64, the deterministic process, as test_fused_integrator_gpu.py): ``LEAN_BUFFERS`` on (the fused
   integrator's lean kernels) and off; also with the fused integrator off (the PyTorch compaction). Positions and
   momenta bitwise equal.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from _support.structures import STRUCTURES, nbcl5
from fastmace.config import NeighbourConfig
from fastmace.neighbors import skin
from fastmace.neighbors.skin import SkinNeighbourList


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_neighbors_lean_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
CUTOFF = 5.0
EDGES = ("edge_index", "unit_shifts", "shifts", "count", "reference_positions")
CASES = [(f"{name}x3", build(repeat=3, rattle=0.05, seed=5)) for name, build in STRUCTURES.items()]
CASES.append(("NbCl5x2", nbcl5(repeat=2, rattle=0.05, seed=5)))


def make(monkeypatch, atoms, lean, mode, dtype):
    monkeypatch.setattr(skin, "LEAN_BUFFERS", lean)
    groups = atoms.numbers % 3 if mode == "groups" else None
    return SkinNeighbourList(atoms.cell.array, CUTOFF, "cuda", dtype, NeighbourConfig(skin=0.4),
                             sort_receivers=mode != "unsorted", gpu_build=True, sender_groups=groups)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("mode", ["unsorted", "sorted", "groups"])
@pytest.mark.parametrize("name,atoms", CASES, ids=[name for name, _ in CASES])
def test_lean_chunked_build_equals_legacy_on_cuda(monkeypatch, name, atoms, mode, dtype):
    lean, legacy = (make(monkeypatch, atoms, flag, mode, dtype) for flag in (True, False))
    monkeypatch.setattr(skin, "BUILD_CHUNK", 1000)
    rng = np.random.default_rng(9)
    positions = atoms.positions
    for _ in range(3):
        x = torch.tensor(positions, device="cuda")
        assert lean.build(x) == legacy.build(x)
        assert (lean.capacity, lean.num_candidates) == (legacy.capacity, legacy.num_candidates)
        assert torch.equal(lean.candidate_index.to(torch.int64), legacy.candidate_index)
        assert torch.equal(lean.candidate_unit_shifts, legacy.candidate_unit_shifts)
        moved = torch.tensor(positions + rng.uniform(-0.1, 0.1, positions.shape), device="cuda")
        lean.compact(moved)
        legacy.compact(moved)
        for key in EDGES:
            assert torch.equal(getattr(lean, key), getattr(legacy, key)), key
        positions = moved.cpu().numpy() + rng.uniform(-0.3, 0.3, positions.shape)


def test_lean_build_peak(monkeypatch):
    atoms = STRUCTURES["NaCl"](repeat=8, rattle=0.05, seed=2)  # 4,096 atoms, about 0.2M screened pairs
    monkeypatch.setattr(skin, "BUILD_CHUNK", 1 << 14)
    nl = make(monkeypatch, atoms, True, "groups", torch.float32)
    x = torch.tensor(atoms.positions, device="cuda")
    nl.build(x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    nl.build(x + 0.01)
    torch.cuda.synchronize()
    found = nl.num_candidates
    assert (torch.cuda.max_memory_allocated() - before) / found < 64


MD_DRIVER = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
from fastmace.neighbors import skin
SPEC = {spec!r}
skin.LEAN_BUFFERS = SPEC["lean"]
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261030, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(**SPEC["engine"]))
sim.run(steps=400)
committed = sim.committed()
info = dict(sim._engine.info())
nb = sim._engine.force_field.neighbours
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "fused": info.get("fused_integrator"),
        "rebuilds": info["neighbour_rebuilds"], "lean": nb.lean, "index_dtype": str(nb.candidate_index.dtype),
        "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier,
        "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "pytorch-stage"])
def test_md_trajectories_are_bitwise_equal(fused):
    engine = {} if fused else {"fused_integrator": False}
    runs = {lean: ENGINE.run_fresh(MD_DRIVER.format(spec={"lean": lean, "engine": engine})) for lean in (True, False)}
    summary = {lean: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for lean, run in runs.items()}
    ENGINE.report(f"cap-nlist-lean-mh1-nacl-{'fused' if fused else 'pytorch'}", summary)
    on, off = runs[True], runs[False]
    assert on["lean"] and not off["lean"] and on["index_dtype"] == "torch.int32", summary
    assert on["step"] == 400 and on["rebuilds"] > 1 and on["tier"] == "graph", summary
    assert (on["fused"] is True) == fused, summary
    assert (on["positions"], on["momenta"]) == (off["positions"], off["momenta"]), summary
