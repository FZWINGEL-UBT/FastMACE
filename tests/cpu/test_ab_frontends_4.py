"""``tools/ab_frontends.py --integrator``: the phase 4 ``fastmace.ase`` classes against ``Simulation`` and
``fm.md`` in one process (plan task 4-T10; SPEC §7.7's acceptance, the A/B harness on the CPU).

The GPU timing itself is a run of the tool (``docs/validation/4-T10.md``); here each integrator's three
sides build on a tiny model, get the same plan and gate, and the ``fm.md`` and shim sides advance alike.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fastmace.bench import step_time

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab_frontends = _load(TOOLS / "ab_frontends.py", "_tool_ab_frontends_4t10")


def test_integrator_choices_are_the_fastmace_ase_classes():
    import fastmace.ase
    from fastmace.ase import md

    assert set(ab_frontends.INTEGRATORS) == set(md.MAPPED) - {"LangevinBAOAB"}
    for name in ab_frontends.INTEGRATORS:
        assert hasattr(fastmace.ase, name)
    args = ab_frontends.parse_args(["--output", "o", "--integrator", "MTKNPT"])
    assert args.integrator == "MTKNPT"
    assert ab_frontends.parse_args(["--output", "o"]).integrator == "Langevin"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("integrator", [name for name in ab_frontends.INTEGRATORS if name != "Langevin"])
def test_three_sides_per_integrator_on_the_cpu(tmp_path, monkeypatch, integrator):
    pytest.importorskip("cuequivariance_torch")
    import torch

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.model import loading
    from fastmace.planner import minimal

    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    loading._LICENCES_SHOWN.add(str(path))
    loaded = loading.load_pristine(str(path), trusted=True)
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    args = ab_frontends.parse_args(["--output", str(tmp_path / "runs"), "--integrator", integrator])
    ab_frontends.thermalize(atoms, args.temperature_K, args.seed)
    built = {name: ab_frontends.BUILDERS[name](args, atoms, loaded) for name in args.sides}
    sides = {name: ab_frontends.ChunkSide(runner, synchronize=lambda: None)
             for name, (_, runner, _, _) in built.items()}
    result = ab_frontends.run_round_robin(sides, step_time.TimingProtocol(warmup=2, block=3, blocks=1))
    assert set(result["ratios_over_first"]) == {"md", "ase"}
    plans = {name: entry[2] for name, entry in built.items()}
    gates = {name: entry[3] for name, entry in built.items()}
    for name in ("md", "ase"):
        assert ab_frontends.plans_equal(plans["simulation"], plans[name]), name
        assert ab_frontends.gate_records_equal(gates["simulation"], gates[name]), name
    sims = {"simulation": built["simulation"][0], "md": built["md"][0].simulation, "ase": built["ase"][0].simulation}
    assert type(sims["ase"].integrator).__name__ == type(sims["simulation"].integrator).__name__
    np.testing.assert_array_equal(sims["md"].committed().positions, sims["ase"].committed().positions)
    built["ase"][0].close()
    for name in ("simulation", "md"):
        sims[name].close()
