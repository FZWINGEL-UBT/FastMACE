"""Interface stubs of plan task 1a-T02 (docs/design/interfaces.md).

The stubs fix the cut between integrator and engine that 1a-T11 (ForceField), 1a-T12 (state and
integrators), 1a-T13 (graph engine) and 1a-T15 (planner) build against. These tests pin the
names and signatures; when a later task changes one, it changes this table with the conductor.
"""

import dataclasses
import importlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# (module, class, member) -> parameter names after `self`, or None for a property.
SIGNATURES = {
    ("fastmace.model.forces", "ForceField", "device"): None,
    ("fastmace.model.forces", "ForceField", "dtype"): None,
    ("fastmace.model.forces", "ForceField", "compute_stress"): None,
    ("fastmace.model.forces", "ForceField", "padding_exact"): None,
    ("fastmace.model.forces", "ForceField", "__call__"): ("positions",),
    ("fastmace.model.forces", "ForceField", "evaluate"): ("atoms", "stress"),
    ("fastmace.model.forces", "ForceField", "set_cell"): ("cell",),
    ("fastmace.model.forces", "ForceField", "needs_rebuild"): ("positions", "fraction"),
    ("fastmace.model.forces", "ForceField", "compact"): ("positions",),
    ("fastmace.model.forces", "ForceField", "rebuild"): ("positions",),
    ("fastmace.model.forces", "ForceField", "use_edges"): ("num_edges",),
    ("fastmace.model.forces", "ForceField", "edge_count"): None,
    ("fastmace.model.forces", "ForceField", "edge_capacity"): None,
    ("fastmace.model.forces", "ForceField", "buffers"): (),
    ("fastmace.model.forces", "ForceField", "info"): (),
    ("fastmace.integrators.base", "Integrator", "variable_cell"): None,
    ("fastmace.integrators.base", "Integrator", "set_temperature"): ("temperature_K",),
    ("fastmace.integrators.base", "Integrator", "prepare_step"): ("state",),
    ("fastmace.integrators.base", "Integrator", "first_half"): ("state",),
    ("fastmace.integrators.base", "Integrator", "second_half"): ("state",),
    ("fastmace.integrators.base", "Integrator", "start"): ("state",),
    ("fastmace.integrators.base", "Integrator", "mid_step_tensors"): ("state",),
    ("fastmace.integrators.base", "Integrator", "buffers"): ("state",),
    ("fastmace.integrators.base", "Integrator", "output_values"): ("state",),
    ("fastmace.integrators.base", "Integrator", "state_dict"): (),
    ("fastmace.integrators.base", "Integrator", "load_state_dict"): ("state_dict",),
    ("fastmace.state", "SimulationState", "from_atoms"): ("atoms", "device", "seed", "compute_stress"),
    ("fastmace.state", "SimulationState", "natoms"): None,
    ("fastmace.state", "SimulationState", "momenta"): None,
    ("fastmace.engine.base", "Engine", "step"): (),
    ("fastmace.engine.base", "Engine", "run"): ("steps",),
    ("fastmace.engine.base", "Engine", "flush"): (),
    ("fastmace.engine.base", "Engine", "snapshot"): (),
    ("fastmace.engine.base", "Engine", "sync"): (),
    ("fastmace.engine.base", "Engine", "potential_energy"): (),
    ("fastmace.engine.base", "Engine", "info"): (),
    ("fastmace.engine.base", "Snapshot", "wait"): (),
    ("fastmace.planner.plan", "Planner", "plan"): ("traits", "atoms", "integrator", "hardware"),
    ("fastmace.planner.plan", "Planner", "build"): ("plan", "model", "atoms"),
    ("fastmace.planner.plan", "Plan", "describe"): (),
}

MODULES = sorted({module for module, _, _ in SIGNATURES})


@pytest.mark.parametrize("name", MODULES)
def test_stub_module_imports_with_a_docstring(name):
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip()


@pytest.mark.parametrize("key", SIGNATURES, ids=lambda k: ".".join(k[1:]))
def test_member_has_documented_signature(key):
    module, cls_name, member = key
    cls = getattr(importlib.import_module(module), cls_name)
    assert cls.__doc__ and cls.__doc__.strip(), f"{cls_name} has no docstring"
    raw = inspect.getattr_static(cls, member)
    expected = SIGNATURES[key]
    if expected is None:
        assert isinstance(raw, property), f"{cls_name}.{member} must be a property"
        assert raw.fget.__doc__, f"{cls_name}.{member} has no docstring"
        return
    function = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    assert callable(function), f"{cls_name}.{member} is not callable"
    assert function.__doc__ and function.__doc__.strip(), f"{cls_name}.{member} has no docstring"
    parameters = list(inspect.signature(function).parameters)
    if not isinstance(raw, staticmethod):
        parameters = parameters[1:]  # self or cls
    assert tuple(parameters) == expected


def test_abstract_bases_cannot_be_instantiated():
    from fastmace.engine.base import Engine
    from fastmace.integrators.base import Integrator

    for cls in (Engine, Integrator):
        assert inspect.isabstract(cls)
        with pytest.raises(TypeError):
            cls()


def test_integrator_half_steps_are_abstract():
    from fastmace.integrators.base import Integrator

    required = {"set_temperature", "prepare_step", "first_half", "second_half", "mid_step_tensors",
                "buffers", "output_values", "state_dict", "load_state_dict", "variable_cell"}
    assert required <= Integrator.__abstractmethods__


def test_engine_methods_are_abstract():
    from fastmace.engine.base import Engine

    assert {"step", "flush", "snapshot", "sync", "potential_energy", "info"} <= Engine.__abstractmethods__


def test_plan_describes_itself():
    from fastmace.planner.plan import Plan

    # SimulationState.from_atoms is filled by 1a-T12 (tests/cpu/test_integrators.py), Plan.describe by
    # 1a-T15 (tests/cpu/test_planner.py).
    plan = Plan(tier="graph")
    assert plan.describe() == str(plan) == "Plan revision 0, tier graph: initial plan"


def test_plan_is_a_frozen_record():
    from fastmace.planner.plan import Plan

    assert dataclasses.is_dataclass(Plan)
    plan = Plan(tier="graph")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.tier = "compiled"
    fields = {f.name for f in dataclasses.fields(Plan)}
    assert {"tier", "rewrites", "engine", "precision", "neighbours", "polar", "integrator",
            "decisions", "fallbacks", "predicted_memory_bytes", "revision", "predecessor",
            "reason"} <= fields


def test_state_fields_of_phase_1a():
    from fastmace.state import RandomStream, SimulationState

    fields = {f.name for f in dataclasses.fields(SimulationState)}
    assert {"positions", "velocities", "masses", "cell", "species", "forces", "energy", "stress",
            "rng", "step"} <= fields
    assert {"generator", "batch", "cursor"} <= {f.name for f in dataclasses.fields(RandomStream)}
    # 1b-T05 added the §10.6 fields and 3-T08 the MC state: nothing is listed as missing.
    import fastmace.state as state

    assert {"charges", "total_charge", "spin_multiplicity", "pbc", "constraints", "p_eps", "force_eps",
            "protocol", "plan_revision", "force_field_identity"} <= fields
    assert state.FIELDS_ADDED_IN_1B == {} and "mc" in fields  # 3-T08: SimulationState.mc


def test_force_field_protocol_is_structural():
    """A class with the protocol's members passes isinstance (runtime_checkable), others do not."""
    from fastmace.model.forces import ForceField

    members = {name for (module, cls, name) in SIGNATURES if cls == "ForceField"}
    duck = type("Duck", (), {name: (lambda self, *a, **k: None) for name in members})
    assert isinstance(duck(), ForceField)
    assert not isinstance(object(), ForceField)


def test_stub_modules_do_not_import_torch_or_the_engine():
    """The interface modules import cheaply: no torch, and integrators/model never pull in engine."""
    code = textwrap.dedent(
        """
        import sys
        import fastmace.integrators.base, fastmace.model.forces, fastmace.state
        print(sorted(m for m in sys.modules if m == "torch" or m.startswith("fastmace.engine")))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, capture_output=True,
                            text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
