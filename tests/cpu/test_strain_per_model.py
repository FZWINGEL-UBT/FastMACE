"""The one-graph strain per model (plan task 2-T08b; SPEC §8.7 "MACE patches"; docs/design/strain-per-model.md).

``single_graph_strain`` no longer patches MACE's modules. A force field binds its strain path into
its own forward: a ``prepare_graph`` with the same code and a snapshot of ``mace.modules.utils``'
globals taken at bind time, in which ``get_symmetric_displacement`` is the one-graph function (on)
or MACE's own (off), and a forward whose ``prepare_graph`` is that variant.

- the variant's globals equal the module's at bind time except the strain function;
- a rebound forward still reaches the live process-wide patches it calls through MACE's own
  functions (``get_outputs`` reads ``compute_forces`` from ``mace.modules.utils``);
- a forward that does not read ``prepare_graph`` as a global, or that reads a global some
  ``_patches`` patch replaces, is refused with the reason;
- a bound model takes its path whatever the process patch is: on equals MACE with the patch, off
  equals MACE without it, bitwise, with the process patch on or off;
- the record installs nothing, and a failing self-check restores nothing.
"""

import copy
import types
from unittest import mock

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator
from fastmace import _patches
from fastmace.rewrites import engine
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")


@pytest.fixture(autouse=True)
def _no_strain_patch():
    """Every test starts and ends without the process patch (the reference calculators' toggle)."""
    _patches.restore("single_graph_strain")
    yield
    _patches.restore("single_graph_strain")
    _patches.restore("first_derivative_only")


def structure():
    atoms = nacl(repeat=1)
    atoms = atoms.repeat((2, 1, 1))
    atoms.set_cell(atoms.cell.array @ np.array([[1.01, 0.02, 0.0], [0.02, 0.99, 0.01], [0.0, 0.01, 1.0]]),
                   scale_atoms=True)
    atoms.rattle(0.05, seed=3)
    return atoms


def batch(calc, atoms):
    return calc._atoms_to_batch(atoms).to_dict()


def stress_of(forward, model, data):
    out = forward(model, dict(data), training=False, compute_force=True, compute_stress=True)
    return [out["energy"].detach(), out["forces"].detach(), out["stress"].detach()]


def assert_bitwise(got, expected):
    for a, b in zip(got, expected, strict=True):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- the snapshot


@pytest.mark.parametrize("one_graph", [True, False])
def test_prepare_graph_variant_is_a_snapshot_of_utils_except_the_strain_function(one_graph):
    from mace.modules import utils

    variant = engine.strain_prepare_graph(one_graph)
    assert variant.__code__ is utils.prepare_graph.__code__
    module = vars(utils)
    assert variant.__globals__ is not module
    assert set(variant.__globals__) == set(module)
    differ = [name for name in module if variant.__globals__[name] is not module[name]]
    # Off without the process patch: MACE's own function, the same object as the module's.
    assert differ == (["get_symmetric_displacement"] if one_graph else [])
    expected = (engine.single_graph_symmetric_displacement if one_graph
                else _patches._original(utils, "get_symmetric_displacement"))
    assert variant.__globals__["get_symmetric_displacement"] is expected


def test_snapshot_is_taken_at_bind_time():
    """The process patch in force when binding does not leak into an 'off' variant."""
    from mace.modules import utils

    _patches.set_single_graph_strain(True)
    variant = engine.strain_prepare_graph(False)
    assert variant.__globals__["get_symmetric_displacement"] is _patches._original(utils, "get_symmetric_displacement")
    assert utils.get_symmetric_displacement is engine.single_graph_symmetric_displacement


@pytest.mark.parametrize("one_graph", [True, False])
def test_rebound_forward_differs_from_its_module_only_in_prepare_graph(one_graph):
    from mace.modules import models

    forward = engine.strain_forward(models.ScaleShiftMACE.forward, one_graph)
    assert forward.__code__ is models.ScaleShiftMACE.forward.__code__
    module = vars(models)
    differ = [name for name in module if forward.__globals__[name] is not module[name]]
    assert differ == ["prepare_graph"]
    assert forward.__globals__["get_outputs"] is models.get_outputs  # MACE's, reading utils' live globals


def test_rebound_forward_reaches_a_patch_installed_after_binding():
    """get_outputs reads compute_forces from mace.modules.utils at call time: a later patch reaches it."""
    from mace.modules import utils

    calc, atoms = tiny_mace_calculator([11, 17]), structure()
    model = calc.models[0]
    forward = engine.strain_forward(type(model).forward, True)
    _patches.set_first_derivative_only(True)  # after binding
    data = batch(calc, atoms)
    with mock.patch.object(utils, "compute_forces", wraps=utils.compute_forces) as live:
        forward(model, dict(data), training=True, compute_force=True, compute_stress=False)
    live.assert_called_once()


# --------------------------------------------------------------------------- refusals


def test_a_forward_without_a_global_prepare_graph_is_refused():
    def forward(self, data, **kwargs):
        return data

    problems = engine.strain_problems(forward)
    assert problems and "prepare_graph" in problems[0]
    with pytest.raises(engine.StrainBindingError, match="prepare_graph"):
        engine.strain_forward(forward, True)


def test_a_forward_reading_a_patched_global_is_refused():
    """A global that a _patches patch replaces would be frozen by the snapshot: refused."""
    from mace.modules import utils

    def reads_compute_forces(self, data, **kwargs):
        prepare_graph(data)  # noqa: F821 - resolved in utils' namespace below
        return compute_forces(data)  # noqa: F821

    forward = types.FunctionType(reads_compute_forces.__code__, vars(utils), "forward")
    problems = engine.strain_problems(forward)
    assert any("mace.modules.utils:compute_forces" in problem for problem in problems)
    with pytest.raises(engine.StrainBindingError, match="compute_forces"):
        engine.strain_forward(forward, True)


def test_mace_forwards_are_bindable():
    from mace.modules import models

    for cls in (models.MACE, models.ScaleShiftMACE):
        assert engine.strain_problems(cls.forward) == [], cls.__name__


# --------------------------------------------------------------------------- bound models


def test_bound_paths_equal_mace_with_and_without_the_patch_whatever_is_installed():
    """On equals MACE with the process patch, off equals MACE without it, bitwise, under either process state."""
    calc, atoms = tiny_mace_calculator([11, 17]), structure()
    model = calc.models[0]
    data = batch(calc, atoms)
    mace_forward = type(model).forward
    stress_of(mace_forward, model, data)  # throwaway: the first evaluation of a process rounds differently
    without = stress_of(mace_forward, model, data)
    _patches.set_single_graph_strain(True)
    with_patch = stress_of(mace_forward, model, data)
    _patches.restore("single_graph_strain")
    for installed in (False, True):
        if installed:
            _patches.set_single_graph_strain(True)
        on = engine.strain_forward(mace_forward, True)
        off = engine.strain_forward(mace_forward, False)
        assert_bitwise(stress_of(on, model, data), with_patch)
        assert_bitwise(stress_of(off, model, data), without)
        _patches.restore("single_graph_strain")


def test_bind_on_a_private_copy_and_on_the_float64_forward():
    """bind_strain_path binds the instance's current forward (the float64 one included), never the class."""
    from fastmace.model import energy as energy64

    calc = tiny_mace_calculator([11, 17])
    source = calc.models[0]
    private = copy.deepcopy(source)
    energy64.install(private)
    engine.bind_strain_path(private, True)
    assert "forward" in private.__dict__ and "forward" not in source.__dict__
    assert type(private).forward is type(source).forward  # the class is untouched
    bound = private.forward.__func__
    assert bound.__code__ is energy64.float64_forward(source).__code__
    assert bound.__globals__["prepare_graph"].__globals__["get_symmetric_displacement"] is \
        engine.single_graph_symmetric_displacement
    assert engine.strain_path(private) == "one_graph"
    plain = copy.deepcopy(source)
    engine.bind_strain_path(plain, False)
    assert engine.strain_path(plain) == "mace" and engine.strain_path(source) is None


# --------------------------------------------------------------------------- the record


def test_the_record_installs_nothing_and_its_selfcheck_passes():
    record = engine.RECORDS["single_graph_strain"]
    detail = record.apply(None, reg.Context())
    assert not isinstance(detail, _patches.PatchRecord)
    assert _patches.record_of("single_graph_strain") is None
    result = reg.CheckResult.coerce(record.selfcheck(None, None, reg.Context()))
    assert result.passed and result.max_relative_error <= 1e-10, result


def test_a_failing_strain_selfcheck_restores_no_patch():
    import dataclasses

    from fastmace.errors import SelfCheckError

    registry = reg.default_registry()
    broken = dataclasses.replace(registry["single_graph_strain"],
                                 selfcheck=lambda o, r, c: reg.CheckResult(False, 1.0, 1e-10, 1, "no", "float64"))
    registry._records["single_graph_strain"] = broken
    resolution = reg.Resolution((reg.Decision("single_graph_strain", "on", True, "on"),))
    _patches.set_single_graph_strain(True)  # a reference calculator's, say
    with pytest.raises(SelfCheckError):
        registry.apply(types.SimpleNamespace(), resolution, original=types.SimpleNamespace())
    assert _patches.record_of("single_graph_strain") is not None  # left as it was
