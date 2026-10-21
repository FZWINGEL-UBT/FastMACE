"""The per-model strain path under torch.compile across builds (plan task 2-T19b, 2-T23's OMOL-0 failure).

``bind_strain_path`` (2-T08b) gives each force field a forward whose ``prepare_graph`` has its own strain
function. Each binding used to rebuild both functions from the same code objects with a fresh copy of their
module's globals. Dynamo caches compiled frames per code object but keeps each ``__compiled_fn_*`` in the
globals dict of the frame it compiled. So a later build, whose ``prepare_graph`` ran as its own frame (a
graph break in the forward, as in OMOL-0's), reused the cached frame in a new dict without that name:
``NameError: name '__compiled_fn_…' is not defined`` (acceptance 2-T23, local, omol0-xl, the fourth build
of the gate's production child). The rebound functions are now made once per process for each
(function, strain choice), so every build runs them in the same globals.
"""

import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator
from fastmace.rewrites import engine


def _batch():
    calculator = tiny_mace_calculator([11, 17])
    return calculator._atoms_to_batch(nacl(repeat=1, rattle=0.05, seed=2)).to_dict()


@pytest.mark.parametrize("one_graph", [True, False])
def test_strain_prepare_graph_is_made_once_per_choice(one_graph):
    first, second = engine.strain_prepare_graph(one_graph), engine.strain_prepare_graph(one_graph)
    assert first is second
    assert first.__globals__ is second.__globals__
    assert engine.strain_prepare_graph(not one_graph) is not first


@pytest.mark.parametrize("one_graph", [True, False])
def test_strain_forward_is_made_once_per_function_and_choice(one_graph):
    from mace.modules import models

    function = models.ScaleShiftMACE.forward
    first, second = engine.strain_forward(function, one_graph), engine.strain_forward(function, one_graph)
    assert first is second and first.__globals__ is second.__globals__
    assert first.__globals__["prepare_graph"] is engine.strain_prepare_graph(one_graph)
    assert engine.strain_forward(models.MACE.forward, one_graph) is not first


def test_each_rebound_function_has_its_own_code_object():
    # Dynamo caches compiled frames per code object: the two strain choices and MACE's own functions
    # (which reference calculators and other code still run) must not share one, or a frame compiled in
    # one globals dict is reused in another (the one-graph frame ran with MACE's strain, or NameError).
    from mace.modules import models, utils

    on, off = engine.strain_prepare_graph(True), engine.strain_prepare_graph(False)
    codes = {id(on.__code__), id(off.__code__), id(utils.prepare_graph.__code__)}
    assert len(codes) == 3
    assert on.__code__.co_code == utils.prepare_graph.__code__.co_code  # the same bytecode
    forwards = [engine.strain_forward(models.ScaleShiftMACE.forward, choice) for choice in (True, False)]
    assert len({id(f.__code__) for f in forwards} | {id(models.ScaleShiftMACE.forward.__code__)}) == 3


@pytest.mark.parametrize("order", [(True, False), (False, True)])
def test_both_strain_choices_compiled_in_one_process(order):
    data = _batch()
    for one_graph in order + order:
        compiled = torch.compile(engine.strain_prepare_graph(one_graph), backend="eager")
        compiled(dict(data), compute_stress=True)


def test_two_builds_bind_the_same_functions():
    from mace.modules import models

    calculator = tiny_mace_calculator([11, 17])
    one, two = (engine.bind_strain_path(engine_copy, True) for engine_copy in
                (_private(calculator.models[0]), _private(calculator.models[0])))
    assert one is not two
    assert one.forward.__func__ is two.forward.__func__
    assert one.forward.__self__ is one and two.forward.__self__ is two
    assert isinstance(one, models.ScaleShiftMACE)


def _private(model):
    from fastmace.model import compile as compiling

    return compiling.private_copy(model)


@pytest.mark.parametrize("one_graph", [True, False])
def test_prepare_graph_compiled_as_its_own_frame_in_several_builds(one_graph):
    # Each "build" binds the strain path anew and compiles its prepare_graph as a top-level frame,
    # as a graph break in the forward does; before 2-T19b the second build raised NameError.
    data = _batch()
    reference = None
    for _ in range(3):
        prepare_graph = engine.strain_prepare_graph(one_graph)
        compiled = torch.compile(prepare_graph, backend="eager")
        context = compiled(dict(data), compute_stress=True)
        if reference is None:
            reference = context.lengths.detach().clone()
        assert torch.equal(context.lengths.detach(), reference)
