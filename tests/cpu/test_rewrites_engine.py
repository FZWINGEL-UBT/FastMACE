"""Engine-level rewrites and their process-wide patches (SPEC §6.2; plan task 1a-T08).

The strain tests are ported from the research engine's ``tests/core/test_upstream_mace.py``
(ResearchModule tag research-engine-baseline-20261017) with their tolerances. Every patch is
installed through ``fastmace._patches``: once, recorded in ``provenance()``, and undone by
``restore``; the fixture below undoes them after each test.
"""

import copy
from unittest import mock

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator
from fastmace import _patches
from fastmace.config import EngineConfig, RewriteConfig
from fastmace.rewrites import engine
from fastmace.rewrites import registry as reg

NAMES = ("capture_safe_mask_head", "first_derivative_only", "unfuse_full_addmm", "single_graph_strain")
SETTERS = {"capture_safe_mask_head": _patches.set_capture_safe_mask_head,
           "first_derivative_only": _patches.set_first_derivative_only,
           "unfuse_full_addmm": _patches.set_unfuse_full_addmm,
           "single_graph_strain": _patches.set_single_graph_strain}


@pytest.fixture(autouse=True)
def _undo_patches():
    yield
    for name in NAMES:
        _patches.restore(name)


def targets_now(record):
    import importlib
    return [getattr(importlib.import_module(t.split(":")[0]), t.split(":")[1]) for t in record.targets]


# --------------------------------------------------------------------------- patch registry


@pytest.mark.parametrize("name", NAMES)
def test_patch_installs_once_is_recorded_and_restores(name):
    import importlib

    before = {}
    record = SETTERS[name](True)
    if record is None:
        pytest.skip(f"{name} is not needed on torch {torch.__version__}")
    for target in record.targets:
        module, attribute = target.split(":")
        before[target] = _patches._ORIGINALS[target]
        assert getattr(importlib.import_module(module), attribute) is not before[target]
    assert SETTERS[name](True) is record  # idempotent: nothing changes
    assert [r.name for r in _patches.applied()].count(name) == 1
    entry = next(item for item in _patches.provenance() if item["name"] == name)
    assert entry["value"] is True and entry["targets"] == list(record.targets) and entry["detail"]
    assert _patches.restore(name)
    for target, original in before.items():
        module, attribute = target.split(":")
        assert getattr(importlib.import_module(module), attribute) is original
    assert _patches.record_of(name) is None and not _patches.restore(name)
    assert SETTERS[name](False) is None


def test_targets():
    from mace.modules import blocks, models, utils

    assert targets_now(_patches.set_capture_safe_mask_head()) == [engine.capture_safe_mask_head]
    assert blocks.mask_head is engine.capture_safe_mask_head
    record = _patches.set_first_derivative_only()
    assert record.targets == ("mace.modules.utils:compute_forces", "mace.modules.utils:compute_forces_virials")
    assert utils.compute_forces is engine.first_derivative_forces
    record = _patches.set_single_graph_strain()
    assert record.targets == ("mace.modules.utils:get_symmetric_displacement",
                              "mace.modules.models:get_symmetric_displacement")
    for module in (utils, models):
        assert module.get_symmetric_displacement is engine.single_graph_symmetric_displacement


def test_install_follows_the_config():
    records = engine.install()
    names = [r.name for r in records]
    assert names[:2] == ["capture_safe_mask_head", "first_derivative_only"] and "single_graph_strain" in names
    for name in NAMES:
        _patches.restore(name)
    records = engine.install(EngineConfig(first_derivative_only=False, unfuse_full_addmm=False),
                             RewriteConfig(single_graph_strain="off"))
    assert [r.name for r in records] == ["capture_safe_mask_head"]
    assert [r.name for r in engine.install(oeq=True) if r.name == "first_derivative_only"] == []


def test_refuses_other_signatures():
    def other(positions, unit_shifts, cell, edge_index, num_graphs, batch):
        raise AssertionError("not called")

    key = "mace.modules.utils:get_symmetric_displacement"
    with mock.patch.dict(_patches._ORIGINALS, {key: other}):
        with pytest.raises(RuntimeError, match="0.3.16 parameters"):
            _patches.set_single_graph_strain()
    assert _patches.record_of("single_graph_strain") is None

    def forces(energy, positions):
        raise AssertionError("not called")

    with mock.patch.dict(_patches._ORIGINALS, {"mace.modules.utils:compute_forces": forces}):
        with pytest.raises(RuntimeError, match="compute_forces"):
            _patches.set_first_derivative_only()


# --------------------------------------------------------------------------- records


def test_records():
    assert set(engine.PROCESS_RECORDS) == set(NAMES) and set(engine.RECORDS) == {"single_graph_strain"}
    for name, record in engine.PROCESS_RECORDS.items():
        assert isinstance(record, reg.Rewrite) and record.name == name
        assert reg.qualify(record).supported, name  # pinned MACE sources match the installed 0.3.16
    assert engine.PROCESS_RECORDS["single_graph_strain"].exactness == "rounding"
    assert all(engine.PROCESS_RECORDS[n].exactness == "exact" for n in NAMES[:3])
    assert engine.CONFIG_FIELDS["first_derivative_only"] == "EngineConfig.first_derivative_only"
    registry = reg.Registry()
    registry.register(engine.RECORDS["single_graph_strain"])
    assert registry.config_field("single_graph_strain") == "single_graph_strain"


@pytest.mark.parametrize("name", NAMES)
def test_record_apply_installs_and_selfcheck_passes(name):
    record = engine.PROCESS_RECORDS[name]
    record.apply(None, reg.Context())
    if name != "unfuse_full_addmm" or _patches.record_of(name) is not None:
        assert _patches.record_of(name) is not None
    result = reg.CheckResult.coerce(record.selfcheck(None, None, reg.Context()))
    assert result.passed and result.max_relative_error <= 1e-10, result


def test_strain_through_the_registry():
    calc = tiny_mace_calculator([11, 17])
    registry = reg.Registry([engine.RECORDS["single_graph_strain"]])
    resolution = registry.resolve(calc.models[0], RewriteConfig())
    assert resolution.enabled == ("single_graph_strain",)
    applied = registry.apply(calc.models[0], resolution, original=calc.models[0])
    assert applied[0].result.passed and applied[0].result.modules == 2
    assert _patches.record_of("single_graph_strain") is not None


# --------------------------------------------------------------------------- mask head, forces


def test_capture_safe_mask_head_equals_mace():
    from mace.modules.irreps_tools import mask_head

    generator = torch.Generator().manual_seed(0)
    for heads, channels in ((1, 3), (4, 6)):
        x = torch.randn((17, heads * channels), generator=generator, dtype=torch.float64)
        head = torch.randint(heads, (17,), generator=generator)
        assert torch.equal(engine.capture_safe_mask_head(x, head, heads), mask_head(x, head, heads))


def tiny_outputs(training):
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    atoms.set_cell(atoms.cell.array @ np.array([[1.01, 0.02, 0.0], [0.02, 0.99, 0.01], [0.0, 0.01, 1.0]]),
                   scale_atoms=True)
    calc = tiny_mace_calculator([11, 17])
    batch = calc._atoms_to_batch(atoms)
    return calc.models[0](batch.to_dict(), training=training, compute_force=True, compute_stress=True)


@pytest.mark.parametrize("training", [True, False])
def test_first_derivative_forces_equal_mace(training):
    tiny_outputs(training)  # throwaway evaluation (docs/development.md)
    expected = tiny_outputs(training)
    _patches.set_first_derivative_only()
    got = tiny_outputs(training)
    # Not bitwise with training=True: create_graph=True records other backward formulas (rounding).
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(got[key].detach(), expected[key].detach(), rtol=1e-12, atol=1e-14)
    if training:  # MACE keeps the second-derivative graph; the replacement does not
        assert expected["forces"].requires_grad and not got["forces"].requires_grad


def test_unfused_addmm_rule():
    post_grad = mock.Mock()
    post_grad._is_bias_like_addmm_input.return_value = False
    rule = engine.unfused_full_addmm_rule(lambda match: False, post_grad)
    match = mock.Mock()
    match.kwargs = {"inp": mock.Mock()}
    match.kwargs["inp"].meta = {"val": mock.Mock(is_cuda=True)}
    assert rule(match) is True and rule.unfused_full_addmm
    post_grad._is_bias_like_addmm_input.return_value = True
    assert rule(match) is False  # bias-like: torch's decision stands
    match.kwargs["inp"].meta["val"].is_cuda = False
    post_grad._is_bias_like_addmm_input.return_value = False
    assert rule(match) is False  # CPU: torch's decision stands
    assert engine.unfused_full_addmm_rule(lambda match: True, post_grad)(match) is True


def test_unfuse_sets_the_cache_key_and_refuses_another_pass():
    pytest.importorskip("torch._inductor.fx_passes.post_grad")
    from torch._inductor import config as inductor_config

    original = inductor_config.post_grad_custom_pre_pass
    record = _patches.set_unfuse_full_addmm()
    if record is None:
        pytest.skip("not needed on this torch")
    assert inductor_config.post_grad_custom_pre_pass.uuid() == engine.UNFUSED_ADDMM_UUID
    _patches.restore("unfuse_full_addmm")
    assert inductor_config.post_grad_custom_pre_pass is original
    with mock.patch.object(inductor_config, "post_grad_custom_pre_pass", object()):
        with pytest.raises(RuntimeError, match="already in use"):
            _patches.set_unfuse_full_addmm()


# --------------------------------------------------------------------------- strain (ported)


def mace_symmetric_displacement():
    from mace.modules import utils

    return _patches._original(utils, "get_symmetric_displacement")


@pytest.mark.parametrize("seed,strain", [(0, 0.0), (1, 0.0), (2, 0.03), (3, 0.03)])
def test_one_graph_matches_mace(seed, strain):
    inputs = engine.strain_inputs(seed=seed, strain=strain)
    expected = engine.strained_outputs_and_gradients(mace_symmetric_displacement(), inputs)
    got = engine.strained_outputs_and_gradients(engine.single_graph_symmetric_displacement, inputs)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-13, atol=1e-13)


def test_one_graph_without_cell_or_displacement():
    inputs = engine.strain_inputs()
    inputs.update(cell=None, displacement=None)
    expected = mace_symmetric_displacement()(**inputs)
    got = engine.single_graph_symmetric_displacement(**inputs)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(b.detach().numpy(), a.detach().numpy(), rtol=1e-13, atol=1e-13)
    assert tuple(got[2].shape) == (1, 3, 3)


def test_several_graphs_use_mace():
    inputs = engine.strain_inputs(num_graphs=3, atoms=12)
    key = "mace.modules.utils:get_symmetric_displacement"
    original = mace_symmetric_displacement()
    with mock.patch.dict(_patches._ORIGINALS, {key: mock.Mock(wraps=original)}):
        engine.single_graph_symmetric_displacement(**inputs)
        _patches._ORIGINALS[key].assert_called_once()


def test_model_stress_unchanged():
    """A tiny MACE model through prepare_graph: energy, forces and stress as with MACE's function."""
    from mace.modules import utils

    atoms = nacl(repeat=1, rattle=0.0)
    atoms = atoms.repeat((2, 1, 1))
    atoms.set_cell(atoms.cell.array @ np.array([[1.01, 0.02, 0.0], [0.02, 0.99, 0.01], [0.0, 0.01, 1.0]]),
                   scale_atoms=True)
    atoms.rattle(0.05, seed=3)

    def evaluate():
        probe = copy.deepcopy(atoms)
        probe.calc = tiny_mace_calculator([11, 17])
        return probe.get_potential_energy(), probe.get_forces(), probe.get_stress(voigt=False)

    evaluate()  # throwaway evaluation
    e0, f0, s0 = evaluate()
    _patches.set_single_graph_strain()
    # prepare_graph looks the function up at call time, so the model really takes the new path.
    with mock.patch.object(utils, "get_symmetric_displacement",
                           wraps=engine.single_graph_symmetric_displacement) as installed:
        e1, f1, s1 = evaluate()
    installed.assert_called_once()
    assert installed.call_args.kwargs["num_graphs"] == 1
    assert np.abs(s0).max() > 1e-5
    assert e1 == pytest.approx(e0, abs=1e-12)
    np.testing.assert_allclose(f1, f0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s1, s0, rtol=0, atol=1e-12)
