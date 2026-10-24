"""The ``polar_graph_sums`` record: PolarMACE's per-graph sums without scatter and index for one graph (opt-4 OPT4-T02).

- the one-graph helpers give MACE's ``scatter_sum(...)[batch]``, ``scatter_mean`` and ``x[batch]`` values and
  gradients (rounding: another summation order), and with several graphs they are MACE's forms, bit for bit;
- the record is rounding-class, deterministic, default only for the qualified POLAR-1 checkpoints (decision 58)
  and needs one graph per batch;
- the source edit replaces exactly the five sites of MACE 0.3.16's forward (three with PR 1762's options,
  whose Fukui normalisation takes the helpers itself), composes with the float64 energy reduction, and
  refuses another source;
- on a tiny PolarMACE (graph_longrange needed) the enabled path with the record gives the energy and forces
  of the path without it.
"""

import copy

import pytest
import torch

pytest.importorskip("mace")

from _support.structures import nacl  # noqa: E402
from _support.tiny_models import polar_available, tiny_family_calculator  # noqa: E402
from fastmace.config import PolarConfig  # noqa: E402
from fastmace.longrange import polar  # noqa: E402
from fastmace.rewrites import registry as reg  # noqa: E402
from fastmace.rewrites.polar import RECORDS, graph_sums  # noqa: E402

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4's constants under torch's weights_only load

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _mace_forms(src, batch, num_graphs):
    from mace.tools.scatter import scatter_mean, scatter_sum

    return (scatter_sum(src=src, index=batch, dim=0, dim_size=num_graphs)[batch],
            scatter_mean(src=src, index=batch, dim=0, dim_size=num_graphs))


def _ours(src, batch, num_graphs):
    return polar.graph_sum(src, batch, num_graphs), polar.graph_mean(src, batch, num_graphs)


def test_one_graph_values_and_gradients():
    generator = torch.Generator().manual_seed(4)
    src = torch.randn(216, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(216, 3, generator=generator, dtype=torch.float64)
    batch = torch.zeros(216, dtype=torch.long)
    for got, expected in zip(_ours(src, batch, 1), _mace_forms(src, batch, 1)):
        torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-14)
        g_got, = torch.autograd.grad((got * weights[:got.shape[0]]).sum(), src)
        g_expected, = torch.autograd.grad((expected * weights[:expected.shape[0]]).sum(), src)
        torch.testing.assert_close(g_got, g_expected, rtol=1e-14, atol=1e-14)
    gathered = polar.graph_gather(src[:1], batch, 1)
    assert torch.equal(gathered, src[:1][batch])


def test_several_graphs_are_maces_forms():
    src = torch.randn(10, 2, dtype=torch.float64)
    batch = torch.tensor([0] * 4 + [1] * 6)
    for got, expected in zip(_ours(src, batch, 2), _mace_forms(src, batch, 2)):
        assert torch.equal(got, expected)
    assert torch.equal(polar.graph_gather(src[:2], batch, 2), src[:2][batch])


def test_the_record_is_rounding_opt_in_and_single_graph():
    record = RECORDS["polar_graph_sums"]
    assert (record["default"], record["exactness"], record["deterministic"]) == ("off", "rounding", True)
    from fastmace.config import RewriteConfig
    from fastmace.planner import fixed

    assert fixed.SINGLE_GRAPH in record["workload_constraints"]
    assert RewriteConfig().polar_graph_sums == "auto"
    assert fixed.AUTO_RULES["RewriteConfig.polar_graph_sums"] == (
        "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints")  # decision 58
    assert "polar_graph_sums" in fixed.REWRITE_FIELDS
    assert reg.default_registry().config_field("polar_graph_sums") == "polar_graph_sums"


def _released_forward():
    import mace.modules.extensions as extensions

    polar_install = polar.installed()
    return polar._original("mace.modules.extensions:PolarMACE.forward") if polar_install \
        else extensions.PolarMACE.forward


@pytest.mark.parametrize("stabilise", [False, True])
@pytest.mark.parametrize("float64_energy", [False, True])
def test_the_source_edit_replaces_every_site(stabilise, float64_energy):
    import inspect

    original = _released_forward()
    source = inspect.getsource(original)
    if stabilise:  # graph_sums_source runs after PR 1762's edits
        for old, new in polar.PR1762_EDITS:
            source = source.replace(old, new)
    source = polar.graph_sums_source(source, stabilise=stabilise)
    assert ')[data["batch"]]' not in source and 'barycenter[data["batch"], :]' not in source
    assert "scatter_mean(" not in source
    assert source.count("_fastmace_graph_sum(") == (2 if stabilise else 4)
    forward = polar.polar_forward(original, stabilise=stabilise, float64_energy=float64_energy, graph_sums=True)
    assert callable(forward) and forward is not original


def test_the_source_edit_refuses_another_source():
    def forward(self, data):
        return data

    with pytest.raises(polar.PolarUnavailableError, match="graph sums"):
        polar.polar_forward(forward, graph_sums=True)


def test_applies_to_polar_models_only():
    calc = tiny_family_calculator("residual", [11, 17])
    support = graph_sums.applies_to(calc.models[0])
    assert not support.supported and "PolarMACE" in support.reason


def test_self_check_passes_and_catches_a_wrong_helper(monkeypatch):
    result = graph_sums.selfcheck(None, None, reg.Context())
    assert result.passed and result.modules == 3, result
    monkeypatch.setattr(polar, "graph_sum", lambda src, batch, num_graphs: 2 * src)
    assert not graph_sums.selfcheck(None, None, reg.Context()).passed


@pytest.fixture
def tiny_polar():
    if not polar_available():
        pytest.skip("PolarMACE needs graph_longrange")
    calc = tiny_family_calculator("polar", [11, 17])
    return calc, calc.models[0], nacl(repeat=1, rattle=0.05, seed=3)


def _evaluate(calc, model, atoms):
    out = model(calc._atoms_to_batch(atoms).to_dict(), compute_stress=False, training=False)
    return out["energy"].detach(), out["forces"].detach()


@needs_polar
def test_qualified_after_the_polar_wrappers_are_installed():
    # The force field installs polar's class-level wrappers before resolving the rewrites; a pin on
    # PolarMACE.forward would then hash the wrapper (the first local gate refused the record so).
    polar.install()
    record = reg.default_registry()["polar_graph_sums"]
    assert reg.qualify(record).supported
    calc = tiny_family_calculator("polar", [11, 17])
    assert graph_sums.applies_to(calc.models[0]).supported


@needs_polar
@pytest.mark.parametrize("config", [PolarConfig(), PolarConfig(fukui_regularisation=0.1)])
def test_enabled_path_with_graph_sums(tiny_polar, config):
    calc, model, atoms = tiny_polar
    plain = copy.deepcopy(model)
    polar.enable(plain, atoms.cell.array, config)
    summed = copy.deepcopy(plain)
    assert graph_sums.applies_to(summed).supported
    graph_sums.apply(summed)
    assert polar._holder(summed).graph_sums and not polar._holder(plain).graph_sums
    e0, f0 = _evaluate(calc, plain, atoms)
    e1, f1 = _evaluate(calc, summed, atoms)
    torch.testing.assert_close(e1, e0, rtol=1e-12, atol=0)
    torch.testing.assert_close(f1, f0, rtol=1e-10, atol=1e-12)
