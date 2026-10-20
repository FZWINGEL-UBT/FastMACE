"""The fusion family (plans/opt-1.md, OPT-T02 stage A; docs/design/opt-T02-fusion.md): ``gated_rows``.

On tiny MH-1-class cuEquivariance models (built with cuEq's CPU ``naive`` method, as
tests/cpu/test_rewrites_mh1.py does), float64, eager: the chain on rows equals the stock chain
bitwise, values and input gradients; the record is opt-in, self-checked and pinned; and the whole
model's energy and forces with the MH-1 package and the linear family are bitwise unchanged by it.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.rewrites import fusion, linear, mh1
from fastmace.rewrites import registry as reg
from fastmace.rewrites.fusion import modules as fm

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                                reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, tiny  # noqa: E402

LINEAR = ("block_linear", "grouped_linear", "merged_residual")


def registry():
    out = reg.Registry()
    for family in (mh1, linear, fusion):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def requests(gated_rows):
    """The MH-1 package and the linear family on, every other record off; ``gated_rows`` as given."""
    regs = registry()
    on = {*PACKAGE, *LINEAR}
    return {name: "on" if name in on else "off" for name in regs.names()} | {"gated_rows": gated_rows}


def rewritten(gated_rows, family="mh1"):
    model, batch = tiny(family, backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, requests(gated_rows), strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_record_is_opt_in_exact_deterministic_and_pinned():
    record = fusion.RECORDS["gated_rows"]
    assert record.default == "off" and record.exactness == "exact" and record.deterministic
    assert record.requires == ("block_linear", "merged_residual")
    assert record.source_hashes == {fusion.GATE_FORWARD: fusion.GATE_FORWARD_SHA256}
    assert reg.source_sha256(fusion.GATE_FORWARD) == fusion.GATE_FORWARD_SHA256
    regs = registry()
    model, _ = tiny("mh1", backend="cueq")
    assert not regs.resolve(copy.deepcopy(model), requests("auto"), strict=False).decision("gated_rows").enabled


def test_default_registry_holds_the_record_after_the_linear_family():
    names = reg.default_registry().names()
    assert names.index("gated_rows") > names.index("merged_residual")


def test_applies_to_gated_interactions_only():
    assert fusion.RECORDS["gated_rows"].applies_to(tiny("mh1", backend="cueq")[0]).supported
    supported, reason = fusion.RECORDS["gated_rows"].applies_to(tiny("residual", backend="cueq")[0])
    assert not supported and "GatedEquivariantBlock" in reason


def test_chain_on_rows_equals_the_stock_chain_bitwise():
    _, model, _, resolution, _ = rewritten("off")
    assert resolution.decision("merged_residual").enabled
    generator = torch.Generator().manual_seed(3)
    checked = 0
    for interaction in model.interactions:
        chain, reason = fm.gated_rows(interaction.residual_linear, interaction.equivariant_nonlin,
                                      interaction.linear_2)
        assert chain is not None, reason
        first = chain.first
        widths = [sum(m * d for _, m, d in part) for part in
                  (first.segments_in[:first.first_count], first.segments_in[first.first_count:])]
        x, y = (torch.randn((9, w), generator=generator, dtype=torch.float64).requires_grad_() for w in widths)
        grad = torch.randn((9, sum(m * d for _, m, d in chain.second.segments_out)), generator=generator,
                           dtype=torch.float64)
        stock = interaction.linear_2(interaction.equivariant_nonlin(interaction.residual_linear(x, y)))
        rows = chain(x, y)
        assert torch.equal(rows, stock)
        for a, b in zip(torch.autograd.grad(rows, (x, y), grad), torch.autograd.grad(stock, (x, y), grad)):
            assert torch.equal(a, b)
        checked += 1
    assert checked == len(model.interactions) > 0


def test_record_applies_and_selfchecks_on_the_tiny_model():
    original, model, _, resolution, applied = rewritten("on")
    assert resolution.decision("gated_rows").enabled
    entry = next(entry for entry in applied if entry.name == "gated_rows")
    assert entry.result.passed and entry.result.modules == len(model.interactions)
    # Against the unrewritten cuEq modules, so the block linears' own rounding shows (float64).
    assert entry.result.max_relative_error <= 1e-12
    assert all(isinstance(i.gated_rows, fm.GatedRows) for i in model.interactions)
    with torch.no_grad():
        model.interactions[1].gated_rows.second.weight_0.mul_(1 + 1e-8)
    assert not fusion.RECORDS["gated_rows"].selfcheck(original, model, reg.Context()).passed


def test_energy_and_forces_are_bitwise_unchanged():
    _, without, batch, _, _ = rewritten("off")
    _, with_rows, _, _, _ = rewritten("on")
    assert all(getattr(i, "gated_rows", None) is None for i in without.interactions)
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(with_rows, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)


def test_refuses_without_merged_residual_or_with_another_nonlinearity():
    _, model, _, _, _ = rewritten("off")
    interaction = model.interactions[0]
    assert fm.gated_rows(None, interaction.equivariant_nonlin, interaction.linear_2)[0] is None
    chain, reason = fm.gated_rows(interaction.residual_linear, torch.nn.SiLU(), interaction.linear_2)
    assert chain is None and "GatedEquivariantBlock" in reason
    gate = copy.deepcopy(interaction.equivariant_nonlin)
    gate._layout_is_ir_mul = False
    chain, reason = fm.gated_rows(interaction.residual_linear, gate, interaction.linear_2)
    assert chain is None and "mul_ir" in reason


def test_a_forward_that_does_not_read_it_is_left_alone():
    model, _ = tiny("mh1", backend="cueq")
    assert fusion.gate_rows(copy.deepcopy(model)) == []  # stock forward: no gated_rows attribute is read
