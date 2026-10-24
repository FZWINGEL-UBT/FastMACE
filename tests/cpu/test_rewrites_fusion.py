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


def test_record_is_default_exact_deterministic_and_pinned():
    record = fusion.RECORDS["gated_rows"]
    assert record.default == "auto" and record.exactness == "exact" and record.deterministic
    assert record.requires == ("block_linear", "merged_residual")
    assert record.source_hashes == {fusion.GATE_FORWARD: fusion.GATE_FORWARD_SHA256}
    assert reg.source_sha256(fusion.GATE_FORWARD) == fusion.GATE_FORWARD_SHA256
    regs = registry()
    model, _ = tiny("mh1", backend="cueq")
    assert regs.resolve(copy.deepcopy(model), requests("auto"), strict=False).decision("gated_rows").enabled
    assert not regs.resolve(copy.deepcopy(model), requests("off"), strict=False).decision("gated_rows").enabled


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


# --------------------------------------------------------------------------- segment_gemm (OPT-T02 stage B)


def _requests_segment(segment):
    return requests("on") | {"segment_gemm": segment}


def _rewritten_segment(segment):
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, _requests_segment(segment), strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_segment_record_is_default_rounding_deterministic():
    record = fusion.RECORDS["segment_gemm"]
    assert record.default == "auto" and record.exactness == "rounding" and record.deterministic
    assert record.requires == ("gated_rows",)
    assert record.source_hashes == {fusion.RESHAPE_FORWARD: fusion.RESHAPE_FORWARD_SHA256}
    assert reg.source_sha256(fusion.RESHAPE_FORWARD) == fusion.RESHAPE_FORWARD_SHA256
    names = reg.default_registry().names()
    assert names.index("segment_gemm") > names.index("gated_rows")


def test_joined_products_on_the_cpu_equal_the_block_linear_bitwise():
    from fastmace.kernels.triton.segment_gemm import joined_products
    from fastmace.rewrites.linear.modules import join_rows

    _, model, _, _, _ = rewritten("on")
    generator = torch.Generator().manual_seed(5)
    for interaction in model.interactions:
        chain = interaction.gated_rows
        assert chain.join_in_place() == (True, "ok")
        second = chain.second
        atoms = 6
        rows = [torch.randn((atoms * second.segments_in[i][2], second.segments_in[i][1]), generator=generator,
                            dtype=torch.float64).requires_grad_() for i, _, _, _ in chain.joined]
        width = sum(m * d for _, m, d in second.segments_out)
        grad = torch.randn((atoms, width), generator=generator, dtype=torch.float64)
        stock = join_rows(second.segments_out, second.products(rows), atoms, rows[0])
        joined = joined_products(rows, [getattr(second, f"weight_{k}") for *_, k in chain.joined],
                                 [getattr(second, f"weight_t_{k}") for *_, k in chain.joined],
                                 [o for _, o, _, _ in chain.joined], [d for _, _, d, _ in chain.joined], width)
        assert torch.equal(joined, stock)
        for a, b in zip(torch.autograd.grad(joined, rows, grad), torch.autograd.grad(stock, rows, grad)):
            assert torch.equal(a, b)


def test_segment_record_applies_selfchecks_and_keeps_the_model_bitwise_on_the_cpu():
    original, model, batch, resolution, applied = _rewritten_segment("on")
    assert resolution.decision("segment_gemm").enabled
    entry = next(entry for entry in applied if entry.name == "segment_gemm")
    assert entry.result.passed and entry.result.max_relative_error <= 1e-12
    assert all(i.gated_rows.joined is not None for i in model.interactions)
    assert all(type(i.reshape).__name__ == "IrrepsView" for i in model.interactions)  # no copy before the product
    _, without, _, _, _ = rewritten("on")
    energy_and_forces(without, batch)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(model, batch)
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)  # the CPU path is PyTorch's


def test_segment_record_needs_gated_rows():
    model, _ = tiny("mh1", backend="cueq")
    regs = registry()
    asked = requests("off") | {"segment_gemm": "on"}
    assert not regs.resolve(copy.deepcopy(model), asked, strict=False).decision("segment_gemm").enabled


def test_without_a_triton_product_the_stock_join_runs():
    _, model, _, _, _ = _rewritten_segment("on")
    chain = model.interactions[0].gated_rows
    rows = [torch.zeros((4 * d, 1)) for _ in range(2) for d in (1,)]
    assert not chain._any_triton(rows, 4)  # CPU: no SMs, so the shape rule never picks Triton


# --------------------------------------------------------------------------- grouped_launch (OPT-T03 candidate 2)


def test_grouped_products_on_the_cpu_equal_one_product_at_a_time_bitwise():
    from fastmace.kernels.triton.grouped_gemm import grouped_products

    generator = torch.Generator().manual_seed(11)
    shapes = [(7, 5, 4), (21, 5, 6), (35, 3, 4), (49, 8, 2), (9, 4, 4)]  # (M, K, N); more than MAX_GROUPS
    lefts = [torch.randn((m, k), generator=generator, dtype=torch.float64).requires_grad_() for m, k, _ in shapes]
    weights = [torch.randn((k, n), generator=generator, dtype=torch.float64) for _, k, n in shapes]
    weights_t = [w.t().contiguous() for w in weights]
    grads = [torch.randn((m, n), generator=generator, dtype=torch.float64) for m, _, n in shapes]
    got = grouped_products(lefts, weights, weights_t)
    want = [x @ w for x, w in zip(lefts, weights)]
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    total = sum((o * g).sum() for o, g in zip(got, grads))
    reference = sum((o * g).sum() for o, g in zip(want, grads))
    for a, b in zip(torch.autograd.grad(total, lefts), torch.autograd.grad(reference, lefts)):
        assert torch.equal(a, b)


def _rewritten_grouped(grouped):
    model, batch = tiny("mh1", backend="cueq")
    regs = registry()
    out = copy.deepcopy(model)
    asked = _requests_segment("on") | {"grouped_launch": grouped}
    resolution = regs.resolve(out, asked, strict=False)
    applied = regs.apply(out, resolution, context(batch), original=model)
    return model, out, batch, resolution, applied


def test_grouped_launch_record_is_opt_in_and_applies_to_every_block_linear(monkeypatch):
    """With a fake one-SM device and a wave threshold no product reaches, every block linear groups all its
    products (the op's CPU path): energy and forces bitwise unchanged."""
    from fastmace.kernels.triton import grouped_gemm, tile_rule
    from fastmace.rewrites.linear.modules import BlockLinear, ResidualBlockLinear

    record = fusion.RECORDS["grouped_launch"]
    assert record.default == "off" and record.exactness == "rounding" and record.deterministic
    assert record.requires == ("block_linear",)
    original, model, batch, resolution, applied = _rewritten_grouped("on")
    assert resolution.decision("grouped_launch").enabled
    entry = next(entry for entry in applied if entry.name == "grouped_launch")
    assert entry.result.passed
    blocks = [m for m in model.modules() if isinstance(m, (BlockLinear, ResidualBlockLinear))]
    assert blocks and all(m.grouped_launch and m.sms is None for m in blocks)  # CPU: no SMs
    _, without, _, _, _ = _rewritten_segment("on")
    energy, forces = energy_and_forces(without, batch)
    calls = []
    real = grouped_gemm.grouped_products_op
    monkeypatch.setattr(grouped_gemm, "grouped_products_op", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(tile_rule, "MIN_TRITON_WAVES", 10 ** 9)
    for m in blocks:
        m.sms = 1
    new_energy, new_forces = energy_and_forces(model, batch)
    assert calls, "no grouped launch ran"
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)


def test_grouped_launch_applies_to_generic_mace(monkeypatch):
    """OPT4-T04: on a generic MACE model (MP-0 / OMAT-0 class, residual interaction) every block linear takes
    the record, its equal-shape products share a launch, and energy and forces stay bitwise unchanged."""
    from fastmace.kernels.triton import grouped_gemm, tile_rule
    from fastmace.rewrites.linear.modules import BlockLinear, ResidualBlockLinear

    model, batch = tiny("residual", backend="cueq")
    regs = registry()
    base = {name: "on" if name in LINEAR else "off" for name in regs.names()}
    without = copy.deepcopy(model)
    regs.apply(without, regs.resolve(without, base, strict=False), context(batch), original=model)
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, base | {"grouped_launch": "on"}, strict=False)
    assert resolution.decision("grouped_launch").enabled
    applied = regs.apply(out, resolution, context(batch), original=model)
    assert next(entry for entry in applied if entry.name == "grouped_launch").result.passed
    blocks = [m for m in out.modules() if isinstance(m, (BlockLinear, ResidualBlockLinear))]
    assert blocks and all(m.grouped_launch for m in blocks)
    energy, forces = energy_and_forces(without, batch)
    calls = []
    real = grouped_gemm.grouped_products_op
    monkeypatch.setattr(grouped_gemm, "grouped_products_op", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(tile_rule, "MIN_TRITON_WAVES", 10 ** 9)
    for m in blocks:
        m.sms = 1
    new_energy, new_forces = energy_and_forces(out, batch)
    assert calls, "no grouped launch ran"
    assert torch.equal(new_energy, energy) and torch.equal(new_forces, forces)


def test_grouped_launch_needs_block_linear():
    model, _ = tiny("mh1", backend="cueq")
    asked = {name: "off" for name in registry().names()} | {"grouped_launch": "on"}
    assert not registry().resolve(copy.deepcopy(model), asked, strict=False).decision("grouped_launch").enabled
