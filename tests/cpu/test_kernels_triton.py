"""fastmace.kernels.triton on the CPU: the PyTorch paths of the custom ops against autograd and the baseline.

Ported from the research engine's tests/core/test_edge_outer_sum.py, test_skinny_linear.py and
test_fused_radial.py (CPU parts). The baseline comparisons load the research engine's modules
from the export named by FASTMACE_BASELINE (ResearchModule tag research-engine-baseline-20261017)
and skip when it is unset. The CPU paths run the same operations in the same order, so they must
agree bitwise.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from fastmace.kernels.triton import edge_outer_sum as eos
from fastmace.kernels.triton import fused_radial as fr
from fastmace.kernels.triton import inplace_linear as il
from fastmace.kernels.triton import skinny_linear as sl


def baseline_module(name):
    """langevin.<name> from the baseline export, without running langevin/__init__.py."""
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no Experiments/Langevin_Thermostat/langevin"
    if "langevin" not in sys.modules:
        package = types.ModuleType("langevin")
        package.__path__ = [str(directory)]
        sys.modules["langevin"] = package
    return importlib.import_module(f"langevin.{name}")


def assert_equal(new, old):
    assert new.dtype == old.dtype and new.shape == old.shape
    assert torch.equal(new, old), f"max |difference| {(new - old).abs().max().item():.3e}"


# --------------------------------------------------------------------------- edge_outer_sum


def reference_outer_sum(a, y, key, slots):
    outer = (y[:, :, None] * a[:, None, :]).reshape(a.shape[0], -1)
    return torch.zeros((slots, outer.shape[1]), dtype=a.dtype).index_add(0, key, outer)


def outer_sum_inputs(device="cpu", dtype=torch.float64, edges=300, k=65, d=16, slots=37):
    generator = torch.Generator().manual_seed(3)
    a = torch.randn((edges, k), generator=generator, dtype=torch.float64)
    y = torch.randn((edges, d), generator=generator, dtype=torch.float64)
    key = torch.randint(0, slots, (edges,), generator=generator)
    grad = torch.randn((slots, d * k), generator=generator, dtype=torch.float64)
    return [t.to(device, dtype if t.is_floating_point() else None) for t in (a, y, key, grad)] + [slots]


def outer_sum_with_gradients(function, a, y, key, grad, slots, **kwargs):
    a_, y_ = a.clone().requires_grad_(), y.clone().requires_grad_()
    out = function(a_, y_, key, slots, **kwargs)
    out.backward(grad)
    return out.detach(), a_.grad, y_.grad


def test_edge_outer_sum_forward_and_gradients_match_autograd():
    a, y, key, grad, slots = outer_sum_inputs()
    new = outer_sum_with_gradients(eos.edge_outer_sum, a, y, key, grad, slots)
    old = outer_sum_with_gradients(reference_outer_sum, a, y, key, grad, slots)
    for n, o in zip(new, old):
        torch.testing.assert_close(n, o, rtol=0, atol=1e-12)


def test_edge_outer_sum_segmented_forward_matches_reference_for_any_key_order():
    a, y, key, grad, slots = outer_sum_inputs()
    for keys in (key, torch.sort(key).values):
        torch.testing.assert_close(eos.edge_outer_sum(a, y, keys, slots, segmented=True),
                                   reference_outer_sum(a, y, keys, slots), rtol=0, atol=1e-12)


def test_edge_outer_sum_handles_no_edges():
    a, y, key, grad, slots = outer_sum_inputs(edges=0)
    for segmented in (False, True):
        out, grad_a, grad_y = outer_sum_with_gradients(eos.edge_outer_sum, a, y, key, grad, slots,
                                                       segmented=segmented)
        assert out.shape == (slots, a.shape[1] * y.shape[1]) and not out.any()
        assert grad_a.shape == a.shape and grad_y.shape == y.shape


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("sort", [False, True])
def test_edge_outer_sum_equals_baseline(segmented, sort):
    base = baseline_module("edge_outer_sum")
    a, y, key, grad, slots = outer_sum_inputs()
    if sort:
        key = torch.sort(key).values
    new = outer_sum_with_gradients(eos.edge_outer_sum, a, y, key, grad, slots, segmented=segmented)
    old = outer_sum_with_gradients(base.edge_outer_sum, a, y, key, grad, slots, segmented=segmented)
    for n, o in zip(new, old):
        assert_equal(n, o)
    assert eos.FORWARD_CHUNK == base.FORWARD_CHUNK


# --------------------------------------------------------------------------- folded_linear

FOLDED_SHAPES = ((300, 96, 65), (77, 40, 33), (5, 17, 9))


def folded_inputs(edges, outputs, width, device="cpu", dtype=torch.float64):
    generator = torch.Generator().manual_seed(5 + edges)
    a = torch.randn((edges, width), generator=generator, dtype=torch.float64).to(device, dtype)
    weight = torch.randn((outputs, width), generator=generator, dtype=torch.float64).to(device, dtype)
    grad = torch.randn((edges, outputs), generator=generator, dtype=torch.float64).to(device, dtype)
    return a, weight, grad


def folded_with_gradient(function, a, weight, grad, **kwargs):
    a_ = a.clone().requires_grad_()
    out = function(a_, weight, **kwargs)
    out.backward(grad)
    return out.detach(), a_.grad


@pytest.mark.parametrize("shape", FOLDED_SHAPES)
def test_folded_linear_matches_linear_on_cpu(shape):
    a, weight, grad = folded_inputs(*shape)
    out, grad_a = folded_with_gradient(sl.folded_linear, a, weight, grad)
    torch.testing.assert_close(out, a @ weight.t(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_a, grad @ weight, rtol=1e-12, atol=1e-12)


def test_folded_linear_weight_with_gradient_falls_back_to_linear():
    a = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    sl.folded_linear(a, weight).sum().backward()
    assert weight.grad is not None


@pytest.mark.parametrize("contiguous_weight", [False, True])
@pytest.mark.parametrize("shape", FOLDED_SHAPES)
def test_folded_linear_equals_baseline(shape, contiguous_weight):
    base = baseline_module("skinny_linear")
    a, weight, grad = folded_inputs(*shape)
    new = folded_with_gradient(sl.folded_linear, a, weight, grad, contiguous_weight=contiguous_weight)
    old = folded_with_gradient(base.folded_linear, a, weight, grad, contiguous_weight=contiguous_weight)
    for n, o in zip(new, old):
        assert_equal(n, o)
    assert (sl.BLOCK_M, sl.BLOCK_K, sl.NUM_WARPS, sl.NUM_STAGES) == (
        base.BLOCK_M, base.BLOCK_K, base.NUM_WARPS, base.NUM_STAGES)


# --------------------------------------------------------------------------- fused_radial


def block_modules(width=64, seed=0):
    torch.manual_seed(seed)
    seq = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.SiLU(), torch.nn.Linear(width, width),
                              torch.nn.LayerNorm(width), torch.nn.SiLU())
    with torch.no_grad():
        for module in seq:
            for parameter in module.parameters():
                parameter.add_(0.3 * torch.randn_like(parameter))
                parameter.requires_grad_(False)
    return seq


def radial_inputs(device="cpu", dtype=torch.float64, rows=300, width=64):
    generator = torch.Generator().manual_seed(11)
    base = torch.randn((rows, 2 * width), generator=generator, dtype=torch.float64).to(device, dtype)
    grad = torch.randn((rows, width), generator=generator, dtype=torch.float64).to(device, dtype)
    return base[:, :width], grad  # strided view, as in the joint radial input


def value_and_input_gradient(function, h, grad):
    x = h.clone().requires_grad_()
    out = function(x)
    return out.detach(), torch.autograd.grad(out, x, grad)[0]


def test_fused_radial_cpu_matches_modules():
    seq = block_modules().double()
    block = fr.hidden_block(list(seq))
    assert block is not None
    h, grad = radial_inputs()
    new = value_and_input_gradient(lambda x: fr.fused_radial_hidden(x, *block), h, grad)
    old = value_and_input_gradient(seq, h, grad)
    for n, o in zip(new, old):
        torch.testing.assert_close(n, o, rtol=1e-12, atol=1e-12)


def test_fused_radial_other_layouts_are_not_fused():
    assert fr.hidden_block([torch.nn.LayerNorm(8), torch.nn.SiLU()]) is None
    assert fr.hidden_block([torch.nn.LayerNorm(8), torch.nn.SiLU(), torch.nn.Linear(8, 4),
                            torch.nn.LayerNorm(4), torch.nn.SiLU()]) is None


def mh1_weight_network(width=64):
    # MH-1's conv_tp_weights: LN, SiLU, Linear, LN, SiLU, Linear, LN, SiLU before its last layer.
    torch.manual_seed(1)
    seq = torch.nn.Sequential(*block_modules(width), torch.nn.Linear(width, width),
                              torch.nn.LayerNorm(width), torch.nn.SiLU()).double()
    for parameter in seq.parameters():
        parameter.requires_grad_(False)
    return seq


def test_fused_radial_mh1_weight_network_is_fused_inside_longer_sequence():
    # The exact-length matcher alone never fired on MH-1 (research engine, 14 October 2026).
    width = 64
    seq = mh1_weight_network(width)
    assert fr.hidden_block(list(seq)) is None
    plan = fr.hidden_plan(seq)
    assert [kind for kind, _ in plan] == ["fused", "module", "module", "module"]
    h = torch.randn((50, width), dtype=torch.float64)
    grad = torch.randn((50, width), dtype=torch.float64)
    new = value_and_input_gradient(lambda x: fr.run_hidden_plan(x, plan), h, grad)
    old = value_and_input_gradient(seq, h, grad)
    for n, o in zip(new, old):
        torch.testing.assert_close(n, o, rtol=1e-12, atol=1e-12)


def test_fused_radial_density_network_has_nothing_to_fuse():
    assert fr.hidden_plan([torch.nn.LayerNorm(8), torch.nn.SiLU()]) is None


def test_fused_radial_equals_baseline():
    base = baseline_module("fused_radial")
    seq = mh1_weight_network()
    new_plan, old_plan = fr.hidden_plan(seq), base.hidden_plan(seq)
    assert [(k, s) for k, s in new_plan] == [(k, s) for k, s in old_plan]
    h, grad = radial_inputs()
    new = value_and_input_gradient(lambda x: fr.run_hidden_plan(x, new_plan), h, grad)
    old = value_and_input_gradient(lambda x: base.run_hidden_plan(x, old_plan), h, grad)
    for n, o in zip(new, old):
        assert_equal(n, o)
    assert fr.BLOCK_ROWS == base.BLOCK_ROWS


# --------------------------------------------------------------------------- inplace_linear

# ir_mul segments (offset, mul, dim): 4x0e + 3x1o + 2x0e in, 2x0e + 5x1o + 2x2e out (2e gets no block).
SEGMENTS_IN = ((0, 4, 1), (4, 3, 3), (13, 2, 1))
SEGMENTS_OUT = ((0, 2, 1), (2, 5, 3), (17, 2, 5))
BLOCK_ENDS = ((0, 0), (1, 1), (2, 0))  # the third block adds into output segment 0 (baddbmm path)


def block_weights(dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(7)
    weights = [torch.randn((SEGMENTS_IN[i][1], SEGMENTS_OUT[j][1]), generator=generator, dtype=torch.float64)
               for i, j in BLOCK_ENDS]
    weights = [w.to(device, dtype) for w in weights]
    return weights, [w.t().contiguous() for w in weights]


def dense_block_linear(x, weights):
    """The same map as a sum over blocks on [n, dim, mul] views, for autograd."""
    n = x.shape[0]
    width_out = sum(m * d for _, m, d in SEGMENTS_OUT)
    pieces = [x.new_zeros((n, d, m)) for _, m, d in SEGMENTS_OUT]
    for (i, j), weight in zip(BLOCK_ENDS, weights):
        offset, mul, dim = SEGMENTS_IN[i]
        pieces[j] = pieces[j] + x[:, offset:offset + dim * mul].reshape(n, dim, mul) @ weight
    out = torch.cat([p.reshape(n, -1) for p in pieces], dim=1)
    assert out.shape[1] == width_out
    return out


def block_inputs(device="cpu", dtype=torch.float64, rows=41):
    generator = torch.Generator().manual_seed(9)
    x = torch.randn((rows, 15), generator=generator, dtype=torch.float64).to(device, dtype)
    grad = torch.randn((rows, 27), generator=generator, dtype=torch.float64).to(device, dtype)
    return x, grad


def test_inplace_block_linear_matches_dense_blocks():
    weights, weights_t = block_weights()
    x, grad = block_inputs()
    new = value_and_input_gradient(
        lambda v: il.block_linear(v, SEGMENTS_IN, SEGMENTS_OUT, BLOCK_ENDS, weights, weights_t), x, grad)
    old = value_and_input_gradient(lambda v: dense_block_linear(v, weights), x, grad)
    for n, o in zip(new, old):
        torch.testing.assert_close(n, o, rtol=1e-12, atol=1e-12)
    assert not new[0][:, 17:].any()  # the output segment without a block is zero


def test_inplace_block_linear_equals_baseline():
    base = baseline_module("inplace_linear")
    weights, weights_t = block_weights()
    x, grad = block_inputs()
    new = value_and_input_gradient(
        lambda v: il.block_linear(v, SEGMENTS_IN, SEGMENTS_OUT, BLOCK_ENDS, weights, weights_t), x, grad)
    old = value_and_input_gradient(
        lambda v: base.block_linear(v, SEGMENTS_IN, SEGMENTS_OUT, BLOCK_ENDS, weights, weights_t), x, grad)
    for n, o in zip(new, old):
        assert_equal(n, o)


# --------------------------------------------------------------------------- registration


def test_custom_ops_are_in_the_fastmace_namespace():
    for name in ("edge_outer_sum_forward", "edge_outer_sum_backward", "folded_linear_input_grad",
                 "fused_radial_hidden_forward", "fused_radial_hidden_backward", "inplace_block_linear"):
        assert hasattr(torch.ops.fastmace, name), name


def test_importing_the_kernels_does_not_import_triton():
    import subprocess

    code = ("import fastmace, sys\n"
            "import fastmace.kernels.triton.edge_outer_sum, fastmace.kernels.triton.skinny_linear\n"
            "import fastmace.kernels.triton.fused_radial, fastmace.kernels.triton.inplace_linear\n"
            "import fastmace.kernels.triton.cache\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('triton', 'langevin')))\n")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
