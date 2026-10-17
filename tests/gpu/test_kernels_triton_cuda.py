"""fastmace.kernels.triton on CUDA: the Triton kernels against PyTorch references and the baseline kernels.

Ported from the research engine's tests/core/test_edge_outer_sum.py, test_skinny_linear.py and
test_fused_radial.py (GPU parts). The baseline comparisons load the research engine's modules from
the export named by FASTMACE_BASELINE (ResearchModule tag research-engine-baseline-20261017) and
skip when it is unset. Both sides launch the same kernel source with the same configuration, so
kernels without atomics must agree bitwise; the segmented forward adds run sums atomically, so its
summation order varies between launches and is compared at FP32 rounding.
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import edge_outer_sum as eos
from fastmace.kernels.triton import fused_radial as fr
from fastmace.kernels.triton import inplace_linear as il
from fastmace.kernels.triton import skinny_linear as sl

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)


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


@pytest.fixture(autouse=True)
def _no_tf32():
    """The kernels use IEEE FP32 dots; keep cuBLAS references in FP32 too."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


# --------------------------------------------------------------------------- edge_outer_sum


def reference_outer_sum(a, y, key, slots):
    outer = (y[:, :, None] * a[:, None, :]).reshape(a.shape[0], -1)
    return torch.zeros((slots, outer.shape[1]), dtype=a.dtype, device=a.device).index_add(0, key, outer)


def outer_sum_inputs(device, dtype, edges=300, k=65, d=16, slots=37):
    generator = torch.Generator().manual_seed(3)
    a = torch.randn((edges, k), generator=generator, dtype=torch.float64)
    y = torch.randn((edges, d), generator=generator, dtype=torch.float64)
    key = torch.randint(0, slots, (edges,), generator=generator)
    grad = torch.randn((slots, d * k), generator=generator, dtype=torch.float64)
    return [t.to(device, dtype if t.is_floating_point() else None) for t in (a, y, key, grad)] + [slots]


def test_triton_segmented_forward_matches_reference():
    a, y, key, grad, slots = outer_sum_inputs("cuda", torch.float32)
    for keys in (key, torch.sort(key).values):
        torch.testing.assert_close(eos.edge_outer_sum(a, y, keys, slots, segmented=True).cpu(),
                                   reference_outer_sum(a.cpu(), y.cpu(), keys.cpu(), slots), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 1e-4), (torch.float64, 1e-12)])
def test_triton_backward_matches_reference(dtype, tolerance):
    a, y, key, grad, slots = outer_sum_inputs("cuda", dtype)
    grad_a, grad_y = eos.edge_outer_sum_backward(grad, a, y, key)
    ref_a, ref_y = eos.edge_outer_sum_backward(grad.cpu(), a.cpu(), y.cpu(), key.cpu())
    torch.testing.assert_close(grad_a.cpu(), ref_a, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(grad_y.cpu(), ref_y, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triton_backward_equals_baseline(dtype):
    base = baseline_module("edge_outer_sum")
    a, y, key, grad, slots = outer_sum_inputs("cuda", dtype, edges=5000, slots=800)
    for n, o in zip(eos.edge_outer_sum_backward(grad, a, y, key), base.edge_outer_sum_backward(grad, a, y, key)):
        assert_equal(n, o)


def test_triton_segmented_forward_equals_baseline():
    base = baseline_module("edge_outer_sum")
    a, y, key, grad, slots = outer_sum_inputs("cuda", torch.float32, edges=5000, slots=800)
    for keys in (key, torch.sort(key).values):
        new = eos.edge_outer_sum(a, y, keys, slots, segmented=True)
        old = base.edge_outer_sum(a, y, keys, slots, segmented=True)
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-5)


def test_triton_unsegmented_path_equals_baseline():
    base = baseline_module("edge_outer_sum")
    a, y, key, grad, slots = outer_sum_inputs("cuda", torch.float32, edges=5000, slots=800)
    key = torch.sort(key).values  # index_add on sorted keys: same order on both sides
    results = []
    for function in (eos.edge_outer_sum, base.edge_outer_sum):
        a_, y_ = a.clone().requires_grad_(), y.clone().requires_grad_()
        out = function(a_, y_, key, slots)
        out.backward(grad)
        results.append((a_.grad, y_.grad))
    for n, o in zip(*results):
        assert_equal(n, o)


# --------------------------------------------------------------------------- folded_linear


def test_triton_input_gradient_matches_reference():
    generator = torch.Generator().manual_seed(5)
    for edges, outputs, width in ((300, 96, 65), (77, 40, 33), (5, 17, 9)):
        a = torch.randn((edges, width), generator=generator).cuda()
        weight = torch.randn((outputs, width), generator=generator).cuda()
        grad = torch.randn((edges, outputs), generator=generator).cuda()
        a_ = a.clone().requires_grad_()
        out = sl.folded_linear(a_, weight)
        out.backward(grad)
        torch.testing.assert_close(out, a @ weight.t(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(a_.grad, grad @ weight, rtol=1e-4, atol=1e-4)
    grad = torch.randn(1000, 1280, device="cuda")
    weight = torch.randn(1280, 65, device="cuda")
    torch.testing.assert_close(sl.folded_linear_input_grad(grad, weight).double(),
                               grad.double() @ weight.double(), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("rows, q, width", [(4000, 1280, 65), (777, 96, 33), (5, 17, 9)])
def test_triton_input_gradient_equals_baseline(rows, q, width):
    base = baseline_module("skinny_linear")
    generator = torch.Generator(device="cuda").manual_seed(13)
    grad = torch.randn((rows, q), device="cuda", generator=generator)
    weight = torch.randn((q, width), device="cuda", generator=generator)
    assert_equal(sl.folded_linear_input_grad(grad, weight), base.folded_linear_input_grad(grad, weight))


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


def value_and_input_gradient(function, h, grad):
    x = h.clone().requires_grad_()
    out = function(x)
    return out.detach(), torch.autograd.grad(out, x, grad)[0]


def radial_inputs(rows=300, width=64):
    generator = torch.Generator(device="cuda").manual_seed(11)
    base = torch.randn((rows, 2 * width), device="cuda", generator=generator)
    grad = torch.randn((rows, width), device="cuda", generator=generator)
    return base[:, :width], grad  # strided view, as in the joint radial input


def test_triton_fused_radial_matches_modules():
    seq = block_modules().cuda()
    block = fr.hidden_block(list(seq))
    h, grad = radial_inputs()
    new = value_and_input_gradient(lambda x: fr.fused_radial_hidden(x, *block), h, grad)
    old = value_and_input_gradient(seq, h, grad)
    for n, o in zip(new, old):
        torch.testing.assert_close(n, o, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("width", [16, 64, 128])
def test_triton_fused_radial_equals_baseline(width):
    base = baseline_module("fused_radial")
    seq = block_modules(width).cuda()
    h, grad = radial_inputs(rows=5000, width=width)
    new = value_and_input_gradient(lambda x: fr.fused_radial_hidden(x, *fr.hidden_block(list(seq))), h, grad)
    old = value_and_input_gradient(lambda x: base.fused_radial_hidden(x, *base.hidden_block(list(seq))), h, grad)
    for n, o in zip(new, old):
        assert_equal(n, o)


# --------------------------------------------------------------------------- inplace_linear


def test_inplace_block_linear_equals_baseline_on_cuda():
    base = baseline_module("inplace_linear")
    segments_in, segments_out = ((0, 128, 1), (128, 128, 3)), ((0, 128, 1), (128, 128, 3))
    ends = ((0, 0), (1, 1))
    generator = torch.Generator(device="cuda").manual_seed(17)
    weights = [torch.randn((128, 128), device="cuda", generator=generator) for _ in ends]
    weights_t = [w.t().contiguous() for w in weights]
    x = torch.randn((3000, 512), device="cuda", generator=generator)
    grad = torch.randn((3000, 512), device="cuda", generator=generator)
    new = value_and_input_gradient(
        lambda v: il.block_linear(v, segments_in, segments_out, ends, weights, weights_t), x, grad)
    old = value_and_input_gradient(
        lambda v: base.block_linear(v, segments_in, segments_out, ends, weights, weights_t), x, grad)
    for n, o in zip(new, old):
        assert_equal(n, o)


# --------------------------------------------------------------------------- CUDA graphs


def test_kernels_replay_in_a_cuda_graph():
    """The custom ops launch on the current stream, so a captured graph replays them."""
    a, y, key, grad, slots = outer_sum_inputs("cuda", torch.float32, edges=2000, slots=300)
    key = torch.sort(key).values
    weight = torch.randn((1280, 65), device="cuda")
    big_grad = torch.randn((2000, 1280), device="cuda")
    eager = (eos.edge_outer_sum(a, y, key, slots, segmented=True), *eos.edge_outer_sum_backward(grad, a, y, key),
             sl.folded_linear_input_grad(big_grad, weight))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside the capture (Triton compiles here)
        eos.edge_outer_sum(a, y, key, slots, segmented=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = (eos.edge_outer_sum(a, y, key, slots, segmented=True),
                    *eos.edge_outer_sum_backward(grad, a, y, key), sl.folded_linear_input_grad(big_grad, weight))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured[0], eager[0], rtol=1e-6, atol=1e-5)
    for n, o in zip(captured[1:], eager[1:]):
        assert_equal(n, o)
