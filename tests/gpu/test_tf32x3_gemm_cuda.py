"""The 3xTF32 GEMM of ``tf32x3_gemms`` on CUDA (plans/opt-4.md, OPT4-T03 stage 1).

On Gaussian operands its error against float64 is FP32-class (within 4x cuBLAS IEEE's, OPT-T01 measured it below);
it repeats bitwise, replays bitwise in a CUDA graph, and its input gradients equal the op applied to the
transposes. ``segment_gemm``'s joined products with ``precision="tf32x3"`` agree with the dense op row by row.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                                reason="needs CUDA with compute capability >= 8.0")

SHAPES = [(216, 512, 512), (1944, 128, 2048), (8 * 216, 10, 64), (7, 33, 5)]


def _operands(m, k, n, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((m, k), generator=generator, device="cuda"),
            torch.randn((k, n), generator=generator, device="cuda"))


@pytest.mark.parametrize("shape", SHAPES)
def test_error_is_fp32_class_and_repeats_bitwise(shape):
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    a, b = _operands(*shape)
    reference = a.double() @ b.double()
    got = tf32x3_mm(a, b)
    ieee = a @ b
    error = float((got.double() - reference).abs().max())
    assert error <= 4 * float((ieee.double() - reference).abs().max()) + 1e-30
    assert torch.equal(tf32x3_mm(a, b), got)
    assert torch.equal(tf32x3_mm(a.t().contiguous().t(), b), got)  # strided A, same arithmetic


def test_graph_replay_is_bitwise():
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    a, b = _operands(512, 256, 384)
    eager = tf32x3_mm(a, b)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tf32x3_mm(a, b)  # warm-up outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = tf32x3_mm(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_gradients_are_the_op_on_the_transposes():
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul, tf32x3_mm

    a, b = _operands(300, 64, 96, seed=1)
    a.requires_grad_()
    b.requires_grad_()
    grad = torch.randn((300, 96), device="cuda")
    grad_a, grad_b = torch.autograd.grad(tf32x3_matmul(a, b), (a, b), grad)
    assert torch.equal(grad_a, tf32x3_mm(grad, b.detach().t()))
    assert torch.equal(grad_b, tf32x3_mm(a.detach().t(), grad))


def test_joined_products_in_tf32x3_equal_the_dense_op():
    from fastmace.kernels.triton.segment_gemm import joined_products
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    atoms, dims, muls = 50, (1, 3), (64, 32)
    generator = torch.Generator(device="cuda").manual_seed(2)
    rows = [torch.randn((atoms * d, 48), generator=generator, device="cuda") for d in dims]
    weights = [torch.randn((48, m), generator=generator, device="cuda") for m in muls]
    offsets = (0, muls[0])
    width = sum(d * m for d, m in zip(dims, muls))
    out = joined_products(rows, weights, [w.t().contiguous() for w in weights], offsets, dims, width, "tf32x3")
    for x, w, offset, d, m in zip(rows, weights, offsets, dims, muls):
        assert torch.equal(out[:, offset:offset + d * m].reshape(atoms * d, m), tf32x3_mm(x, w))


def test_subtile_products_use_simt_ieee():
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm, uses_tf32x3_tile

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    assert not uses_tf32x3_tile(6, 96)
    assert not uses_tf32x3_tile(512, 6)
    assert uses_tf32x3_tile(32, 32)
    for shape in [(216, 6, 96), (216, 512, 6), (216, 1024, 1)]:
        a, b = _operands(*shape)
        got = tf32x3_mm(a, b)
        assert torch.equal(got, a @ b)
        assert torch.equal(tf32x3_mm(a.t().contiguous().t(), b), got)
