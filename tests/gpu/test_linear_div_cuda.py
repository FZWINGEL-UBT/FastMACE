"""Idea 31's epilogue GEMM (``fastmace.kernels.cutlass.linear_div``) against today's path, bitwise.

A grouped ``BlockLinear`` with OMAT-0 medium's interaction-2 ``linear`` shapes (blocks l = 0..3, K = 256, 384, 384,
256 to 128 channels, the left factors as ``wprime_backward.BLOCK_MESSAGE`` hands them over), compiled as production
compiles it: ``linear(x) / (density + 1)`` (``mm_into`` per block, Inductor's cat and division kernels) against
``linear.divided(x, density)`` (the remapped, divided store in the GEMM epilogue). The output, every left factor's
gradient and the density's gradient must be equal bit for bit, with zero rows and a ragged atom count. Runs where the
CUTLASS kernels load (compute capability 8.0 and 9.0).
"""

import pytest
import torch

DIMS = (1, 3, 5, 7)
KS = (256, 384, 384, 256)
MUL = 128


def _available():
    if not torch.cuda.is_available():
        return False
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    return tf32x3_gemm.backend("cuda") == "cutlass" and tf32x3.available("cuda")


pytestmark = pytest.mark.skipif(not _available(), reason="needs the CUTLASS 3xTF32 kernels (capability 8.0 or 9.0)")


def _linear(generator):
    from fastmace.rewrites.linear.modules import BlockLinear

    segments_in, segments_out, blocks, offset_in, offset_out = [], [], [], 0, 0
    for j, (dim, k) in enumerate(zip(DIMS, KS)):
        segments_in.append((offset_in, k, dim))
        segments_out.append((offset_out, MUL, dim))
        offset_in += k * dim
        offset_out += MUL * dim
        blocks.append((j, j, torch.randn((k, MUL), generator=generator, device="cuda") / k ** 0.5))
    linear = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), segments_in, segments_out, blocks, grouped=True)
    linear.tf32x3 = True
    linear.linear_div = True
    return linear


@pytest.mark.parametrize("n", [5000, 997])
def test_divided_is_bitwise(n):
    generator = torch.Generator(device="cuda").manual_seed(n)
    linear = _linear(generator)
    lefts = [torch.randn((n * dim, k), generator=generator, device="cuda") for dim, k in zip(DIMS, KS)]
    lefts[2][: 5 * 7] = 0.0  # whole zero rows: the epilogue's 0 / (d + 1)
    density = torch.rand((n, 1), generator=generator, device="cuda") * 4.0
    grad = torch.randn((n, sum(MUL * d for d in DIMS)), generator=generator, device="cuda")

    def today(xs, d):
        return linear(tuple(xs)) / (d + 1)

    def epilogue(xs, d):
        return linear.divided(tuple(xs), d)

    results = {}
    for name, fn in (("today", today), ("epilogue", epilogue)):
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        xs = [x.clone().requires_grad_(True) for x in lefts]
        d = density.clone().requires_grad_(True)
        out = compiled(xs, d)
        out.backward(grad)
        torch.cuda.synchronize()
        results[name] = (out.detach(), [x.grad for x in xs], d.grad)
    out_a, grads_a, density_a = results["today"]
    out_b, grads_b, density_b = results["epilogue"]
    assert torch.equal(out_a.view(torch.int32), out_b.view(torch.int32))
    for a, b in zip(grads_a, grads_b):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert torch.equal(density_a.view(torch.int32), density_b.view(torch.int32))


def test_divided_declines_without_switch():
    generator = torch.Generator(device="cuda").manual_seed(1)
    linear = _linear(generator)
    linear.linear_div = False
    lefts = tuple(torch.randn((10 * dim, k), generator=generator, device="cuda") for dim, k in zip(DIMS, KS))
    assert linear.divided(lefts, torch.rand((10, 1), device="cuda")) is None
