"""Idea 31's first-interaction epilogue (``fastmace.kernels.cutlass.first_div``) against today's path, bitwise.

OMAT-0 medium's first density interaction after the generic fold: four reassociated products (l = 0..3, rows
atoms * (2l+1), K = 3 slots x 64, 128 channels) joined, divided by (density + 1) and fed to the species skip (three
species, 3xTF32). Compiled as production compiles it, today's ``skip(cat_l(left_l @ right_l) / (d + 1))`` against
``skip(blocks_div(lefts, rights, d))`` (the blocks divided in the epilogue, read in place by the skip). The
output, every left factor's gradient and the density's gradient must be equal bit for bit (deterministic algorithms, so
that both sides' density reductions run one config rather than each its benchmarked one). Runs where the CUTLASS
kernels load (compute capability 8.0 and 9.0).
"""

import pytest
import torch

LS = (0, 1, 2, 3)
K = 192
C = 128
SPECIES = 3


def _available():
    if not torch.cuda.is_available():
        return False
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    return tf32x3_gemm.backend("cuda") == "cutlass" and tf32x3.available("cuda")


pytestmark = pytest.mark.skipif(not _available(), reason="needs the CUTLASS 3xTF32 kernels (capability 8.0 or 9.0)")


def _skip(generator):
    from fastmace.rewrites.generic.modules import SpeciesSkip

    segments, blocks, offset = [], [], 0
    for i, degree in enumerate(LS):
        segments.append((offset, C, C * (2 * degree + 1)))
        offset += C * (2 * degree + 1)
        blocks.append((i, i, torch.randn((SPECIES, C, C), generator=generator, device="cuda") / C ** 0.5))
    skip = SpeciesSkip(torch.arange(SPECIES, device="cuda"), segments, segments, blocks, "ir_mul")
    skip.tf32x3 = True
    skip.species_gemm_gpu = False
    return skip


@pytest.fixture
def deterministic():
    """Deterministic algorithms for the whole test, which Inductor reads at every compile (its own ``deterministic``
    flag does not survive the first compile in torch 2.14): each reduction runs its heuristic config instead of the
    benchmarked one, so that both sides' density reductions run the same config."""
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(before[0], warn_only=before[1])


@pytest.mark.parametrize("n", [5000, 997])
def test_blocks_div_is_bitwise(n, deterministic):
    from fastmace.kernels.cutlass.first_div import applies, blocks_div
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

    generator = torch.Generator(device="cuda").manual_seed(n)
    skip = _skip(generator)
    lefts = [torch.randn((n * (2 * l + 1), K), generator=generator, device="cuda") for l in LS]  # noqa: E741
    lefts[1][:9] = 0.0  # zero rows: the epilogue's 0 / (d + 1)
    rights = [torch.randn((K, C), generator=generator, device="cuda") / K ** 0.5 for _ in LS]
    density = torch.rand((n, 1), generator=generator, device="cuda") * 4.0
    attrs = torch.nn.functional.one_hot(torch.randint(SPECIES, (n,), generator=generator, device="cuda"),
                                        SPECIES).float()
    grad = torch.randn((n, sum(C * (2 * l + 1) for l in LS)), generator=generator, device="cuda")  # noqa: E741
    assert applies(lefts, rights, density)

    def today(xs, density):
        blocks = [tf32x3_matmul(x, r).view(n, 2 * d + 1, C).reshape(n, -1) for x, r, d in zip(xs, rights, LS)]
        return skip(torch.cat(blocks, dim=1) / (density + 1), attrs)

    def epilogue(xs, d):
        outs = blocks_div(xs, rights, d)
        return skip(tuple(o.view(n, 2 * degree + 1, C) for o, degree in zip(outs, LS)), attrs)

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
    bad = (density_a.view(torch.int32) != density_b.view(torch.int32)).nonzero()[:, 0]
    assert bad.numel() == 0, (bad.tolist()[:20], bad.numel(), (density_a - density_b)[bad][:5].tolist(),
                              density_a[bad][:5].tolist())
