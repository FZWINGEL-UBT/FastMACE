"""Idea 31's species-skip selection (``fastmace.kernels.cutlass.skip_select``) against today's path, bitwise.

OMAT-0 medium's species skips on SrTiO3 (three species, 128 channels, 3xTF32): interaction 1's (l = 0..3 in and out,
the [atoms, 2048] input) and interaction 2's (0e only, read from the first 128 columns of a [atoms, 512] input).
Compiled as production compiles it, ``SpeciesSkip`` with ``skip_select`` off (today: the full products, the
own-species sum and the cat) against on (the selection in the epilogue). The output and the input's gradient must be
equal bit for bit. Runs where the CUTLASS kernels load (compute capability 8.0 and 9.0).
"""

import pytest
import torch

C = 128
SPECIES = 3


def _available():
    if not torch.cuda.is_available():
        return False
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    return tf32x3_gemm.backend("cuda") == "cutlass" and tf32x3.available("cuda")


pytestmark = pytest.mark.skipif(not _available(), reason="needs the CUTLASS 3xTF32 kernels (capability 8.0 or 9.0)")


@pytest.fixture
def deterministic():
    """Deterministic algorithms (Inductor's heuristic configs on both sides)."""
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def _skip(generator, degrees):
    from fastmace.rewrites.generic.modules import SpeciesSkip

    segments, blocks, offset = [], [], 0
    for i, degree in enumerate(degrees):
        segments.append((offset, C, C * (2 * degree + 1)))
        offset += C * (2 * degree + 1)
        blocks.append((i, i, torch.randn((SPECIES, C, C), generator=generator, device="cuda") / C ** 0.5))
    skip = SpeciesSkip(torch.arange(SPECIES, device="cuda"), segments, segments, blocks, "ir_mul")
    skip.tf32x3 = True
    skip.species_gemm_gpu = False
    return skip


@pytest.mark.parametrize(("n", "degrees", "width_in"), [(5000, (0, 1, 2, 3), 2048), (997, (0, 1, 2, 3), 2048),
                                                        (5000, (0,), 512)])
def test_skip_select_is_bitwise(n, degrees, width_in, deterministic):
    from fastmace.kernels.cutlass import skip_select

    generator = torch.Generator(device="cuda").manual_seed(n + width_in)
    skip = _skip(generator, degrees)
    x = torch.randn((n, width_in), generator=generator, device="cuda")
    x[:7] = 0.0  # zero rows
    attrs = torch.nn.functional.one_hot(torch.randint(SPECIES, (n,), generator=generator, device="cuda"),
                                        SPECIES).float()
    width_out = sum(C * (2 * degree + 1) for degree in degrees)
    grad = torch.randn((n, width_out), generator=generator, device="cuda")
    launches = []
    original = skip_select.tf32x3_skip_select

    def counted(*args):
        launches.append(1)
        return original(*args)

    results = {}
    for side in (False, True):
        skip.skip_select = side
        skip_select.tf32x3_skip_select = counted
        try:
            torch._dynamo.reset()
            compiled = torch.compile(lambda x1: skip(x1, attrs), fullgraph=True)
            x1 = x.clone().requires_grad_(True)
            out = compiled(x1)
            out.backward(grad)
            torch.cuda.synchronize()
        finally:
            skip_select.tf32x3_skip_select = original
        results[side] = (out.detach(), x1.grad)
    assert launches, "skip_select never ran"
    (out_a, grad_a), (out_b, grad_b) = results[False], results[True]
    assert torch.equal(out_a.view(torch.int32), out_b.view(torch.int32))
    assert torch.equal(grad_a.view(torch.int32), grad_b.view(torch.int32))


def test_select_declines_without_switch():
    generator = torch.Generator(device="cuda").manual_seed(1)
    skip = _skip(generator, (0,))
    assert not skip.skip_select
