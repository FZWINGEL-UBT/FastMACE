"""fastmace.kernels.cutlass.tf32x3 tiles on CUDA: every ``TILE`` (64x64, 32x64, 64x32) within 1e-6 of the row scale
|A| |B| of float64 and bitwise the 64x64 tile's (the same K loop per element), both B layouts, atom-GEMM shapes and
ragged edges. The sm_80 cubin loads on any compute capability 8.x (RTX 4000 Ada, 4 November 2026: largest error
2.5e-7); production admits 8.0 alone (``available``). The sm_90 cubin (same source) runs on 9.0 (H100, H200).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (8, 9),
                                reason="needs a compute capability 8.x or 9.0 GPU")


@pytest.mark.parametrize("m, k, n", [(5000, 128, 128), (5000, 1024, 128), (4999, 96, 132), (37, 64, 20)])
def test_tiles_agree(monkeypatch, m, k, n):
    from fastmace.kernels.cutlass import tf32x3

    generator = torch.Generator(device="cuda").manual_seed(m + k + n)
    a = torch.randn(m, k, device="cuda", generator=generator)
    b = torch.randn(k, n, device="cuda", generator=generator)
    exact = a.double() @ b.double()
    scale = a.double().abs() @ b.double().abs()
    for operand in (b, b.t().contiguous().t()):
        outs = {}
        for tile in ("64x64", "32x64", "64x32"):
            monkeypatch.setattr(tf32x3, "TILE", tile)
            out = torch.full((m, n), float("nan"), device="cuda")
            tf32x3.mm_into(a, operand, out)
            assert float(((out.double() - exact).abs() / scale).max()) < 1e-6, tile
            outs[tile] = out
        assert torch.equal(outs["32x64"], outs["64x64"]) and torch.equal(outs["64x32"], outs["64x64"])
