"""fastmace.kernels.triton.gated_products on CUDA (plans/opt-10.md, OPT10-T01, ``gate_epilogue``).

The joined products over gated rows with the gate's backward in the input-gradient GEMM, against the chain
(``joined_products`` over the same rows built by autograd), float32: with enough atoms that the shape rule picks
Triton for every product (the fused kernel) at the SPEC §6.4 threshold, and on a few atoms (cuBLAS, the stock
operations) to rounding; two launches agree bitwise (no atomics: deterministic).
"""

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import tile_rule
from fastmace.kernels.triton.gated_products import _gated_rows, gated_joined_products
from fastmace.kernels.triton.segment_gemm import joined_products

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)

MUL, DIMS = 64, (1, 3, 5, 7)


def _case(atoms, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(shape, generator=g, device="cuda")

    weights = [r(MUL, MUL) / MUL ** 0.5 for _ in DIMS]
    offsets, at = [], 0
    for d in DIMS:
        offsets.append(at)
        at += d * MUL
    return dict(scalars=r(atoms, MUL).requires_grad_(), gated=[r(atoms * d, MUL).requires_grad_() for d in DIMS[1:]],
                gates=torch.sigmoid(r(atoms, MUL * 3)).requires_grad_(), gate_offsets=[0, MUL, 2 * MUL],
                weights=weights, weights_t=[w.t().contiguous() for w in weights], offsets=offsets, dims=list(DIMS),
                inputs=[0, 1, 2, 3], width=at)


def _both(case):
    leaves = (case["scalars"], case["gates"], *case["gated"])
    atoms = case["scalars"].shape[0]
    grad = torch.randn((atoms, case["width"]), generator=torch.Generator(device="cuda").manual_seed(9), device="cuda")
    rows = _gated_rows(case["scalars"], case["gated"], case["gates"], case["gate_offsets"])
    stock = joined_products([rows[i] for i in case["inputs"]], case["weights"], case["weights_t"], case["offsets"],
                            case["dims"], case["width"])
    folded = gated_joined_products(**case)
    return ([stock.detach(), *torch.autograd.grad(stock, leaves, grad)],
            [folded.detach(), *torch.autograd.grad(folded, leaves, grad)])


def test_fused_kernel_matches_the_chain_and_is_deterministic():
    sms = tile_rule.device_sms("cuda")
    atoms = -(-tile_rule.MIN_TRITON_WAVES * sms * tile_rule.BLOCK_M // min(DIMS)) + 5  # a partial atom tile
    assert all(tile_rule.segment_gemm_choice(atoms * d, MUL, MUL, sms).backend == "triton" for d in DIMS)
    case = _case(atoms)
    want, got = _both(case)
    assert torch.equal(got[0], want[0])  # the forward is segment_gemm's
    for a, b in zip(got[1:], want[1:]):
        assert float((a - b).abs().max() / b.abs().max()) <= 1e-5
    _, again = _both(case)
    assert all(torch.equal(a, b) for a, b in zip(again, got))


def test_small_cells_keep_cublas_and_the_stock_operations():
    case = _case(8, seed=1)
    want, got = _both(case)
    assert torch.equal(got[0], want[0])
    for a, b in zip(got[1:], want[1:]):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_compiled_matches_eager():
    """Dynamo traces the autograd function and the custom op (the graph tier compiles the model)."""
    sms = tile_rule.device_sms("cuda")
    case = _case(-(-tile_rule.MIN_TRITON_WAVES * sms * tile_rule.BLOCK_M // 3), seed=2)
    leaves = (case["scalars"], case["gates"], *case["gated"])

    def energy(scalars, gates, *gated):
        out = gated_joined_products(scalars, list(gated), gates, case["gate_offsets"], case["weights"],
                                    case["weights_t"], case["offsets"], case["dims"], case["inputs"], case["width"])
        return (torch.tanh(out) ** 2).sum()

    eager = torch.autograd.grad(energy(*leaves), leaves)
    compiled = torch.autograd.grad(torch.compile(energy, dynamic=False)(*leaves), leaves)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
