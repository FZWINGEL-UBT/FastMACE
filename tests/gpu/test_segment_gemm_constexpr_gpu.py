"""segment_gemm's irrep size D as ``tl.constexpr`` (plans/capacity-ideas-20261030.md, fourth list): exact.

The joined products and their input gradients for D = 1, 3, 5, 7, with the shape rule forced to Triton, are
bitwise equal to the same kernel with D a runtime argument (``main``'s kernel before the change, kept here as
the reference).
"""

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import segment_gemm as sg
from fastmace.kernels.triton import tile_rule

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)

DIMS, MUL = (1, 3, 5, 7), 64


def _runtime_d_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, rows, N, K, D, a_n, a_m, b_k, b_n, c_n, c_m,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        atom, comp = r // D, r % D
        a_row = atom * a_n + comp * a_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
            b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc,
                 mask=(r < rows)[:, None] & (j < N)[None, :])

    return gemm


def _products(atoms):
    g = torch.Generator(device="cuda").manual_seed(3)
    rows = [torch.randn((atoms * d, MUL), generator=g, device="cuda") for d in DIMS]
    weights = [torch.randn((MUL, MUL), generator=g, device="cuda") / MUL ** 0.5 for _ in DIMS]
    offsets, offset = [], 0
    for d in DIMS:
        offsets.append(offset)
        offset += d * MUL
    grad = torch.randn((atoms, offset), generator=g, device="cuda")
    out = sg.joined_products_op(rows, weights, offsets, list(DIMS), offset, "ieee")
    grads = sg.joined_products_input_grads(grad, [w.t().contiguous() for w in weights], offsets, list(DIMS), "ieee")
    return out, grads


@pytest.mark.parametrize("extra_atoms", [0, 37])  # a full last tile and a ragged one
def test_constexpr_d_is_bitwise_equal_to_runtime_d(monkeypatch, extra_atoms):
    sms = tile_rule.device_sms("cuda")
    atoms = -(-tile_rule.MIN_TRITON_WAVES * sms * tile_rule.BLOCK_M // min(DIMS)) + extra_atoms
    assert all(tile_rule.segment_gemm_choice(atoms * d, MUL, MUL, sms).backend == "triton" for d in DIMS)
    out, grads = _products(atoms)
    reference = _runtime_d_kernel()
    monkeypatch.setattr(sg, "_kernel", lambda: reference)
    ref_out, ref_grads = _products(atoms)
    assert torch.equal(out, ref_out)
    assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))


def test_kernel_takes_d_as_constexpr():
    kernel = sg._kernel()
    assert kernel.params[kernel.arg_names.index("D")].is_constexpr
