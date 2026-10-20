"""fastmace.kernels.triton.segment_gemm on CUDA (plan opt-1, OPT-T02 stage B): the gathered-row Triton GEMM.

The joined products and their input gradients, with the shape rule forced to Triton by enough atoms, against
cuBLAS products and the block linears' join, float32 at the SPEC §6.4 threshold; and two launches agree
bitwise (no atomics: deterministic).
"""

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import segment_gemm as sg
from fastmace.kernels.triton import tile_rule

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)

DIMS, MUL = (1, 3, 5, 7), 64


def _case(atoms, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = [torch.randn((atoms * d, MUL), generator=g, device="cuda").requires_grad_() for d in DIMS]
    weights = [torch.randn((MUL, MUL), generator=g, device="cuda") / MUL ** 0.5 for _ in DIMS]
    offsets, offset = [], 0
    for d in DIMS:
        offsets.append(offset)
        offset += d * MUL
    return rows, weights, [w.t().contiguous() for w in weights], offsets, offset


def test_triton_path_matches_cublas_and_join():
    sms = tile_rule.device_sms("cuda")
    atoms = -(-tile_rule.MIN_TRITON_WAVES * sms * tile_rule.BLOCK_M // min(DIMS))
    assert all(tile_rule.segment_gemm_choice(atoms * d, MUL, MUL, sms).backend == "triton" for d in DIMS)
    rows, weights, weights_t, offsets, width = _case(atoms)
    grad = torch.randn((atoms, width), device="cuda")
    joined = sg.joined_products(rows, weights, weights_t, offsets, list(DIMS), width)
    reference = torch.cat([(x @ w).reshape(atoms, -1) for x, w in zip(rows, weights)], dim=1)
    assert float((joined - reference).abs().max() / reference.abs().max()) <= 1e-4
    got = torch.autograd.grad(joined, rows, grad)
    want = torch.autograd.grad(reference, rows, grad)
    for a, b in zip(got, want):
        assert float((a - b).abs().max() / b.abs().max()) <= 1e-4
    again = sg.joined_products(rows, weights, weights_t, offsets, list(DIMS), width)
    assert torch.equal(again, joined)


def test_small_cells_keep_cublas_and_equal_the_join_bitwise():
    rows, weights, weights_t, offsets, width = _case(8, seed=1)
    joined = sg.joined_products(rows, weights, weights_t, offsets, list(DIMS), width)
    reference = torch.cat([(x @ w).reshape(8, -1) for x, w in zip(rows, weights)], dim=1)
    assert torch.equal(joined, reference)
