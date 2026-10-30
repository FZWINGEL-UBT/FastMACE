"""C1 side finding: does segment_gemm re-read A from DRAM because of its tile order?

``fastmace.kernels.triton.segment_gemm`` launches a (row tiles, column tiles) grid. Program ids run row tiles
fastest, so the column tiles of one row tile run a whole pass apart and A ([rows, K], larger than L2) is read once
per column tile. This probe times the kernel as built against the same kernel with the column tiles of a row tile
adjacent (one-dimensional grid, column index fastest), on MH-1's ``linear_2`` shapes at NbCl5 4x4x8 (9,216 atoms):
l = 0..3, rows = atoms * (2l + 1), K = N = 512, float32 IEEE. Same tiles and the same K order, so the outputs must be
bitwise equal. Alternating blocks in one process; run inside a local GPU reservation.
"""

import sys

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import torch
import triton
import triton.language as tl

from fastmace.kernels.triton import segment_gemm as sg
from fastmace.kernels.triton import tile_rule


@triton.jit
def gemm_column_fastest(a_ptr, b_ptr, c_ptr, rows, N, K, D, a_n, a_m, b_k, b_n, c_n, c_m,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BLOCK_N)
    pm, pn = pid // tiles_n, pid % tiles_n
    r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    atom, comp = r // D, r % D
    a_row = atom * a_n + comp * a_m
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + kk
        a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
        b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision=PRECISION)
    tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc, mask=(r < rows)[:, None] & (j < N)[None, :])


def run(order, a, b, c, rows, dim, choice):
    k, n = b.shape
    args = (a, b, c, rows, n, k, dim, k * dim, k, b.stride(0), b.stride(1), n * dim, n)
    meta = dict(BLOCK_M=choice.block_m, BLOCK_N=choice.block_n, BLOCK_K=choice.block_k, PRECISION="ieee",
                num_warps=choice.num_warps, num_stages=choice.num_stages)
    if order == "built":
        sg._kernel()[(triton.cdiv(rows, choice.block_m), triton.cdiv(n, choice.block_n))](*args, **meta)
    else:
        gemm_column_fastest[(triton.cdiv(rows, choice.block_m) * triton.cdiv(n, choice.block_n),)](*args, **meta)


def main():
    atoms = int(sys.argv[1]) if len(sys.argv) > 1 else 9216
    reps, blocks = 20, 6
    sms = tile_rule.device_sms(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(1)
    total = {"built": 0.0, "column_fastest": 0.0}
    for ell in range(4):
        dim = 2 * ell + 1
        rows = atoms * dim
        a = torch.randn((atoms, dim, 512), device="cuda", generator=g)
        b = torch.randn((512, 512), device="cuda", generator=g) / 23
        choice = tile_rule.segment_gemm_choice(rows, 512, 512, sms)
        outs = {}
        for order in total:
            outs[order] = torch.empty((atoms, dim, 512), device="cuda")
            run(order, a, b, outs[order], rows, dim, choice)
        torch.cuda.synchronize()
        equal = torch.equal(outs["built"], outs["column_fastest"])
        times = {order: [] for order in total}
        for block in range(blocks):
            for order in (("built", "column_fastest") if block % 2 == 0 else ("column_fastest", "built")):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    run(order, a, b, outs[order], rows, dim, choice)
                stop.record()
                torch.cuda.synchronize()
                times[order].append(start.elapsed_time(stop) / reps * 1e3)
        med = {order: sorted(v)[len(v) // 2] for order, v in times.items()}
        for order in total:
            total[order] += med[order]
        print(f"l={ell} rows={rows} {choice.backend} block_m={choice.block_m} block_n={choice.block_n} "
              f"built {med['built']:.1f} us, column-fastest {med['column_fastest']:.1f} us, "
              f"ratio {med['column_fastest'] / med['built']:.3f}, bitwise equal {equal}")
    print(f"sum built {total['built']:.1f} us, column-fastest {total['column_fastest']:.1f} us, "
          f"ratio {total['column_fastest'] / total['built']:.3f}")


if __name__ == "__main__":
    main()
