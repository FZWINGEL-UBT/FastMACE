"""OPT4-T01 diagnostic probe: what limits the write-bound radial forward kernel (store, FMA dot or K step).

    python tools/radial_forward_probe.py [--out results/<run-id>/radial_forward_probe.json]

Variants at the production layout (a [E, 65], weight [1280, 65]), CUDA-graph replays, median of 7 blocks
of 50, against cuBLAS on the padded [E, 68] operand:
- ``store``: the kernel's grid and tiles storing zeros (its write rate alone);
- ``dot<BK>``: the kernel with a K step of BK (16, 32, 64);
- ``tma``: a persistent kernel over all tiles, the output written through a TMA tensor descriptor, so that
  each tile's store can overlap the next tile's product.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastmace  # noqa: F401,E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from radial_forward_bench import EDGES, P, Q, timed  # noqa: E402


@triton.jit
def _kernel(a_ptr, w_ptr, out_ptr, M, N, K, sam, sak, swn, swk, som, K_MAIN: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, STORE_ONLY: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    m = (pid // n_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = (pid % n_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask, col_mask = m < M, n < N
    m = m.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if not STORE_ONLY:
        k = tl.arange(0, BLOCK_K)
        for start in range(0, K_MAIN, BLOCK_K):
            a = tl.load(a_ptr + m[:, None] * sam + (start + k)[None, :] * sak, mask=row_mask[:, None], other=0.0)
            w = tl.load(w_ptr + (start + k)[:, None] * swk + n[None, :] * swn, mask=col_mask[None, :], other=0.0)
            acc = tl.dot(a, w, acc, input_precision="ieee")
        for j in range(K_MAIN, K):
            a = tl.load(a_ptr + m * sam + j * sak, mask=row_mask, other=0.0)
            w = tl.load(w_ptr + n * swn + j * swk, mask=col_mask, other=0.0)
            acc += a[:, None] * w[None, :]
    tl.store(out_ptr + m[:, None] * som + n[None, :], acc, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _tma_kernel(a_ptr, w_ptr, out_desc, M, N, K, sam, sak, swn, swk, K_MAIN: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_SMS: tl.constexpr):
    n_tiles = tl.cdiv(N, BLOCK_N)
    tiles = tl.cdiv(M, BLOCK_M) * n_tiles
    k = tl.arange(0, BLOCK_K)
    for tile in tl.range(tl.program_id(0), tiles, NUM_SMS, flatten=True):
        m0 = (tile // n_tiles) * BLOCK_M
        n0 = (tile % n_tiles) * BLOCK_N
        m = m0 + tl.arange(0, BLOCK_M)
        n = n0 + tl.arange(0, BLOCK_N)
        row_mask, col_mask = m < M, n < N
        m = m.to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K_MAIN, BLOCK_K):
            a = tl.load(a_ptr + m[:, None] * sam + (start + k)[None, :] * sak, mask=row_mask[:, None], other=0.0)
            w = tl.load(w_ptr + (start + k)[:, None] * swk + n[None, :] * swn, mask=col_mask[None, :], other=0.0)
            acc = tl.dot(a, w, acc, input_precision="ieee")
        for j in range(K_MAIN, K):
            a = tl.load(a_ptr + m * sam + j * sak, mask=row_mask, other=0.0)
            w = tl.load(w_ptr + n * swn + j * swk, mask=col_mask, other=0.0)
            acc += a[:, None] * w[None, :]
        out_desc.store([m0, n0], acc)


def run(a, weight, out, bm, bn, bk, warps, stages, store_only=False):
    rows, k = a.shape
    grid = (triton.cdiv(rows, bm) * triton.cdiv(Q, bn),)
    _kernel[grid](a, weight, out, rows, Q, k, a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
                  out.stride(0), K_MAIN=k // bk * bk, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, STORE_ONLY=store_only,
                  num_warps=warps, num_stages=stages)


def run_tma(a, weight, desc, bm, bn, bk, warps, stages, programs):
    rows, k = a.shape
    _tma_kernel[(programs,)](a, weight, desc, rows, Q, k, a.stride(0), a.stride(1), weight.stride(0),
                             weight.stride(1), K_MAIN=k // bk * bk, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                             NUM_SMS=programs, num_warps=warps, num_stages=stages)


def main(argv=None):
    from triton.tools.tensor_descriptor import TensorDescriptor

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    rows_out = []
    tiles = [(64, 128, 4), (128, 64, 4), (128, 128, 8)]
    for cell, e in EDGES.items():
        hidden, cutoff = torch.randn(e, P, device="cuda"), torch.rand(e, 1, device="cuda")
        a = torch.cat([hidden * cutoff, cutoff], 1).contiguous()
        weight = torch.randn(Q, P + 1, device="cuda") / 8
        a68 = torch.nn.functional.pad(a, (0, 3)).contiguous()
        w68 = torch.nn.functional.pad(weight, (0, 3)).contiguous()
        out = torch.empty(e, Q, device="cuda")
        reference = a.double() @ weight.double().t()
        row = {"cell": cell, "E": e, "mm68_us": timed(lambda: torch.mm(a68, w68.t(), out=out)),
               "fill_us": timed(lambda: out.fill_(1.0))}
        for bm, bn, w in tiles:
            name = f"{bm}x{bn}w{w}"
            row[f"store {name}"] = timed(lambda: run(a, weight, out, bm, bn, 32, w, 1, True))
            for bk in (16, 32, 64):
                row[f"dot{bk} {name}"] = timed(lambda: run(a, weight, out, bm, bn, bk, w, 1))
            desc = TensorDescriptor.from_tensor(out, [bm, bn])
            for occupancy in (1, 2, 4):
                try:
                    row[f"tma x{occupancy} {name}"] = timed(
                        lambda: run_tma(a, weight, desc, bm, bn, 32, w, 1, sms * occupancy))
                except Exception as error:
                    row[f"tma x{occupancy} {name}"] = repr(error)[:160]
            out.zero_()
            try:
                run_tma(a, weight, desc, bm, bn, 32, w, 1, sms * 2)
                row[f"tma err {name}"] = float((out.double() - reference).abs().max() / reference.abs().max())
            except Exception as error:
                row[f"tma err {name}"] = repr(error)[:160]
        rows_out.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"gpu": props.name, "sms": sms, "torch": torch.__version__,
                                        "triton": triton.__version__, "rows": rows_out}, indent=1))


if __name__ == "__main__":
    main()
