"""Prototype: ``gated_products``' fused kernel without padding the D = 2l + 1 rows of an atom to a power of two.

Today's kernel stacks the D components of BLOCK_A atoms m-major in DP = next_power_of_2(D) row groups, so 1, 3 and 1
of every 4, 8 and 8 rows are masked (D = 3, 5, 7). Here a tile of R rows holds A = R // D whole atoms packed
atom-major (row r is component r % D of atom r // D), so at most D - 1 rows of R are masked and no atom crosses a
tile. The gate is read per row through the row's atom. The m-sum ``dgate[a] = sum_m acc[a D + m] * gated[a D + m]``
is a second ``tl.dot`` with the 0/1 selection matrix ``S[a, r] = (r // D == a)``: exact products by 0 and 1, the D
terms summed in ``tl.dot``'s order, deterministic, in registers. dgated rows are written contiguously.

Benchmark (the three gated segments of one call, MH-1's widths, synthetic operands): today's kernel, today's kernel
with the tile screen's large-cell tile, and this kernel over a small tile grid; every variant checked against today's
within 1e-5 relative.
    python unpadded.py --atoms 1944 21600 25920 --output DIR
"""

import argparse
import itertools
import json
import os
import statistics
from functools import cache
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

MUL = 512
DIMS = (3, 5, 7)


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unpadded_input_grads(g_ptr, wt_ptr, gated_ptr, gate_ptr, dgated_ptr, dgate_ptr, atoms_n, N, K, g_n, g_m,
                             gate_n, dgate_n, D: tl.constexpr, A: tl.constexpr, AP: tl.constexpr, R: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pa, pn = tl.program_id(0), tl.program_id(1)
        r = tl.arange(0, R)
        local = r // D
        m = r - local * D
        atom = (pa * A + local).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rmask = (local < A) & (atom < atoms_n)
        jmask = j < N
        tile = rmask[:, None] & jmask[None, :]
        g_row = atom * g_n + m * g_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((R, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(g_ptr + g_row[:, None] + k[None, :], mask=rmask[:, None] & (k < K)[None, :], other=0.0)
            b = tl.load(wt_ptr + k[:, None] * N + j[None, :], mask=(k < K)[:, None] & jmask[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        row = atom * D + m
        gate = tl.load(gate_ptr + atom[:, None] * gate_n + j[None, :], mask=tile, other=0.0)
        gated = tl.load(gated_ptr + row[:, None] * N + j[None, :], mask=tile, other=0.0)
        tl.store(dgated_ptr + row[:, None] * N + j[None, :], acc * gate, mask=tile)
        # dgate[a] = sum over the tile's rows of atom a: the 0/1 selection matrix times acc * gated
        sel_a = tl.arange(0, AP)
        select = ((local[None, :] == sel_a[:, None]) & rmask[None, :]).to(tl.float32)
        dgate = tl.dot(select, acc * gated, input_precision=PRECISION)
        first = (pa * A + sel_a).to(tl.int64)
        tl.store(dgate_ptr + first[:, None] * dgate_n + j[None, :], dgate,
                 mask=((sel_a < A) & (first < atoms_n))[:, None] & jmask[None, :])

    return unpadded_input_grads


def launch(grad, offset, wt, gated, gates, gate_offset, dgated, dgates, dim, tile, precision="ieee"):
    """One gated segment: dgated and dgates[:, gate_offset:gate_offset + N], as ``gated_products._fused``."""
    import triton

    rows, block_n, block_k, warps, stages = tile
    atoms, width = grad.shape
    mul_out, mul_in = int(wt.shape[0]), int(wt.shape[1])
    per = rows // dim
    padded = max(16, triton.next_power_of_2(per))
    grid = (triton.cdiv(atoms, per), triton.cdiv(mul_in, block_n))
    _kernel()[grid](grad.reshape(-1)[offset:], wt, gated, gates.reshape(-1)[gate_offset:], dgated,
                    dgates.reshape(-1)[gate_offset:], atoms, mul_in, mul_out, width, mul_out, gates.stride(0),
                    dgates.stride(0), D=dim, A=per, AP=padded, R=rows, BLOCK_N=block_n, BLOCK_K=block_k,
                    PRECISION=precision, num_warps=warps, num_stages=stages)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, nargs="+", default=[1944, 21600, 25920])
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import triton

    from fastmace.kernels.triton import gated_products as gp

    torch.manual_seed(0)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    width = MUL * (1 + sum(DIMS))
    today = (gp.ROWS, gp.BLOCK_N, gp.BLOCK_K, gp.NUM_WARPS, gp.NUM_STAGES)
    variants = [("today", today), ("padded-128x128", (128, 128, 32, 8, 3))]
    variants += [("unpadded", t) for t in itertools.product((64, 128), (64, 128), (32,), (4, 8), (2, 3))]
    padded_kernel = gp._kernel()
    results = []
    for atoms in args.atoms:
        grad = torch.randn(atoms, width, device=dev)
        gates = torch.randn(atoms, MUL * len(DIMS), device=dev)
        wts = [torch.randn(MUL, MUL, device=dev) / MUL ** 0.5 for _ in DIMS]
        gated = [torch.randn(atoms * d, MUL, device=dev) for d in DIMS]
        offsets = [MUL * (1 + sum(DIMS[:i])) for i in range(len(DIMS))]

        def run(kind, tile, outs):
            for i, d in enumerate(DIMS):
                if kind == "unpadded":
                    launch(grad, offsets[i], wts[i], gated[i], gates, MUL * i, outs[0][i], outs[1], d, tile)
                    continue
                rows, block_n, block_k, warps, stages = tile
                padded = triton.next_power_of_2(d)
                block_a = max(1, rows // padded)
                grid = (triton.cdiv(atoms, block_a), triton.cdiv(MUL, block_n))
                padded_kernel[grid](grad.reshape(-1)[offsets[i]:], wts[i], gated[i], gates.reshape(-1)[MUL * i:],
                                    outs[0][i], outs[1].reshape(-1)[MUL * i:], atoms, MUL, MUL, width, MUL,
                                    gates.stride(0), outs[1].stride(0), D=d, DP=padded, BLOCK_A=block_a,
                                    BLOCK_N=block_n, BLOCK_K=block_k, PRECISION="ieee", num_warps=warps,
                                    num_stages=stages)

        def buffers():
            return [torch.empty(atoms * d, MUL, device=dev) for d in DIMS], torch.empty(atoms, MUL * len(DIMS),
                                                                                          device=dev)

        ref, out = buffers(), buffers()
        run("today", today, ref)
        valid = []
        for kind, tile in variants:
            try:
                run(kind, tile, out)
                torch.cuda.synchronize()
            except Exception as error:  # shared memory or registers
                print(f"{atoms} {kind} {tile}: skipped ({type(error).__name__})", flush=True)
                continue
            err = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip([*out[0], out[1]],
                                                                                 [*ref[0], ref[1]]))
            if err > 1e-5:
                print(f"{atoms} {kind} {tile}: REJECTED, relative error {err:.2e}", flush=True)
                continue
            valid.append((kind, tile, err))
        times = {(kind, tile): [] for kind, tile, _ in valid}
        for _ in range(args.blocks):
            for kind, tile, _ in valid:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeat):
                    run(kind, tile, out)
                stop.record()
                torch.cuda.synchronize()
                times[(kind, tile)].append(start.elapsed_time(stop) / args.repeat)
        base = statistics.median(times[("today", today)])
        flop = 2 * atoms * sum(DIMS) * MUL * MUL
        errors = {(kind, tile): err for kind, tile, err in valid}
        ranked = sorted((statistics.median(v), k) for k, v in times.items())
        print(f"\n{atoms} atoms ({sms} SMs): today {base:.3f} ms ({flop / base / 1e9:.2f} TFLOP/s)", flush=True)
        for ms, (kind, tile) in ranked[:8]:
            print(f"  {kind:15s} {tile}  {ms:.3f} ms  {ms / base:.3f}x  {flop / ms / 1e9:.2f} TFLOP/s  "
                  f"err {errors[(kind, tile)]:.1e}", flush=True)
        padded_best = statistics.median(times[("padded-128x128", (128, 128, 32, 8, 3))]) if (
            "padded-128x128", (128, 128, 32, 8, 3)) in times else None
        for ms, (kind, tile) in ranked:
            results.append(dict(atoms=atoms, kind=kind, tile=tile, ms=ms, ratio=ms / base,
                                tflops=flop / ms / 1e9, error=errors[(kind, tile)]))
        if padded_best:
            print(f"  padded 128x128 tile: {padded_best / base:.3f}x", flush=True)
    record = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), sms=sms, torch=torch.__version__,
                  triton=triton.__version__, today=today, results=results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"unpadded-{os.uname().nodename}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
