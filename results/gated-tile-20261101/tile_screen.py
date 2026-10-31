"""Tile screen of ``gated_products``' fused kernel (``gated_input_grads``) on MH-1's shapes: the three gated segments
(D = 3, 5, 7; K = N = 512) of one ``linear_2`` input-gradient call, per atom count and tile. Synthetic operands; each
tile's outputs are checked against today's tile (``gated_products.ROWS``, ``BLOCK_N``, ``BLOCK_K``, ``NUM_WARPS``,
``NUM_STAGES``) before it is timed. Time: CUDA events around ``--repeat`` calls, median of ``--blocks`` blocks, the
tiles interleaved per block.
    python tile_screen.py --atoms 1944 21600 25920 --output DIR
"""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

MUL = 512
DIMS = (3, 5, 7)
GRID = dict(rows=(64, 128, 256), block_n=(64, 128), block_k=(32, 64), warps=(4, 8), stages=(2, 3, 4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, nargs="+", default=[1944, 21600, 25920])
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import triton

    from fastmace.kernels.triton import gated_products as gp

    torch.manual_seed(0)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    width = MUL * (1 + sum(DIMS))
    base = (gp.ROWS, gp.BLOCK_N, gp.BLOCK_K, gp.NUM_WARPS, gp.NUM_STAGES)
    tiles = [t for t in itertools.product(*GRID.values()) if t[0] >= 64]
    if base not in tiles:
        tiles.append(base)
    kernel = gp._kernel()
    results = []
    for atoms in args.atoms:
        grad = torch.randn(atoms, width, device=dev)
        gates = torch.randn(atoms, MUL * len(DIMS), device=dev)
        wts = [torch.randn(MUL, MUL, device=dev) / MUL ** 0.5 for _ in DIMS]
        gated = [torch.randn(atoms * d, MUL, device=dev) for d in DIMS]
        offsets = [MUL * (1 + sum(DIMS[:i])) for i in range(len(DIMS))]

        def run(tile, outs):
            rows, block_n, block_k, warps, stages = tile
            for i, d in enumerate(DIMS):
                padded = triton.next_power_of_2(d)
                block_a = max(1, rows // padded)
                grid = (triton.cdiv(atoms, block_a), triton.cdiv(MUL, block_n))
                kernel[grid](grad.reshape(-1)[offsets[i]:], wts[i], gated[i], gates.reshape(-1)[MUL * i:],
                             outs[0][i], outs[1].reshape(-1)[MUL * i:], atoms, MUL, MUL, width, MUL,
                             gates.stride(0), outs[1].stride(0), D=d, DP=padded, BLOCK_A=block_a,
                             BLOCK_N=block_n, BLOCK_K=block_k, PRECISION="ieee", num_warps=warps,
                             num_stages=stages)

        def buffers():
            return [torch.empty(atoms * d, MUL, device=dev) for d in DIMS], torch.empty(atoms, MUL * len(DIMS),
                                                                                          device=dev)

        ref = buffers()
        run(base, ref)
        out = buffers()  # one set of outputs for every tile: the check reads it right after the tile's run
        valid = []
        for tile in tiles:
            try:
                run(tile, out)
                torch.cuda.synchronize()
            except Exception as error:  # out of shared memory or registers for this tile
                print(f"{atoms} {tile}: skipped ({type(error).__name__})", flush=True)
                continue
            err = max(float(((a - b).abs().max() / b.abs().max())) for a, b in zip([*out[0], out[1]],
                                                                                   [*ref[0], ref[1]]))
            if err > 1e-5:
                print(f"{atoms} {tile}: rejected, relative error {err:.2e}", flush=True)
                continue
            valid.append(tile)
        times = {tile: [] for tile in valid}
        for _ in range(args.blocks):
            for tile in valid:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeat):
                    run(tile, out)
                stop.record()
                torch.cuda.synchronize()
                times[tile].append(start.elapsed_time(stop) / args.repeat)
        base_ms = statistics.median(times[base])
        flop = 2 * atoms * sum(DIMS) * MUL * MUL
        ranked = sorted(((statistics.median(v), t) for t, v in times.items()))
        for ms, tile in ranked[:6]:
            results.append(dict(atoms=atoms, tile=tile, ms=ms, ratio=ms / base_ms, tflops=flop / ms / 1e9))
        results.append(dict(atoms=atoms, tile=base, ms=base_ms, ratio=1.0, tflops=flop / base_ms / 1e9, base=True))
        print(f"\n{atoms} atoms ({sms} SMs): today's tile {base} {base_ms:.3f} ms, {flop / base_ms / 1e9:.2f} TFLOP/s",
              flush=True)
        for ms, tile in ranked[:8]:
            print(f"  {tile}  {ms:.3f} ms  {ms / base_ms:.3f}x  {flop / ms / 1e9:.2f} TFLOP/s", flush=True)
    record = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), sms=sms, torch=torch.__version__,
                  triton=triton.__version__, base=base, grid=GRID, results=results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"tile-screen-{os.uname().nodename}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
