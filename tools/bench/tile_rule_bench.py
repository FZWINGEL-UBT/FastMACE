"""Bound of a wave-aware tile rule for the CUTLASS 3xTF32 GEMMs (H200 research answers, 5 November 2026): every
recorded GEMM shape (``tools/bench/gemm_shapes.py`` output) timed with each output tile (``tf32x3.TILE``: 64x64,
32x64, 64x32) under a CUDA graph, alternating rounds in one process, with the smaller tiles checked bitwise against
64x64. Reports per shape the 64x64 grid's CTA count against the SM count and each tile's time, and the saving per
force call of choosing the best tile only where the 64x64 grid fills fewer than ``--max-waves`` waves of one CTA per
SM. Stop rule of the answers: drop the idea below 30 us per force call.

    python tools/bench/tile_rule_bench.py --shapes shapes.json [--output FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (soft file locks before torch)


def graph_of(fn):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def replay_us(graph, reps):
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        graph.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / reps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--max-waves", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from fastmace.kernels.cutlass import tf32x3

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gemms = json.loads(args.shapes.read_text())["gemms"]
    torch.manual_seed(0)
    rows, saving = [], 0.0
    for g in gemms:
        m, k, n, layout, calls = g["m"], g["k"], g["n"], g["layout"], g["calls"]
        if not g.get("tf32x3_tile", True) or not tf32x3.takes(m, k, n):
            continue
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda") if layout == "rr" else torch.randn(n, k, device="cuda").t()
        outs = {t: torch.empty(m, n, device="cuda") for t in tf32x3._TILES}

        def run(tile):
            def fn():
                tf32x3.TILE = tile
                tf32x3.mm_into(a, b, outs[tile])
            return fn

        graphs = {}
        for tile in tf32x3._TILES:
            graphs[tile] = graph_of(run(tile))
        tf32x3.TILE = "64x64"
        times = {t: [] for t in graphs}
        for r in range(args.rounds):
            order = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
            for t in order:
                times[t].append(replay_us(graphs[t], args.reps))
        med = {t: sorted(v)[len(v) // 2] for t, v in times.items()}
        bitwise = {t: bool(torch.equal(outs[t], outs["64x64"])) for t in outs}
        ctas = -(-m // 64) * -(-n // 64)
        waves = ctas / sms
        best = min(med, key=med.get)
        gain = (med["64x64"] - med[best]) * calls if waves < args.max_waves else 0.0
        saving += gain
        row = dict(m=m, k=k, n=n, layout=layout, calls=calls, ctas_64x64=ctas, waves=round(waves, 2),
                   us={t: round(v, 2) for t, v in med.items()}, bitwise=bitwise, best=best,
                   rule_saving_us_per_call=round(gain, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(gpu=torch.cuda.get_device_name(), sms=sms, max_waves=args.max_waves,
                   rule_saving_us_per_force_call=round(saving, 2), rows=rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
