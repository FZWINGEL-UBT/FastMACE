"""OPT4-T06: the OEq streaming forward on its real NbCl5 inputs, kernel only, across launch shapes.

Builds the default plan (as tools/step_breakdown.py), runs one force call to capture the interaction-2
forward's inputs, then rebuilds the kernel with other block sizes and NVRTC options and times each on the
same inputs (CUDA events, median of blocks), with registers, theoretical occupancy and a bitwise check
against the default launch. With --ncu-range, one launch of each variant runs between cudaProfilerStart/Stop
for `ncu --profile-from-start off`. Untimed microbenchmark: run inside a tools/ci/local_gpu.py reservation.

    PYTHONPATH=<src> python results/opt4-T06/probe_forward.py --checkpoint ~/models/mace-mh-1.model \
        --structure NbCl5 --output DIR [--ncu-range]
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

VARIANTS = [  # (label, num_threads, NVRTC options)
    ("default-128", 128, ()),
    ("block-32", 32, ()),
    ("block-64", 64, ()),
    ("block-256", 256, ()),
    ("block-512", 512, ()),
    ("maxreg-64", 128, ("--maxrregcount=64",)),
    ("maxreg-72", 128, ("--maxrregcount=72",)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--launches", type=int, default=50)
    p.add_argument("--ncu-range", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path.cwd() / "tools"))
    import ab_step_time
    import step_breakdown as sb

    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming import runtime

    built = []
    original_init = runtime.StreamingKernels.__init__

    def recording_init(self, problem, dtype=np.float32, num_threads=128, layout="mul_ir", **kwargs):
        original_init(self, problem, dtype, num_threads, layout, **kwargs)
        built.append((self, problem, dtype, layout))

    runtime.StreamingKernels.__init__ = recording_init
    captured = []
    original_forward = runtime.StreamingKernels.forward

    def recording_forward(self, x, sh, weights, senders, row_ptr, out):
        original_forward(self, x, sh, weights, senders, row_ptr, out)
        captured.append((self, x.clone(), sh.clone(), weights.clone(), senders.clone(), row_ptr.clone(), out.clone()))

    ns = argparse.Namespace(checkpoint=args.checkpoint.expanduser().resolve(), model="mh1-r2scan", backend="hybrid",
                            temperature_K=1000.0, rewrites={})
    args.checkpoint = ns.checkpoint
    _, atoms = ab_step_time.read_structure(args.structure, None)
    engine, _ = sb.build_engine(ns, atoms)
    field = engine.force_field
    positions = engine.state.positions.detach().clone()
    field(positions)
    torch.cuda.synchronize()
    runtime.StreamingKernels.forward = recording_forward
    field(positions)
    torch.cuda.synchronize()
    runtime.StreamingKernels.forward = original_forward
    if len(captured) != 1:
        raise SystemExit(f"expected one streaming forward per force call, got {len(captured)}")
    default, x, sh, weights, senders, row_ptr, reference = captured[0]
    problem, dtype, layout = next((pr, dt, lay) for k, pr, dt, lay in built if k is default)
    degrees = (row_ptr[1:] - row_ptr[:-1]).float()
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), structure=args.structure,
                  device=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0).multi_processor_count,
                  nodes=int(x.shape[0]), edges=int(sh.shape[0]), real_edges=int(row_ptr[-1].item()),
                  x=list(x.shape), sh=list(sh.shape), weights=list(weights.shape), out=list(reference.shape),
                  channels=default.channels, layout=layout,
                  degree=dict(mean=degrees.mean().item(), min=degrees.min().item(), max=degrees.max().item(),
                              std=degrees.std().item()),
                  bytes=dict(weights=weights.numel() * 4, x=x.numel() * 4, sh=sh.numel() * 4, out=reference.numel() * 4),
                  variants=[])

    def attributes(kernels):
        function = kernels._forward
        regs = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function)[1]
        local = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function)[1]
        per_sm = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, kernels.num_threads,
                                                                    kernels.shared_memory)[1]
        return regs, local, per_sm

    def timed(launch):
        for _ in range(10):
            launch()
        torch.cuda.synchronize()
        per_launch = []
        for _ in range(args.blocks):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.launches):
                launch()
            stop.record()
            stop.synchronize()
            per_launch.append(start.elapsed_time(stop) * 1000.0 / args.launches)
        return statistics.median(per_launch), min(per_launch), max(per_launch)

    variants = []
    for label, threads, options in VARIANTS:
        kernels = default if label == "default-128" else runtime.StreamingKernels(
            problem, dtype, threads, layout, extra_options=options)
        variants.append((label, threads, options, kernels))
    max_warps = torch.cuda.get_device_properties(0).max_threads_per_multi_processor // 32
    for label, threads, options, kernels in variants:
        out = torch.empty_like(reference)
        median, low, high = timed(lambda: kernels.forward(x, sh, weights, senders, row_ptr, out))
        regs, local, per_sm = attributes(kernels)
        blocks = (x.shape[0] * kernels.channels + threads - 1) // threads
        record["variants"].append(dict(label=label, num_threads=threads, options=list(options), registers=regs,
                                       local_bytes=local, blocks=blocks, blocks_per_sm=per_sm,
                                       theoretical_occupancy_pct=100.0 * per_sm * threads / 32 / max_warps,
                                       waves=blocks / (per_sm * record["sms"]), median_us=median, min_us=low,
                                       max_us=high, bitwise_equal=bool(torch.equal(out, reference)),
                                       max_abs_diff=(out - reference).abs().max().item(),
                                       weights_read_gbs=record["bytes"]["weights"] / median / 1e3))
        print(json.dumps(record["variants"][-1]), flush=True)
    # Floors on the same buffers: a pure read of the weights (sum), and a copy (read + write) of them.
    scratch = torch.empty_like(weights)
    for name, launch in (("weights_sum", lambda: weights.sum()), ("weights_copy", lambda: scratch.copy_(weights))):
        median, low, high = timed(launch)
        record[name] = dict(median_us=median, min_us=low, max_us=high)
    record["weights_read_floor_gbs"] = record["bytes"]["weights"] / record["weights_sum"]["median_us"] / 1e3
    print(json.dumps({k: record[k] for k in ("weights_sum", "weights_copy", "weights_read_floor_gbs")}), flush=True)
    if args.ncu_range:
        out = torch.empty_like(reference)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for label, threads, options, kernels in variants:
            kernels.forward(x, sh, weights, senders, row_ptr, out)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        record["ncu_order"] = [v[0] for v in variants]
    name = "probe_ncu.json" if args.ncu_range else "probe.json"
    (args.output / name).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
