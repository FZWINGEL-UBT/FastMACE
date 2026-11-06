"""Idea 33 timing bound (H200 ideas plan): CUTLASS 3.x native Hopper TF32 wgmma GEMMs against today's sm_80
mma.sync 3xTF32 GEMMs (fastmace.kernels.cutlass.tf32x3) on the force call's own shapes, in one process on one GPU.

1. Shapes: the 3xTF32 products that go to CUTLASS (``tf32x3_gemm._launch`` with ``tf32x3.takes``) in one eager
   force call of the minimal plan's engine (as tools/step_breakdown.py builds it), after equilibration steps.
2. Per distinct (M, K, N, B layout), random operands, every time the median µs per launch of 20 launches captured in
   one CUDA graph (no host launch cost, as in MD):
   - today: ``tf32x3.mm_into`` (production tile), timed before and after the wgmma side (the smaller is kept);
   - wgmma single pass: every config of libidea33_wgmma.so (B as K-major B^T, which a weight can be stored as);
   - wgmma pre-split, one GEMM: K x 3 with A' = [A_big | A_big | A_small], B' = [B_big; B_small; B_big] (a correct
     3xTF32 sum) and K x 2 with A' = [A_big | A_small], B' = [B_big; B_big] (two of the three terms: a timing
     proxy only); the split of A is timed separately (three torch kernels; B, a weight, is split once);
   - errors against a float64 product, scaled by |A| |B| (opt-T01 §7).
3. Ratio per shape: 3 x best single-pass µs / today's µs; the six largest shapes by today's time per force call.

    python tools/bench/idea33/bench.py --checkpoint $FASTMACE_MODELS/mace-omat-0-medium.model \
        --structure results/loup-a100-20261103/SrTiO3.cif@10x10x10 --output $OUT/idea33.json
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent
sys.path.insert(0, str(TOOLS))

import fastmace  # noqa: E402,F401  (before torch: soft file locks)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", required=True)
    p.add_argument("--lib", type=Path, default=HERE / "libidea33_wgmma.so")
    p.add_argument("--steps", type=int, default=30, help="equilibration MD steps before the recorded force call")
    p.add_argument("--launches", type=int, default=20)
    p.add_argument("--replays", type=int, default=25)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import ab_step_time
    oeq_runtime = ab_step_time.prepare_fastmace_hybrid()
    import step_breakdown
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    record = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                  sms=torch.cuda.get_device_properties(0).multi_processor_count, torch=torch.__version__,
                  oeq_runtime=oeq_runtime, structure=args.structure, model=args.model,
                  today_tile=tf32x3.TILE, backend=tf32x3_gemm.backend(torch.device("cuda")),
                  build=(HERE / "build.txt").read_text() if (HERE / "build.txt").is_file() else None)
    print(json.dumps({k: record[k] for k in ("gpu", "capability", "sms", "backend", "today_tile")}), flush=True)

    # 1. Shapes of one eager force call.
    started = time.perf_counter()
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    _, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(1))
    Stationary(atoms)
    build_args = types.SimpleNamespace(backend="hybrid", model=args.model, checkpoint=args.checkpoint, rewrites=None,
                                       temperature_K=300.0, timestep_fs=1.0)
    engine, plan = step_breakdown.build_engine(build_args, atoms)
    for _ in range(args.steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    field = engine.force_field
    positions = engine.state.positions.detach().clone()
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    seen = collections.Counter()
    other = collections.Counter()
    original = tf32x3_gemm._launch

    def launch(a, b, out):
        m, k = a.shape
        n = b.shape[1]
        layout = "rc" if b.stride(0) == 1 and n > 1 else "rr"
        key = (int(m), int(k), int(n), layout)
        if tf32x3_gemm.uses_tf32x3_tile(k, n) and tf32x3_gemm._cutlass(a.device) and tf32x3.takes(m, k, n):
            seen[key] += 1
        else:
            other[key] += 1
        return original(a, b, out)

    tf32x3_gemm._launch = launch
    try:
        field(positions)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch = original
    record["setup_s"] = time.perf_counter() - started
    record["natoms"] = len(atoms)
    record["cutlass_launches_per_force_call"] = sum(seen.values())
    record["other_tf32x3_mm_launches"] = [dict(m=m, k=k, n=n, layout=lay, calls=c)
                                          for (m, k, n, lay), c in sorted(other.items())]
    print(f"CUTLASS products per force call: {sum(seen.values())} launches, {len(seen)} shapes "
          f"(setup {record['setup_s']:.0f} s)", flush=True)
    del field, engine
    torch.cuda.empty_cache()

    # 2. Timings.
    lib = ctypes.CDLL(str(args.lib))
    lib.idea33_configs.restype = ctypes.c_int
    lib.idea33_name.restype = ctypes.c_char_p
    lib.idea33_name.argtypes = [ctypes.c_int]
    lib.idea33_create.restype = ctypes.c_void_p
    lib.idea33_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.idea33_run.restype = ctypes.c_int
    lib.idea33_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.idea33_run_n.restype = ctypes.c_int
    lib.idea33_run_n.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.idea33_destroy.argtypes = [ctypes.c_void_p]
    configs = [lib.idea33_name(i).decode() for i in range(lib.idea33_configs())]
    record["wgmma_configs"] = configs

    def graph_timed(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.launches):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.replays):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1e3 / args.launches)
        del graph
        return statistics.median(times)

    def loop_timed(handle):
        """Fallback when capture fails: ``launches`` launches from C between two events."""
        s = torch.cuda.current_stream().cuda_stream
        lib.idea33_run_n(handle, s, 3)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.replays):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            lib.idea33_run_n(handle, s, args.launches)
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1e3 / args.launches)
        return statistics.median(times)

    def scaled_error(out, reference, scale):
        return float(((out.double() - reference).abs() / scale.clamp_min(1e-30)).max())

    def wgmma_all(a, bt, m, n, k, reference=None, scale=None):
        """{config: dict(us, error, timing)} for C = a @ bt.T, a [m, k] and bt [n, k] contiguous."""
        out = torch.empty(m, n, device="cuda")
        results = {}
        for i, name in enumerate(configs):
            status = ctypes.c_int(-1)
            handle = lib.idea33_create(i, m, n, k, a.data_ptr(), bt.data_ptr(), out.data_ptr(),
                                       ctypes.byref(status))
            if not handle:
                results[name] = dict(status=status.value)
                continue
            try:
                out.zero_()
                code = lib.idea33_run(handle, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                entry = dict(status=0, run_code=code)
                if reference is not None:
                    entry["error"] = scaled_error(out, reference, scale)
                try:
                    entry["us"] = graph_timed(lambda: lib.idea33_run(handle, torch.cuda.current_stream().cuda_stream))
                    entry["timing"] = "graph"
                except Exception as error:  # noqa: BLE001  (capture of the foreign runtime's launch)
                    torch.cuda.synchronize()
                    entry["graph_error"] = repr(error)[:200]
                    entry["us"] = loop_timed(handle)
                    entry["timing"] = "c-loop"
                results[name] = entry
            finally:
                torch.cuda.synchronize()
                lib.idea33_destroy(handle)
        return results

    def best(results):
        ok = {name: r for name, r in results.items() if "us" in r and r.get("run_code", 1) == 0}
        if not ok:
            return None, None
        name = min(ok, key=lambda x: ok[x]["us"])
        return name, ok[name]["us"]

    mask = torch.tensor(-8192, dtype=torch.int32, device="cuda")  # 0xFFFFE000: the TF32 part (truncation)

    def split(x):
        big = (x.view(torch.int32) & mask).view(torch.float32)
        return big, x - big

    generator = torch.Generator(device="cpu").manual_seed(33)
    shapes = []
    for (m, k, n, layout), calls in sorted(seen.items(), key=lambda item: -item[0][0] * item[0][1] * item[0][2]):
        a = torch.randn(m, k, generator=generator).cuda()
        b_rows = torch.randn(k, n, generator=generator).cuda() if layout == "rr" else None
        b = b_rows if layout == "rr" else torch.randn(n, k, generator=generator).cuda().t()
        bt = b.t().contiguous()  # [n, k]: K-major
        out = torch.empty(m, n, device="cuda")
        reference = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        a_fit, b_fit = tf32x3.fit(a, b)
        assert tf32x3.eligible(a_fit, b_fit, out), (m, k, n, layout)
        tf32x3.mm_into(a_fit, b_fit, out)
        torch.cuda.synchronize()
        today_error = scaled_error(out, reference, scale)
        today_1 = graph_timed(lambda: tf32x3.mm_into(a_fit, b_fit, out))
        single = wgmma_all(a, bt, m, n, k, reference, scale)
        a_big, a_small = split(a)
        b_big, b_small = split(bt)
        a3 = torch.cat([a_big, a_big, a_small], dim=1).contiguous()
        b3 = torch.cat([b_big, b_small, b_big], dim=1).contiguous()
        k3 = wgmma_all(a3, b3, m, n, 3 * k, reference, scale)
        a2 = torch.cat([a_big, a_small], dim=1).contiguous()
        b2 = torch.cat([b_big, b_big], dim=1).contiguous()
        k2 = wgmma_all(a2, b2, m, n, 2 * k, reference, scale)
        a3i = a3.view(torch.int32)

        def split_a():
            torch.bitwise_and(a.view(torch.int32), mask, out=a3i[:, :k])
            a3i[:, k:2 * k].copy_(a3i[:, :k])
            torch.sub(a, a3[:, :k], out=a3[:, 2 * k:])

        split_us = graph_timed(split_a)
        today_2 = graph_timed(lambda: tf32x3.mm_into(a_fit, b_fit, out))
        today = min(today_1, today_2)
        s_name, s_us = best(single)
        k3_name, k3_us = best(k3)
        k2_name, k2_us = best(k2)
        row = dict(m=m, k=k, n=n, layout=layout, calls=calls, today_us=today, today_us_runs=[today_1, today_2],
                   today_error=today_error, today_us_per_force_call=today * calls,
                   single_best=s_name, single_us=s_us, x3_us=None if s_us is None else 3 * s_us,
                   ratio_x3=None if s_us is None else 3 * s_us / today,
                   presplit_k3_best=k3_name, presplit_k3_us=k3_us,
                   ratio_k3=None if k3_us is None else k3_us / today,
                   ratio_k3_with_split=None if k3_us is None else (k3_us + split_us) / today,
                   presplit_k2_best=k2_name, presplit_k2_us=k2_us,
                   ratio_k2=None if k2_us is None else k2_us / today, split_a_us=split_us,
                   single=single, presplit_k3=k3, presplit_k2=k2)
        shapes.append(row)
        print(f"{m:6d} {k:4d} {n:4d} {layout} x{calls:2d}  today {today:7.1f} ({today_1:.1f}/{today_2:.1f}) err "
              f"{today_error:.1e} | 1-pass {s_us if s_us is None else round(s_us, 1)} [{s_name}] err "
              f"{single.get(s_name, {}).get('error', float('nan')):.1e} -> x3/today "
              f"{row['ratio_x3'] if row['ratio_x3'] is None else round(row['ratio_x3'], 3)} | K3 "
              f"{k3_us if k3_us is None else round(k3_us, 1)} err {k3.get(k3_name, {}).get('error', float('nan')):.1e}"
              f" | K2 {k2_us if k2_us is None else round(k2_us, 1)} | split {split_us:.1f}", flush=True)
        del a, b, bt, out, reference, scale, a3, b3, a2, b2, a_fit, b_fit, b_rows
        torch.cuda.empty_cache()
    record["shapes"] = shapes

    # 3. The six largest by today's time per force call.
    six = sorted(shapes, key=lambda r: -r["today_us_per_force_call"])[:6]
    total = dict(
        today_us=sum(r["today_us_per_force_call"] for r in six),
        x3_us=sum(r["x3_us"] * r["calls"] for r in six if r["x3_us"] is not None),
        k3_us=sum(r["presplit_k3_us"] * r["calls"] for r in six if r["presplit_k3_us"] is not None),
        k3_split_us=sum((r["presplit_k3_us"] + r["split_a_us"]) * r["calls"] for r in six
                        if r["presplit_k3_us"] is not None),
        k2_us=sum(r["presplit_k2_us"] * r["calls"] for r in six if r["presplit_k2_us"] is not None),
        all_shapes_today_us=sum(r["today_us_per_force_call"] for r in shapes))
    total["ratio_x3"] = total["x3_us"] / total["today_us"]
    total["ratio_k3"] = total["k3_us"] / total["today_us"]
    total["ratio_k3_with_split"] = total["k3_split_us"] / total["today_us"]
    total["ratio_k2"] = total["k2_us"] / total["today_us"]
    total["threshold"] = 0.6
    total["verdict"] = "build" if total["ratio_x3"] <= 0.6 else "stop"
    record["six"] = [dict(m=r["m"], k=r["k"], n=r["n"], layout=r["layout"], calls=r["calls"]) for r in six]
    record["six_total"] = total
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    print(json.dumps(total, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
