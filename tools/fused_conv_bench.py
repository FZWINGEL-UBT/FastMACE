"""OPT2-T01 gate 1: the folded radial last layer fused into the streaming convolution's forward (prototype bench).

For MH-1's second interaction (its OEq problem, read from the loaded hybrid model) and the receiver-sorted edges
of NaCl, NbCl5 and NaNbO3 3x3x3 (ASE neighbour list at the model's r_max), random float32 inputs:

- reference: ``w = a @ Wf.T`` (cuBLAS, as the folded layer) and the vendored streaming forward reading ``w``;
- fused: ``tools/fused_conv/fused_forward.cuh``, which forms ``w`` in registers from ``a`` and ``Wf``.

It checks the message against the reference and times both, each captured in a CUDA graph (median of --repeats
replays). Configurations are --configs CS,R,T (channel slice, receivers per block, edges per chunk). No library
code changes; nothing is cached. Use an idle GPU (local reservation).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    p.add_argument("--configs", nargs="+", default=["16,8,4", "16,8,2", "8,16,4", "32,4,4"])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--model", default="mh1-r2scan")
    return p.parse_args(argv)


def edges_of(atoms, cutoff):
    import numpy as np
    from ase.neighborlist import neighbor_list

    i, j = neighbor_list("ij", atoms, cutoff)
    order = np.argsort(j, kind="stable")  # receiver-sorted (MACE: edge_index[1] = receiver)
    return i[order].astype(np.int32), j[order].astype(np.int32)


def render(problem, layout, cs, r, t, k):
    import numpy as np
    from jinja2 import Environment, FileSystemLoader

    from fastmace.kernels.oeq_streaming import jinja_env
    from fastmace.kernels.oeq_streaming.runtime import render_kernels
    from fastmace.kernels.oeq_streaming.schedule import Input

    schedule, _ = render_kernels(problem, np.float32, 128, layout)
    base = jinja_env.get_jinja_environment()
    env = Environment(loader=FileSystemLoader([str(HERE / "fused_conv"),
                                               str(Path(jinja_env.__file__).parent)]),
                      extensions=["jinja2.ext.do"])
    env.globals.update(base.globals)
    source = env.get_template("fused_forward.cuh").render(
        scalar="float", schedule=schedule, layout=layout, Input=Input, k=k, cs=cs, r=r, t=t,
        forward_jvp_active=(False, False, False), backward_jvp_active=(False, False, False, False),
        backward_jvp_dout_active=False)
    return schedule, source


class Fused:
    def __init__(self, source, schedule, cs, r, t, k):
        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

        self.cs, self.r, self.t, self.k = cs, r, t, k
        self.schedule = schedule
        cubin = compile_cubin(source, torch.cuda.get_device_capability())
        torch.cuda.current_stream()
        self.module = _check(driver.cuModuleLoadData(cubin))
        self.function = _check(driver.cuModuleGetFunction(self.module, b"fused_projected_forward"))
        slots = sum(p.edge_mul for p in schedule.paths)
        self.shared = 4 * (k * slots * cs + r * t * k)
        if self.shared > 48 * 1024:
            _check(driver.cuFuncSetAttribute(
                self.function, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.shared))
        self.registers = _check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.function))

    def __call__(self, x, sh, a, wf, senders, row_ptr, out):
        import ctypes

        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check

        nodes, edges = x.shape[0], sh.shape[0]
        grid = ((nodes + self.r - 1) // self.r, self.schedule.channels // self.cs)
        p = ctypes.c_void_p
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), a.data_ptr(), wf.data_ptr(), senders.data_ptr(),
                  row_ptr.data_ptr(), out.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(self.function, grid[0], grid[1], 1, self.r * self.cs, 1, 1, self.shared,
                                     stream, (values, types), 0))


def timed(fn, repeats):
    import torch

    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        out.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(out)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from ase.io import read

    sys.path.insert(0, str(HERE))
    import force_stress_gate as gate  # structure_spec

    from fastmace.kernels.oeq_streaming import StreamingKernels
    from fastmace.model import loading

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = loading.load(args.model, backend="hybrid", dtype="float32", device="cuda:0")
    model = loaded.model
    interaction = model.interactions[1]
    conv = interaction.conv_tp
    problem = conv.input_args["problem"]
    layout = "ir_mul" if conv.layout_transpose_in is not None else "mul_ir"
    cutoff = float(model.r_max)
    reference = StreamingKernels(problem, np.float32, layout=layout)
    k = 65
    g = torch.Generator(device="cuda").manual_seed(0)
    summary = dict(tool="fused_conv_bench", created_utc=datetime.now(timezone.utc).isoformat(),
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__, layout=layout,
                   weight_dim=reference.weight_dim, input_dim=reference.input_dim, output_dim=reference.output_dim,
                   edge_dim=reference.edge_dim, channels=reference.channels, cutoff=cutoff, cases={})
    kernels = {}
    for config in args.configs:
        cs, r, t = (int(v) for v in config.split(","))
        schedule, source = render(problem, layout, cs, r, t, k)
        kernels[config] = Fused(source, schedule, cs, r, t, k)
        (args.output / f"fused_forward_{cs}_{r}_{t}.cu").write_text(source)
    for name in args.structures:
        _, path, repeat = gate.structure_spec(name, (3, 3, 3), os.environ.get("FASTMACE_STRUCTURES", "."))
        atoms = read(path).repeat(repeat)
        senders, receivers = edges_of(atoms, cutoff)
        nodes, edges = len(atoms), len(senders)
        counts = np.bincount(receivers, minlength=nodes)
        row_ptr = torch.tensor(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32), device="cuda")
        senders_t = torch.tensor(senders, device="cuda")
        x = torch.randn((nodes, reference.input_dim), generator=g, device="cuda")
        sh = torch.randn((edges, reference.edge_dim), generator=g, device="cuda")
        a = torch.randn((edges, k), generator=g, device="cuda")
        wf = torch.randn((reference.weight_dim, k), generator=g, device="cuda") / k ** 0.5
        ref_out = x.new_empty((nodes, reference.output_dim))
        w = torch.empty((edges, reference.weight_dim), device="cuda")

        def ref():
            torch.mm(a, wf.t(), out=w)
            reference.forward(x, sh, w, senders_t, row_ptr, ref_out)

        ref_us = timed(ref, args.repeats)
        gemm_us = timed(lambda: torch.mm(a, wf.t(), out=w), args.repeats)
        conv_us = timed(lambda: reference.forward(x, sh, w, senders_t, row_ptr, ref_out), args.repeats)
        ref()
        torch.cuda.synchronize()
        expected = ref_out.clone()
        entry = dict(nodes=nodes, edges=edges, mean_degree=edges / nodes, reference_us=ref_us, gemm_us=gemm_us,
                     conv_us=conv_us, fused={})
        for config, kern in kernels.items():
            out = torch.zeros_like(ref_out)
            us = timed(lambda kern=kern, out=out: kern(x, sh, a, wf, senders_t, row_ptr, out), args.repeats)
            error = float((out - expected).abs().max() / expected.abs().max())
            entry["fused"][config] = dict(us=us, saving=1 - us / ref_us, max_rel_error=error,
                                          registers=kern.registers, shared_bytes=kern.shared)
        summary["cases"][name] = entry
        print(name, {k_: (round(v, 1) if isinstance(v, float) else v) for k_, v in entry.items() if k_ != "fused"},
              {c: (round(f["us"], 1), round(f["saving"], 3), f"{f['max_rel_error']:.1e}", f["registers"])
               for c, f in entry["fused"].items()}, flush=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
