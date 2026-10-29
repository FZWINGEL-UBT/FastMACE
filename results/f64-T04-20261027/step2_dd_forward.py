"""f64-T04 step 2 (falsifier B): df64 streaming forward versus the native float64 forward on recorded operands.

Usage: python step2_dd_forward.py OUT.json ops64-*.pt ...
For each record: the native float64 kernels (StreamingKernels, float64, the plan's layout and block size) and the
df64 forward from ddconv.dd_forward_source (variants cvt and bits), launched with the same grid. Timing: each
kernel captured as a CUDA graph of 20 calls, each after a 96 MB L2 flush (flush timed alone and subtracted; the
helper of ~/tmp/opt7/common.py), in ROUNDS alternating rounds native, dd_cvt, dd_bits; median over rounds.
Accuracy: max |dd - native| / max |native| and the same for a float32 rendering of the kernel on the rounded
operands (scale), plus the native kernel against the plan's recorded output.
"""
import ctypes
import json
import statistics
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)
import numpy as np
import torch
from cuda.bindings import driver
from openequivariance.core.e3nn_lite import TPProblem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddconv import dd_forward_source  # noqa: E402
from fastmace.kernels.oeq_streaming import StreamingKernels  # noqa: E402
from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin  # noqa: E402

ROUNDS = 7


def graph_time(fn, calls=20, blocks=9, flush_bytes=96 << 20):
    """Median ms per call of fn in a CUDA graph of `calls` calls each after an L2 flush, flush subtracted."""
    junk = torch.empty(flush_bytes // 4, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            junk.fill_(1.0)
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs = []
    for body in (lambda: (junk.fill_(1.0), fn()), lambda: junk.fill_(1.0)):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                body()
        graphs.append(g)
    times = {0: [], 1: []}
    for _ in range(blocks):
        for index, g in enumerate(graphs):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()
            stop.record()
            torch.cuda.synchronize()
            times[index].append(start.elapsed_time(stop) / calls)
    return statistics.median(times[0]) - statistics.median(times[1])


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))


def make_problem(p):
    return TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                     [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                      for i in p["instructions"]],
                     irrep_normalization="none", path_normalization="none", shared_weights=False,
                     internal_weights=False)


class DDForward:
    def __init__(self, native, convert):
        self.native = native
        source = dd_forward_source(native.source, convert)
        self.source = source
        cubin = compile_cubin(source, torch.cuda.get_device_capability(), ("--fmad=false",))
        self.module = _check(driver.cuModuleLoadData(cubin))
        self.function = _check(driver.cuModuleGetFunction(self.module, b"oeq_projected_forward_dd"))
        self.registers = _check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.function))

    def forward(self, x, sh, weights, senders, row_ptr, out):
        k = self.native
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (nodes * k.channels + k.num_threads - 1) // k.num_threads
        p = ctypes.c_void_p
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(self.function, blocks, 1, 1, k.num_threads, 1, 1, k.shared_memory, stream,
                                     ((nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(),
                                       senders.data_ptr(), row_ptr.data_ptr(), out.data_ptr()),
                                      (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p)), 0))


def main():
    out_json, paths = sys.argv[1], sys.argv[2:]
    rows = []
    for path in paths:
        record = torch.load(path)["records"][0]
        r = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in record.items()}
        p = r["problem"]
        x, sh, w, s, rp = (r[k].contiguous() for k in ("x", "sh", "w", "senders", "row_ptr"))
        assert x.dtype == torch.float64, x.dtype
        problem = make_problem(p)
        threads = p.get("num_threads", 128)
        native = StreamingKernels(problem, np.float64, num_threads=threads, layout=p["layout"])
        native_fn = native._forward
        native_regs = _check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, native_fn))
        dds = {c: DDForward(native, c) for c in ("cvt", "bits")}
        out_native = torch.zeros(x.shape[0], native.output_dim, dtype=torch.float64, device="cuda")
        native.forward(x, sh, w, s, rp, out_native)
        outs = {}
        for c, dd in dds.items():
            outs[c] = torch.zeros_like(out_native)
            dd.forward(x, sh, w, s, rp, outs[c])
        # float32 scale: the float32 rendering on rounded operands.
        k32 = StreamingKernels(problem, np.float32, num_threads=threads, layout=p["layout"])
        out32 = torch.zeros(x.shape[0], k32.output_dim, dtype=torch.float32, device="cuda")
        k32.forward(x.float(), sh.float(), w.float(), s, rp, out32)
        torch.cuda.synchronize()
        row = dict(file=Path(path).name, nodes=x.shape[0], edges=sh.shape[0], layout=p["layout"],
                   threads=threads, native_vs_plan_output=rel(out_native, r["out"]),
                   rel_dd_cvt=rel(outs["cvt"], out_native), rel_dd_bits=rel(outs["bits"], out_native),
                   rel_fp32=rel(out32, out_native), registers=dict(native=native_regs,
                                                                   cvt=dds["cvt"].registers,
                                                                   bits=dds["bits"].registers))
        fns = {"native": lambda: native.forward(x, sh, w, s, rp, out_native),
               "dd_cvt": lambda: dds["cvt"].forward(x, sh, w, s, rp, outs["cvt"]),
               "dd_bits": lambda: dds["bits"].forward(x, sh, w, s, rp, outs["bits"]),
               "fp32": lambda: k32.forward(x.float(), sh.float(), w.float(), s, rp, out32)}
        x32, sh32, w32 = x.float(), sh.float(), w.float()
        fns["fp32"] = lambda: k32.forward(x32, sh32, w32, s, rp, out32)
        times = {k: [] for k in fns}
        for _ in range(ROUNDS):
            for name, fn in fns.items():
                times[name].append(graph_time(fn))
        row["ms_rounds"] = times
        row["ms"] = {k: statistics.median(v) for k, v in times.items()}
        row["speedup_vs_native"] = {k: row["ms"]["native"] / v for k, v in row["ms"].items()}
        print(json.dumps({k: v for k, v in row.items() if k != "ms_rounds"}), flush=True)
        rows.append(row)
    Path(out_json).write_text(json.dumps(dict(device=torch.cuda.get_device_name(), rows=rows), indent=1))


if __name__ == "__main__":
    main()
