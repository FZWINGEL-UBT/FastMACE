"""OPT8-T01 falsifier: the radial-table output straight into the streaming convolution (plans/opt-8.md).

Today's chain on the second MH-1 interaction (the "chain"):
- forward: ``w = a @ W.T`` (cuBLAS, ``w [E, 1280]`` in DRAM), then the receiver-streaming forward reads ``w``;
- backward: ``streaming_conv_folded_backward`` (``chunked_conv_backward``, default): the edge-owned streaming
  backward writes ``dw`` per L2-sized edge chunk and the Triton input gradient ``da = dw W`` reads it on a side
  stream.

Variants timed against it (same process, CUDA graphs of ``--reps`` calls, each call after an L2 flush that is
timed alone and subtracted, ``--blocks`` alternating blocks, medians):

Ceilings (part a):
- ``stub_fwd``/``stub_bwd``: the vendored streaming kernels with every ``w`` read replaced by a register-only
  value and every ``dw`` store by a register sum (no ``w``/``dw`` traffic; the arithmetic stays);
- ``gemm_l2``: the ``w`` product of all edges, chunk by chunk into one L2-resident buffer (cuBLAS, no DRAM
  write); ``igrad_l2``: the Triton input gradient of all edges from one L2-resident ``dw`` chunk buffer.
A fused kernel does the stub's work plus the products' arithmetic: forward >= stub_fwd and >= gemm_l2,
backward >= stub_bwd and >= gemm_l2 + igrad_l2 (the backward rebuilds ``w``).

Prototype (part b): ``fused_fwd``/``fused_bwd`` from ``opt8_fused_kernels.py`` (plain FP32 SIMT), checked
against the chain.

Stop rule: fused forward + backward >= 25 % faster than the chain on NaNbO3 and NbCl5.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ops", nargs="+", type=Path, help="ops-*.pt from opt8_record_operands.py")
    p.add_argument("--output", type=Path, required=True, help="JSON lines file (appended)")
    p.add_argument("--variants", nargs="+", default=["stub", "l2"],
                   help="groups: stub (stub_fwd, stub_bwd), l2 (gemm_l2, igrad_l2), fused (fused_fwd, fused_bwd)")
    p.add_argument("--fused-config", default="", help="key=value,... passed to the fused kernels")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--rounds", type=int, default=1, help="repeat the whole alternation (fresh graphs)")
    return p.parse_args(argv)


def _check(result):
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA call failed: {error}")
    return result[1:] if len(result) > 2 else (result[1] if len(result) == 2 else None)


class Module:
    """A cubin compiled from source with NVRTC, launched on the current stream."""

    def __init__(self, source, names, extra_options=()):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import compile_cubin
        cubin = compile_cubin(source, torch.cuda.get_device_capability(), extra_options)
        torch.cuda.current_stream()
        self.module = _check(driver.cuModuleLoadData(cubin))
        self.functions = {n: _check(driver.cuModuleGetFunction(self.module, n.encode())) for n in names}

    def set_shared(self, name, nbytes):
        from cuda.bindings import driver
        attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(self.functions[name], attr, int(nbytes)))

    def attrs(self, name):
        from cuda.bindings import driver
        A = driver.CUfunction_attribute
        get = lambda a: int(_check(driver.cuFuncGetAttribute(a, self.functions[name])))  # noqa: E731
        return dict(regs=get(A.CU_FUNC_ATTRIBUTE_NUM_REGS), local_bytes=get(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES))

    def launch(self, name, grid, block, args, shared=0):
        from cuda.bindings import driver
        values, types = [], []
        for a in args:
            if torch.is_tensor(a):
                values.append(a.data_ptr())
                types.append(ctypes.c_void_p)
            elif isinstance(a, float):
                values.append(a)
                types.append(ctypes.c_float)
            else:
                values.append(int(a))
                types.append(ctypes.c_longlong)
        grid = grid if isinstance(grid, tuple) else (grid, 1, 1)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(self.functions[name], *grid, block, 1, 1, shared, stream,
                                     (tuple(values), tuple(types)), 0))


def function_span(source, name):
    start = source.index(f'extern "C" __global__ void {name}(')
    end = source.find('extern "C"', start + 10)
    return start, (len(source) if end < 0 else end)


def stub_source(source):
    """oeq_projected_forward/backward with register-only per-edge weights and no weight-gradient stores."""
    s0, s1 = function_span(source, "oeq_projected_forward")
    body = source[s0:s1]
    n = body.count("weights[radial_weight_index]")
    body = body.replace("weights[radial_weight_index]",
                        "__int_as_float(0x3f800000 | int(radial_weight_index & 7))")
    assert n >= 1
    forward = body
    b0, b1 = function_span(source, "oeq_projected_backward")
    body = source[b0:b1]
    assert body.count("weights[radial_weight_index]") >= 1 and body.count("dweights[radial_weight_index] =") >= 1
    body = body.replace("const scalar_t radial_weight = weights[radial_weight_index];",
                        "const scalar_t radial_weight = __int_as_float(0x3f800000 | int(radial_weight_index & 7));")
    body = body.replace("dweights[radial_weight_index] = scalar_t(weight_gradient);", "dw_sink += weight_gradient;")
    body = body.replace("acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};",
                        "acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};\n    acc_t dw_sink = acc_t(0);", 1)
    last = body.rindex("}")
    body = body[:last] + "    if (dw_sink == acc_t(-1.2345e30f)) dweights[e] = dw_sink;\n}" + body[last + 1:]
    backward = body
    head = source[:min(s0, b0)]
    return head + forward.replace("oeq_projected_forward(", "stub_forward(") + \
        backward.replace("oeq_projected_backward(", "stub_backward(")


def capture(fn, reps, flush):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            flush()
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            flush()
            fn()
    torch.cuda.synchronize()
    return g


def alternate(graphs, reps, blocks):
    times = {k: [] for k in graphs}
    for _ in range(blocks):
        for k, g in graphs.items():
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()
            stop.record()
            torch.cuda.synchronize()
            times[k].append(start.elapsed_time(stop) / reps)
    return times


def rel(got, ref):
    return float((got.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30))


def load(path):
    record = torch.load(path)["records"][0]
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.float()
        out[key] = value
    return out


def make_kernels(p):
    from openequivariance.core.e3nn_lite import TPProblem

    from fastmace.kernels.oeq_streaming import StreamingKernels
    problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                        [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                         for i in p["instructions"]],
                        irrep_normalization="none", path_normalization="none", shared_weights=False,
                        internal_weights=False)
    return StreamingKernels(problem, np.float32, layout=p["layout"], receiver_backward=False)


def run_file(path, args, out_file):
    from fastmace.kernels.oeq_streaming.runtime import streaming_conv_folded_backward
    from fastmace.kernels.triton.skinny_linear import input_grad_into

    r = load(path)
    kernels = make_kernels(r["problem"])
    x, sh, s, rcv, rp, dout = (r[k] for k in ("x", "sh", "senders", "receivers", "row_ptr", "dout"))
    a, W = r["a"].contiguous(), r["weight"].contiguous()
    main, last = W[:, :-1].contiguous(), W[:, -1].contiguous()
    chunk, tile_rule = int(r["chunk_edges"]), r["tile_rule"]
    E, N = sh.shape[0], x.shape[0]
    Q = W.shape[0]
    junk = torch.empty((2 * torch.cuda.get_device_properties(0).L2_cache_size) // 4, device="cuda")

    def flush():
        junk.fill_(1.0)

    w = torch.empty((E, Q), device="cuda")
    msg = torch.empty((N, kernels.output_dim), device="cuda")
    res = {}

    def chain_fwd():
        torch.mm(a, W.t(), out=w)
        kernels.forward(x, sh, w, s, rp, msg)

    def chain_bwd():
        res["chain"] = streaming_conv_folded_backward(x, sh, w, s, rcv, dout, W, main, last, kernels.id, chunk,
                                                      tile_rule)

    chain_fwd()
    chain_bwd()
    torch.cuda.synchronize()
    ref_msg = msg.clone()
    ref_dx, ref_dsh, ref_da = (t.clone() for t in res["chain"])
    row = dict(file=str(path), structure=path.stem.split("-")[-1], nodes=N, edges=E, chunk_edges=chunk,
               tile_rule=tile_rule, chain_msg_vs_record=rel(ref_msg, r["out"]))
    fns = dict(chain_fwd=chain_fwd, chain_bwd=chain_bwd)

    if "stub" in args.variants:
        stub = Module(stub_source(kernels.source), ["stub_forward", "stub_backward"])
        dxs, dshs, dws = torch.zeros_like(x), torch.empty_like(sh), torch.empty((E, Q), device="cuda")
        nb_f = (N * kernels.channels + kernels.num_threads - 1) // kernels.num_threads
        nb_b = (E * kernels.cohort + kernels.num_threads - 1) // kernels.num_threads
        fns["stub_fwd"] = lambda: stub.launch("stub_forward", nb_f, kernels.num_threads,
                                              (N, E, x, sh, w, s, rp, msg))
        fns["stub_bwd"] = lambda: (dxs.zero_(), stub.launch("stub_backward", nb_b, kernels.num_threads,
                                                            (N, E, x, sh, w, s, rcv, dout, dxs, dshs, dws)))
    if "l2" in args.variants:
        buf = torch.empty((chunk, Q), device="cuda")
        da2 = torch.empty_like(ref_da)
        starts = list(range(0, E, chunk))

        def gemm_l2():
            for c0 in starts:
                c1 = min(E, c0 + chunk)
                torch.mm(a[c0:c1], W.t(), out=buf[:c1 - c0])

        def igrad_l2():
            for c0 in starts:
                c1 = min(E, c0 + chunk)
                input_grad_into(buf[:c1 - c0], W, main, last, da2[c0:c1], tile_rule)
        fns["gemm_l2"], fns["igrad_l2"] = gemm_l2, igrad_l2
    if "fused" in args.variants:
        import opt8_fused_kernels as fk
        cfg = dict(kv.split("=") for kv in args.fused_config.split(",") if kv)
        fused = fk.FusedConv(kernels.schedule, kernels.layout, W, cfg)
        out_f = torch.full_like(msg, float("nan"))  # every element must be written
        dx_f, dsh_f, da_f = torch.zeros_like(x), torch.empty_like(sh), torch.empty_like(ref_da)
        fns["fused_fwd"] = lambda: fused.forward(a, x, sh, s, rcv, rp, out_f)
        fns["fused_bwd"] = lambda: fused.backward(a, x, sh, s, rcv, rp, dout, dx_f, dsh_f, da_f)
        fns["fused_fwd"]()
        fns["fused_bwd"]()
        torch.cuda.synchronize()
        row.update(fused_config=dict(fused.describe(), fwd=fused.mod.attrs("fused_forward"),
                                     bwd=fused.mod.attrs("fused_backward")),
                   fused_msg=rel(out_f, ref_msg), fused_dx=rel(dx_f, ref_dx), fused_dsh=rel(dsh_f, ref_dsh),
                   fused_da=rel(da_f, ref_da))
        print("exactness", {k: row[k] for k in row if k.startswith("fused_")}, flush=True)

    for rnd in range(args.rounds):
        graphs = {k: capture(f, args.reps, flush) for k, f in fns.items()}
        graphs["flush"] = capture(lambda: None, args.reps, flush)
        times = alternate(graphs, args.reps, args.blocks)
        base = statistics.median(times.pop("flush"))
        med = {k: round(statistics.median(v) - base, 4) for k, v in times.items()}
        out = dict(row, round=rnd, flush_ms=round(base, 4), ms=med,
                   utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        chain = med["chain_fwd"] + med["chain_bwd"]
        if "stub_fwd" in med and "gemm_l2" in med:
            lb = max(med["stub_fwd"], med["gemm_l2"]) + max(med["stub_bwd"], med["gemm_l2"] + med["igrad_l2"])
            add = med["stub_fwd"] + med["stub_bwd"] + 2 * med["gemm_l2"] + med["igrad_l2"]
            out.update(chain_ms=round(chain, 4), bound_overlap_ms=round(lb, 4), bound_additive_ms=round(add, 4),
                       bound_overlap_ratio=round(lb / chain, 4), bound_additive_ratio=round(add / chain, 4))
        if "fused_fwd" in med:
            fused_total = med["fused_fwd"] + med["fused_bwd"]
            out.update(chain_ms=round(chain, 4), fused_ms=round(fused_total, 4),
                       fused_ratio=round(fused_total / chain, 4))
        print(json.dumps(out), flush=True)
        with open(out_file, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        del graphs


def main(argv=None):
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    for path in args.ops:
        t0 = time.time()
        run_file(path, args, args.output)
        print(f"{path.name}: {time.time() - t0:.0f} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
