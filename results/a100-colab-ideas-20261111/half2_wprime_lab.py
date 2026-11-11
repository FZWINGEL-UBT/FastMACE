"""Idea 1, stage 3 (11 Nov 2026): GPU lab of the two-far-edges-per-half2 w' channel walk.

Times the census harness kernels (``half2_wprime_census.py``: today's w' channel loop, unchanged in ``fp32``,
two far edges per ``__half2`` in ``half2``, and ``half2_lb3`` with __launch_bounds__(256, 3), the fp32 kernel's
80 registers) on far-edge-like data: 131,072 edges (about the far edges of SrTiO3 10x10x10 at r_h 5.0 A), 5,000
senders, the real table geometry (731 slabs x 1,280 weights x 4 terms). Falsifier: half2 per edge not faster than
fp32 per edge, or numbers wrong (the edge and radial gradients and dx of half2 against fp32 on the same inputs,
max|d|/max|a| per quantity; float16 rounding of 1e-3 order is expected, more is wrong).

Usage: python half2_wprime_lab.py --output DIR [--edges 131072] [--reps 50]
"""
import argparse
import ctypes
import glob
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import half2_wprime_census as hc  # noqa: E402


def includes():
    for d in (*glob.glob(f"{sys.prefix}/lib/python3*/site-packages/nvidia/cu13/include"),
              *glob.glob(f"{sys.prefix}/lib/python3*/site-packages/nvidia/cuda_runtime/include"),
              "/usr/local/cuda/include"):
        if os.path.isdir(d) and os.path.exists(f"{d}/cuda_fp16.h"):
            return [f"-I{d}", "-w"]
    raise SystemExit("cuda_fp16.h not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--edges", type=int, default=131072)
    p.add_argument("--nodes", type=int, default=5000)
    p.add_argument("--reps", type=int, default=50)
    args = p.parse_args()
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

    args.output.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    body = hc.body(hc.SOURCE.read_text())
    sources = {"fp32": hc.FP32.replace("@BODY@", body), "half2": hc.HALF2.replace("@BODY@", hc.to_half2(body))}
    sources["half2_lb3"] = sources["half2"].replace("__launch_bounds__(256)", "__launch_bounds__(256, 3)")
    torch.cuda.current_stream()
    fns, regs = {}, {}
    for name, src in sources.items():
        (args.output / f"{name}.cu").write_text(src, newline="\n")
        module = _check(driver.cuModuleLoadData(compile_cubin(src, cap, includes())))
        fns[name] = (module, _check(driver.cuModuleGetFunction(module, b"census_wprime")))
        regs[name] = int(_check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fns[name][1])))

    g = torch.Generator(device=dev).manual_seed(11)
    n, pairs, nodes, slabs_n = args.edges, args.edges // 2, args.nodes, 731

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, device=dev, generator=g) * scale

    x = rnd(nodes, 512, scale=0.5)
    table = rnd(slabs_n, 1280, 4, scale=0.05)            # power-basis coefficients in t, |t| <= 1
    senders = torch.randint(0, nodes, (n,), device=dev, generator=g, dtype=torch.int32)
    slabs = torch.randint(0, slabs_n, (n,), device=dev, generator=g, dtype=torch.int32)
    ts = torch.rand(n, device=dev, generator=g) * 2 - 1
    scs = torch.rand(n, device=dev, generator=g) + 0.5
    flags = torch.full((n,), 3, device=dev, dtype=torch.int32)
    sh = rnd(n, 16, scale=0.5)
    dout = rnd(5120, scale=0.02)
    xh, th = x.half(), table.half()
    sh2 = torch.stack((sh[0::2], sh[1::2]), dim=-1).half().contiguous()   # [pairs, 16, 2]
    dout2 = torch.stack((dout, dout), dim=-1).half().contiguous()        # one receiver: broadcast
    dx = torch.zeros(nodes * 512, device=dev)
    outs = {k: torch.zeros(n * 32 * 17, device=dev) for k in sources}
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    P, I = ctypes.c_void_p, ctypes.c_int

    def launch(name, store=0):
        half = name != "fp32"
        units = pairs if half else n
        vals = ((xh if half else x).data_ptr(), dx.data_ptr(), (th if half else table).data_ptr(), senders.data_ptr(),
                slabs.data_ptr(), ts.data_ptr(), scs.data_ptr(), flags.data_ptr(), (sh2 if half else sh).data_ptr(),
                (dout2 if half else dout).data_ptr(), outs[name].data_ptr(), units, store)
        _check(driver.cuLaunchKernel(fns[name][1], (units + 7) // 8, 1, 1, 256, 1, 1, 0, stream,
                                     (vals, (P,) * 11 + (I, I)), 0))

    # numbers: same inputs (fp32 sees the float16-rounded x, table, sh and dout, so only the arithmetic differs)
    x_keep, table_keep, sh_keep, dout_keep = x.clone(), table.clone(), sh.clone(), dout.clone()
    x.copy_(xh.float()); table.copy_(th.float()); sh.copy_(sh.half().float()); dout.copy_(dout.half().float())
    check = {}
    dxs = {}
    for name in sources:
        dx.zero_()
        launch(name, 1)
        torch.cuda.synchronize()
        dxs[name] = dx.clone()
    ref = outs["fp32"].view(n, 32, 17)
    for name in ("half2", "half2_lb3"):
        o = outs[name].view(n, 32, 17)
        eg, rg = (o[..., :16] - ref[..., :16]), (o[..., 16] - ref[..., 16])
        check[name] = dict(edge_gradient=float(eg.abs().max() / ref[..., :16].abs().max()),
                           radial_gradient=float(rg.abs().max() / ref[..., 16].abs().max()),
                           dx=float((dxs[name] - dxs["fp32"]).abs().max() / dxs["fp32"].abs().max()),
                           finite=bool(torch.isfinite(o).all()))
    x.copy_(x_keep); table.copy_(table_keep); sh.copy_(sh_keep); dout.copy_(dout_keep)

    times = {k: [] for k in sources}
    for name in sources:
        for _ in range(5):
            launch(name)
    torch.cuda.synchronize()
    for _ in range(args.reps):  # interleaved, one launch per sample
        for name in sources:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            launch(name)
            b.record()
            torch.cuda.synchronize()
            times[name].append(a.elapsed_time(b) * 1e3)
    med = {k: statistics.median(v) for k, v in times.items()}
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, edges=n, pairs=pairs, reps=args.reps,
                  registers=regs, median_us=med, us_per_edge={k: med[k] / n for k in med},
                  half2_over_fp32=med["half2"] / med["fp32"], half2_lb3_over_fp32=med["half2_lb3"] / med["fp32"],
                  check=check)
    (args.output / "half2_wprime_lab.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
