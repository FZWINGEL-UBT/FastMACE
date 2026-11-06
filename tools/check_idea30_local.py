"""Idea 30's local checks (no H200 needed): ``python tools/check_idea30_local.py RENDER.cu OUTDIR [--run]``.

- Renders ws30 and its falsifiers from RENDER (an H200 lab render of the fused forward, ``lab-forward/forward.cu``)
  into OUTDIR and compiles each with NVRTC for sm_90a (registers, spills: ptxas -v).
- ``--run`` (any CUDA GPU with 99 KB of shared memory per block): the debug render (four consumer warps, an FP32 SIMT
  product reading W from global memory: the same mbarrier/cp.async protocol, chunk walk and TP text) on a synthetic
  CSR graph with empty rows and chunks, against ``lab_idea30.reference_kernel``; its outputs must match.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_idea30 as L  # noqa: E402


def nvrtc_compile(source, arch, want_cubin=False):
    from cuda.bindings import nvrtc
    program = nvrtc.nvrtcCreateProgram(source.encode(), b"idea30.cu", 0, [], [])[1]
    options = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", b"-default-device", b"--ptxas-options=-v"]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    size = nvrtc.nvrtcGetProgramLogSize(program)[1]
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(program, log)
    log = log.decode(errors="replace")
    if int(result[0]) != 0:
        raise RuntimeError(log[-3000:])
    regs = [int(v) for v in re.findall(r"Used (\d+) registers", log)]
    spills = [int(v) for v in re.findall(r"(\d+) bytes spill stores", log)]
    cubin = None
    if want_cubin:
        size = nvrtc.nvrtcGetCUBINSize(program)[1]
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(program, cubin)
    return dict(registers=max(regs) if regs else None, spill=max(spills) if spills else 0, log=log), cubin


def run_debug(render):
    import ctypes

    import numpy as np
    import torch
    from cuda.bindings import driver

    def check(r):
        err = r[0] if isinstance(r, tuple) else r
        if int(err) != 0:
            raise RuntimeError(f"CUDA driver error {err}")
        return r[1] if isinstance(r, tuple) and len(r) == 2 else r

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    head = {k: int(v) for k, v in re.findall(r"constexpr (?:int|int64_t) (\w+) = (\d+);", render)}
    kas, as_, ed, in_dim, q = head["KA"], head["AS"], head["ED"], head["IN_DIM"], head["Q"]
    widths = {}
    for name in ("out0", "out1", "out2", "out3"):
        widths[name] = int(re.search(name + r"\[cur \* (\d+) \+", render).group(1))
    rng = np.random.default_rng(30)
    n = 3001
    deg = rng.integers(0, 140, n)
    deg[rng.random(n) < 0.15] = 0  # empty rows, some whole chunks empty
    deg[100:110] = 0
    row_ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int32)
    e = int(row_ptr[-1])
    receivers = np.repeat(np.arange(n), deg).astype(np.int32)
    senders = rng.integers(0, n, e).astype(np.int32)
    dev = "cuda"
    a = torch.tensor(rng.standard_normal((e, as_)), dtype=torch.float32, device=dev)
    w_full = torch.tensor(rng.standard_normal((q, kas)) * 0.2, dtype=torch.float32, device=dev)
    x = torch.tensor(rng.standard_normal((n, in_dim)), dtype=torch.float32, device=dev)
    sh = torch.tensor(rng.standard_normal((e, ed)), dtype=torch.float32, device=dev)
    snd, rcv, rp = (torch.tensor(v, device=dev) for v in (senders, receivers, row_ptr))
    channels = 128
    outs = {k: torch.zeros((n, w), dtype=torch.float32, device=dev) for k, w in widths.items()}
    refs = {k: torch.zeros((n, w), dtype=torch.float32, device=dev) for k, w in widths.items()}

    def load(source, name):
        info, cubin = nvrtc_compile(source, arch, want_cubin=True)
        module = check(driver.cuModuleLoadData(cubin))
        return check(driver.cuModuleGetFunction(module, name)), info

    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    p = ctypes.c_void_p
    ref, _ = load(L.reference_kernel(render), b"ref_part1")
    values = (n, e, a.data_ptr(), w_full.data_ptr(), x.data_ptr(), sh.data_ptr(), snd.data_ptr(), rcv.data_ptr(),
              rp.data_ptr(), *[refs[k].data_ptr() for k in sorted(refs)])
    types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * (len(values) - 2)
    check(driver.cuLaunchKernel(ref, n, channels // 32, 1, 32, 1, 1, 0, stream, (values, types), 0))
    debug = L.render_ws(render, consumers=4, product="simt", w_source="global")
    fn, info = load(debug, b"fused_folded_forward")
    shared = L.shared_bytes(L.extract(render), consumers=4, w_source="global")
    attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    check(driver.cuFuncSetAttribute(fn, attribute, shared))
    for repeat in range(3):
        for o in outs.values():
            o.fill_(float("nan"))
        counters = torch.zeros(8, dtype=torch.int32, device=dev)
        wout = torch.zeros(1, dtype=torch.float32, device=dev)
        values = (n, e, a.data_ptr(), w_full.data_ptr(), x.data_ptr(), sh.data_ptr(), snd.data_ptr(), rcv.data_ptr(),
                  rp.data_ptr(), counters.data_ptr(), *[outs[k].data_ptr() for k in sorted(outs)], wout.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * (len(values) - 2)
        check(driver.cuLaunchKernel(fn, 33, channels // 32, 2, 32 * 8, 1, 1, shared, stream, (values, types), 0))
        torch.cuda.synchronize()
        for k in sorted(outs):
            # part 1's columns only (every row: rows without edges must be written as zeros); the rest stay NaN here
            mask = torch.zeros_like(refs[k], dtype=torch.bool)
            for line in L.extract(render)["flush_lines"]:
                name, offset = re.match(r"(out\d)\[cur \* \d+ \+ (\d+) \+ ch\]", line).groups()
                if name == k:
                    mask[:, int(offset):int(offset) + channels] = True
            d = (outs[k][mask] - refs[k][mask]).abs().max().item() if mask.any() else 0.0
            nan_part1 = torch.isnan(outs[k][mask]).sum().item()
            scale = refs[k].abs().max().item() or 1.0
            print(f"repeat {repeat} {k}: part-1 entries {int(mask.sum())}, max|d|/max|ref| {d / scale:.2e}, "
                  f"NaN {nan_part1}, bitwise {bool(torch.equal(outs[k][mask], refs[k][mask]))}", flush=True)
    print(f"debug render: {info['registers']} registers, {info['spill']} B spill; {shared} B shared; E = {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("render", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    render = args.render.read_text()
    args.outdir.mkdir(parents=True, exist_ok=True)
    info = L.extract(render)
    print(f"part 1: {info['npl']} paths, {info['nacc']} accumulators, {info['nx']} x rows; "
          f"shared {L.shared_bytes(info)} B")
    for name, options in {"ws30": {}, "ws30-n192": dict(halves=1), "F-ws30-no-tp": dict(tp=False), "F-ws30-no-product": dict(product="none"),
                          "debug-simt": dict(consumers=4, product="simt", w_source="global")}.items():
        source = L.render_ws(render, **options)
        (args.outdir / f"{name}.cu").write_text(source, newline="\n")
        try:
            result, _ = nvrtc_compile(source, "sm_90a")
            print(f"{name:20s} sm_90a: {result['registers']} registers, {result['spill']} B spill stores", flush=True)
        except RuntimeError as error:
            print(f"{name:20s} sm_90a: FAILED\n{error}", flush=True)
    (args.outdir / "reference.cu").write_text(L.reference_kernel(render), newline="\n")
    if args.run:
        run_debug(render)


if __name__ == "__main__":
    main()
