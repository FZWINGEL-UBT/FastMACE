"""Gathered-row segment GEMM against cuBLAS + the copies it would remove (plan opt-1, OPT-T02 stage B pre-check).

Stage B (docs/design/opt-T02-BC-go-no-go.md) would replace each per-l product of a block linear and the
copy that feeds or follows it by one Triton GEMM that reads its A rows straight from the wide ir_mul
tensor, ``A[(n, m), k] = X[n, offset + m * mul + k]`` (with an optional per-atom scale, the density
division), and writes its C rows straight into a strided destination, ``C[(n, m), j] -> Y[n, offset' + m *
width + j]``. This bench times, on NbCl5 3x3x3's shapes (1,944 atoms, MH-1), float32:

- ``linear2_forward``: the gate's rows ``[N (2l+1), 512] @ W_l`` written into the ``[N, 16, 512]`` input of
  the symmetric contraction. Baseline: cuBLAS into contiguous rows, then the join (one copy of all l).
- ``linear2_input_grad``: the contraction's gradient ``[N, 16, 512]``, sliced per l, ``@ W_l^T``. Baseline:
  a contiguous copy of each slice, then cuBLAS.
- ``residual_forward``: segments of ``message [N, W]`` divided per atom by the density, ``@ W_l``. Baseline:
  the fused divide-and-copy, then cuBLAS.

and reports, per case, the segment GEMM's time against cuBLAS + copy and against cuBLAS alone. The Triton
kernel uses the fastest of a fixed list of tile configurations per shape: a screen's upper bound. The stop
rule (go/no-go note): B goes on only if the segment GEMM beats cuBLAS + copy. Errors are checked against a
float64 product. Output: <output>/summary.json. Use an idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ATOMS = 1944
# (l, components, channels) of MH-1's hidden irreps 512x0e + 512x1o + 512x2e + 512x3o (ir_mul, [N, 16, 512]).
HIDDEN = ((0, 1, 512), (1, 3, 512), (2, 5, 512), (3, 7, 512))
# Residual inputs (segments of the [N, W] message after the convolution): l, components, channels, N out.
RESIDUAL = ((0, 1, 384, 2048), (2, 5, 384, 512), (3, 7, 256, 512))
CONFIGS = ((64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3), (32, 64, 32, 4, 3))


def triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, s_ptr, rows, N, K, D, a_n, a_m, a_k, b_k, b_n, c_n, c_m, c_j,
             SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Row r of A and C is (atom r // D, component r % D): gathered loads, strided stores.
        pm, pn = tl.program_id(0), tl.program_id(1)
        r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        atom, comp = r // D, r % D
        a_row = atom * a_n + comp * a_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_ptr + a_row[:, None] + k[None, :] * a_k, mask=(r < rows)[:, None] & (k < K)[None, :],
                        other=0.0)
            b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision="ieee")
        if SCALE:
            acc = acc * tl.load(s_ptr + atom, mask=r < rows, other=1.0)[:, None]
        tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :] * c_j, acc,
                 mask=(r < rows)[:, None] & (j < N)[None, :])

    return gemm


def time_us(torch, fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1e3)
    return statistics.median(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atoms", type=int, default=ATOMS)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args(argv)
    import torch
    import triton

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time  # noqa: E402

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    args.output.mkdir(parents=True, exist_ok=False)
    kernel = triton_kernel()
    g = torch.Generator(device="cuda").manual_seed(20261020)
    n = args.atoms
    dummy = torch.ones(1, device="cuda")

    def run(a, a_strides, base_a, b, c, c_strides, base_c, rows, d, K, N, config, scale=None):
        bm, bn, bk, warps, stages = config
        grid = (triton.cdiv(rows, bm), triton.cdiv(N, bn))
        kernel[grid](a.view(-1)[base_a:], b, c.view(-1)[base_c:], dummy if scale is None else scale, rows, N, K, d,
                     *a_strides, b.stride(0), b.stride(1), *c_strides, SCALE=scale is not None,
                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)

    def best(fn):
        results = []
        for config in CONFIGS:
            try:
                fn(config)
            except Exception:  # noqa: BLE001  (a configuration the compiler refuses)
                continue
            results.append((time_us(torch, lambda c=config: fn(c), args.repeats), config))
        return min(results)

    def rel(got, ref):
        return float((got.double() - ref).abs().max() / ref.abs().max())

    cases = {}
    width = sum(dim * mul for _, dim, mul in HIDDEN)  # 8192
    weights = {deg: torch.randn(mul, mul, generator=g, device="cuda") / mul ** 0.5 for deg, _, mul in HIDDEN}
    weights_t = {deg: w.t().contiguous() for deg, w in weights.items()}
    # linear_2 forward: rows -> [N, 16, 512]
    rows_in = {deg: torch.randn(n * dim, mul, generator=g, device="cuda") for deg, dim, mul in HIDDEN}
    out_buf = torch.empty(n, width, device="cuda")
    entry = {}
    offsets, offset = {}, 0
    for deg, dim, mul in HIDDEN:
        offsets[deg] = offset
        offset += dim * mul
    outs = {deg: torch.empty(n * dim, mul, device="cuda") for deg, dim, mul in HIDDEN}

    def base_forward():
        for deg, dim, mul in HIDDEN:
            torch.mm(rows_in[deg], weights[deg], out=outs[deg])
        torch.cat([outs[deg].view(n, dim * mul) for deg, dim, mul in HIDDEN], dim=1, out=out_buf)

    def gemms_only():
        for deg, dim, mul in HIDDEN:
            torch.mm(rows_in[deg], weights[deg], out=outs[deg])

    base_us, gemm_us = time_us(torch, base_forward, args.repeats), time_us(torch, gemms_only, args.repeats)
    seg_us, seg_cfg = 0.0, {}
    for deg, dim, mul in HIDDEN:
        def f(config, deg=deg, dim=dim, mul=mul):
            run(rows_in[deg], (dim * mul, mul, 1), 0, weights[deg], out_buf, (width, mul, 1), offsets[deg], n * dim, dim,
                mul, mul, config)
        us, cfg = best(f)
        seg_us += us
        seg_cfg[deg] = list(cfg)
        f(cfg)
    torch.cuda.synchronize()
    ref = torch.cat([(rows_in[deg].double() @ weights[deg].double()).view(n, dim * mul) for deg, dim, mul in HIDDEN], 1)
    entry.update(cublas_plus_copy_us=base_us, cublas_only_us=gemm_us, segment_us=seg_us, configs=seg_cfg,
                 max_rel_error=rel(out_buf, ref))
    cases["linear2_forward"] = entry

    # linear_2 input gradient: [N, 16, 512] slices -> rows
    grad = torch.randn(n, width, generator=g, device="cuda")
    grad_rows = {deg: torch.empty(n * dim, mul, device="cuda") for deg, dim, mul in HIDDEN}

    def base_grad():
        for deg, dim, mul in HIDDEN:
            left = grad[:, offsets[deg]:offsets[deg] + dim * mul].reshape(n * dim, mul)  # a copy (not contiguous)
            torch.mm(left, weights_t[deg], out=grad_rows[deg])

    contiguous = {deg: grad[:, offsets[deg]:offsets[deg] + dim * mul].reshape(n * dim, mul).contiguous()
                  for deg, dim, mul in HIDDEN}

    def grad_gemms_only():
        for deg, dim, mul in HIDDEN:
            torch.mm(contiguous[deg], weights_t[deg], out=grad_rows[deg])

    base_us, gemm_us = time_us(torch, base_grad, args.repeats), time_us(torch, grad_gemms_only, args.repeats)
    seg_us, seg_cfg, errors = 0.0, {}, []
    for deg, dim, mul in HIDDEN:
        def f(config, deg=deg, dim=dim, mul=mul):
            run(grad, (width, mul, 1), offsets[deg], weights_t[deg], grad_rows[deg], (dim * mul, mul, 1), 0, n * dim, dim,
                mul, mul, config)
        us, cfg = best(f)
        seg_us += us
        seg_cfg[deg] = list(cfg)
        f(cfg)
        torch.cuda.synchronize()
        errors.append(rel(grad_rows[deg], contiguous[deg].double() @ weights_t[deg].double()))
    cases["linear2_input_grad"] = dict(cublas_plus_copy_us=base_us, cublas_only_us=gemm_us, segment_us=seg_us,
                                       configs=seg_cfg, max_rel_error=max(errors))

    # residual forward: message segments / density -> rows
    msg_width = sum(dim * mul for _, dim, mul, _ in RESIDUAL)
    message = torch.randn(n, msg_width, generator=g, device="cuda")
    density = torch.rand(n, generator=g, device="cuda") + 1.0
    inverse = (1.0 / density).contiguous()
    res_w = {deg: torch.randn(mul, out, generator=g, device="cuda") / mul ** 0.5 for deg, _, mul, out in RESIDUAL}
    res_out = {deg: torch.empty(n * dim, out, device="cuda") for deg, dim, _, out in RESIDUAL}
    res_off, offset = {}, 0
    for deg, dim, mul, _ in RESIDUAL:
        res_off[deg] = offset
        offset += dim * mul

    def base_res():
        for deg, dim, mul, _ in RESIDUAL:
            left = (message[:, res_off[deg]:res_off[deg] + dim * mul] / density[:, None]).reshape(n * dim, mul)
            torch.mm(left, res_w[deg], out=res_out[deg])

    lefts = {deg: (message[:, res_off[deg]:res_off[deg] + dim * mul] / density[:, None]).reshape(n * dim, mul)
             for deg, dim, mul, _ in RESIDUAL}

    def res_gemms_only():
        for deg, dim, mul, _ in RESIDUAL:
            torch.mm(lefts[deg], res_w[deg], out=res_out[deg])

    base_us, gemm_us = time_us(torch, base_res, args.repeats), time_us(torch, res_gemms_only, args.repeats)
    seg_us, seg_cfg, errors = 0.0, {}, []
    for deg, dim, mul, out in RESIDUAL:
        def f(config, deg=deg, dim=dim, mul=mul, out=out):
            # The scale goes after the product: (x / s) W = (x W) / s per row of one atom (rounding differs).
            run(message, (msg_width, mul, 1), res_off[deg], res_w[deg], res_out[deg], (dim * out, out, 1), 0, n * dim, dim,
                mul, out, config, scale=inverse)
        us, cfg = best(f)
        seg_us += us
        seg_cfg[deg] = list(cfg)
        f(cfg)
        torch.cuda.synchronize()
        errors.append(rel(res_out[deg], lefts[deg].double() @ res_w[deg].double()))
    cases["residual_forward"] = dict(cublas_plus_copy_us=base_us, cublas_only_us=gemm_us, segment_us=seg_us,
                                     configs=seg_cfg, max_rel_error=max(errors))

    for name, c in cases.items():
        c["segment_over_cublas_plus_copy"] = c["segment_us"] / c["cublas_plus_copy_us"]
        c["segment_over_cublas_only"] = c["segment_us"] / c["cublas_only_us"]
        c["copy_us"] = c["cublas_plus_copy_us"] - c["cublas_only_us"]
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "configs"},
              flush=True)
    summary = dict(tool="segment_gemm_bench", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   triton=triton.__version__, fastmace_commit=ab_step_time.fastmace_commit(), atoms=n,
                   load=list(os.getloadavg()), cases=cases,
                   go=all(c["segment_over_cublas_plus_copy"] < 1.0 for c in cases.values()))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("go" if summary["go"] else "stop", flush=True)


if __name__ == "__main__":
    main()
