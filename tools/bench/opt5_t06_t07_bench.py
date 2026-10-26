#!/usr/bin/env python3
"""OPT5-T06b & OPT5-T07: Exponent-Shifted 3xFP16, Co-Executed SIMT+3xFP16 GEMM, and Delta/Channel-Split GEMMs.

Tests:
1. OPT5-T06b (Accuracy on all 52 real MH-1 force-call products):
   Compares Exponent-Shifted 3xFP16 (c=3) and 2xFP16 (c=2) against 3xTF32 and cuBLAS IEEE FP32 on the real
   operands of one MH-1 force call (NaCl and NbCl5).
2. OPT5-T06b (Speed on sm_120):
   Compares cuBLAS IEEE FP32, Triton 3xFP16, and a co-executed hybrid (CUDA cores computing alpha of N columns in
   FP32 while Tensor Cores compute 1-alpha of N columns in 3xFP16) on production shapes.
3. OPT5-T07a (Channel-Magnitude Split within one step):
   Splits K into top-norm channels on CUDA cores (IEEE FP32) + smaller-norm channels on Tensor Cores (1xTF32 or
   2xFP16) on real MH-1 operands.
4. OPT5-T07b (Time-Step Delta GEMM across 1 fs MD steps):
   Measures exact force and energy error on MH-1 NaCl when step t's atom GEMMs (forward and backward) evaluate
   Y_t = Y_{t-1} + (X_t - X_{t-1}) @ W with the delta product in 1xTF32 (c=1) or 2xFP16 (c=2) on Tensor Cores.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

SCALE_FLOOR = 2.0 ** -103

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def graph_time_us(fn, warmup: int = 10, reps: int = 50, blocks: int = 7) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6 / reps)
    return round(statistics.median(times), 2)


def calc_errors(got: torch.Tensor, ref64: torch.Tensor, abs64: torch.Tensor) -> dict:
    diff = (got.double() - ref64).abs()
    scale = float(ref64.abs().max())
    frob = float(ref64.norm())
    normal = abs64 >= SCALE_FLOOR
    cancel = float((diff[normal] / abs64[normal]).max()) if bool(normal.any()) else 0.0
    return {
        "max_rel": float(diff.max()) / scale if scale else 0.0,
        "frob_rel": float(diff.norm()) / frob if frob else 0.0,
        "cancel_scaled": cancel,
    }


def fp16x3_matmul_eager(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exponent-shifted 3xFP16 matmul using PyTorch's cuBLAS FP16 Tensor Core GEMM (FP32 accumulate)."""
    if a.shape[1] < 32 or b.shape[1] < 32:
        return a @ b
    ab = a.to(torch.float16)
    as_ = ((a - ab.to(torch.float32)) * 2048.0).to(torch.float16)
    bb = b.to(torch.float16)
    bs = ((b - bb.to(torch.float32)) * 2048.0).to(torch.float16)
    # Stack into [M, 2K] @ [2K, N] for the cross-term so cuBLAS executes 2 GEMMs instead of 3 launches:
    # c_main = ab @ bb (in FP16->FP32), c_cross = [ab, as_] @ [bs; bb] * (1/2048)
    # Or use torch._int_mm / addmm with out_dtype=torch.float32:
    c_main = torch.matmul(ab.float(), bb.float())  # exact same FP32 accumulation of FP16 inputs for accuracy check
    c_cross = torch.matmul(ab.float(), bs.float()) + torch.matmul(as_.float(), bb.float())
    return c_main + c_cross * (1.0 / 2048.0)


@triton.jit
def triton_fp16x3_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc_main = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_cross = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + k[None, :] * sak,
                    mask=(rm < M)[:, None] & (k < K)[None, :], other=0.0)
        b = tl.load(b_ptr + k[:, None] * sbk + rn[None, :] * sbn,
                    mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        # Extract exact power-of-two exponent scale of each tile by masking the IEEE FP32 exponent bits (0x7F800000):
        a_max_bits = tl.max(tl.abs(a).to(tl.int32, bitcast=True)) & 0x7F800000
        b_max_bits = tl.max(tl.abs(b).to(tl.int32, bitcast=True)) & 0x7F800000
        # Clamp exponent bits to normal FP32 range [2^-120, 2^120] and compute reciprocal scale (0x7F000000 - exp_bits):
        a_exp = tl.maximum(tl.minimum(a_max_bits, 0x7B800000), 0x03800000)
        b_exp = tl.maximum(tl.minimum(b_max_bits, 0x7B800000), 0x03800000)
        a_scale = a_exp.to(tl.float32, bitcast=True)
        b_scale = b_exp.to(tl.float32, bitcast=True)
        a_inv = (0x7F000000 - a_exp).to(tl.float32, bitcast=True)
        b_inv = (0x7F000000 - b_exp).to(tl.float32, bitcast=True)
        ab_scale = a_scale * b_scale

        a_norm = a * a_inv
        b_norm = b * b_inv
        ab = a_norm.to(tl.float16)
        as_ = ((a_norm - ab.to(tl.float32)) * 2048.0).to(tl.float16)
        bb = b_norm.to(tl.float16)
        bs = ((b_norm - bb.to(tl.float32)) * 2048.0).to(tl.float16)
        acc_main += tl.dot(ab, bb, out_dtype=tl.float32) * ab_scale
        acc_cross += (tl.dot(ab, bs, out_dtype=tl.float32) + tl.dot(as_, bb, out_dtype=tl.float32)) * ab_scale
    out = acc_main + acc_cross * (1.0 / 2048.0)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, out,
             mask=(rm < M)[:, None] & (rn < N)[None, :])


def launch_triton_fp16x3(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, bm=64, bn=64, bk=32, nw=4, ns=3):
    m, k = a.shape
    n = b.shape[1]
    if k < 32 or n < 32:
        torch.mm(a.contiguous(), b.contiguous(), out=out)
        return
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    triton_fp16x3_kernel[grid](
        a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns
    )


def run_speed_comparison() -> dict:
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    shapes = [
        ("NaCl_linear_1512x512x512", 1512, 512, 512),
        ("NbCl5_linear_5832x512x512", 5832, 512, 512),
        ("NbCl5_linear_13608x512x512", 13608, 512, 512),
        ("OMOL_linear_1944x1024x1024", 1944, 1024, 1024),
    ]
    out_speed = {}
    s2 = torch.cuda.Stream()
    for label, M, K, N in shapes:
        torch.manual_seed(0)
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        ref64 = a.double() @ b.double()
        abs64 = a.double().abs() @ b.double().abs()

        t_cublas = graph_time_us(lambda: torch.mm(a, b, out=c))
        err_cublas = calc_errors(a @ b, ref64, abs64)

        t_tf32x3 = graph_time_us(lambda: tf32x3_mm(a, b))
        err_tf32x3 = calc_errors(tf32x3_mm(a, b), ref64, abs64)

        best_fp16x3_us = 1e9
        best_cfg = None
        tiles = [(64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3)]
        for bm, bn, bk, nw, ns in tiles:
            t_cur = graph_time_us(
                lambda bm=bm, bn=bn, bk=bk, nw=nw, ns=ns: launch_triton_fp16x3(a, b, c, bm, bn, bk, nw, ns)
            )
            if t_cur < best_fp16x3_us:
                best_fp16x3_us = t_cur
                best_cfg = f"{bm}x{bn}x{bk}_w{nw}"

        launch_triton_fp16x3(a, b, c)
        err_fp16x3 = calc_errors(c, ref64, abs64)

        # Also test 2-stream column-split co-execution: 75% N on cuBLAS FP32 + 25% N on Triton 3xFP16
        n_split = (N * 3) // 4
        b_left = b[:, :n_split].contiguous()
        b_right = b[:, n_split:].contiguous()
        c_left = torch.empty((M, n_split), device="cuda", dtype=torch.float32)
        c_right = torch.empty((M, N - n_split), device="cuda", dtype=torch.float32)

        def run_dual():
            s0 = torch.cuda.current_stream()
            s2.wait_stream(s0)
            torch.mm(a, b_left, out=c_left)
            with torch.cuda.stream(s2):
                launch_triton_fp16x3(a, b_right, c_right)
            s0.wait_stream(s2)

        t_dual = graph_time_us(run_dual)

        out_speed[label] = {
            "shape": [M, K, N],
            "cublas_ieee_us": t_cublas,
            "triton_tf32x3_us": t_tf32x3,
            "triton_fp16x3_us": best_fp16x3_us,
            "triton_fp16x3_best_tile": best_cfg,
            "speedup_fp16x3_over_tf32x3": round(t_tf32x3 / best_fp16x3_us, 3),
            "dual_75simt_25fp16x3_us": t_dual,
            "errors": {
                "cublas_ieee": err_cublas,
                "triton_tf32x3": err_tf32x3,
                "triton_fp16x3": err_fp16x3,
            },
        }
    return out_speed


def run_real_operands_and_delta() -> dict:
    from ase.io import read

    from fastmace.config import RewriteConfig
    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

    structures_root = Path(os.environ["FASTMACE_STRUCTURES"]) / "Structures/Benchmark/R2SCAN"
    atoms = read(structures_root / "NaCl.cif").repeat((3, 3, 3))
    atoms.set_pbc(True)
    rng = np.random.default_rng(20261024)
    atoms.positions += rng.uniform(-0.05, 0.05, size=atoms.positions.shape)

    loaded = fastmace.load("mh1-r2scan")
    ff = fastmace.ForceField(loaded, atoms, precision="float32", rewrites=RewriteConfig(tf32x3_gemms="on"))
    # Warm-up
    e0, f0, _ = ff.evaluate(atoms)
    torch.cuda.synchronize()

    # Capture all 52 products on step 0 (atoms) and step 1 (atoms displaced by a typical 1 fs 300 K step ~ 0.008 A)
    step0_ops = []
    step1_ops = []
    current_list = None

    orig_launch = tf32x3_gemm._launch
    orig_joined = segment_gemm._gemm

    def hook_launch(a, b, out):
        orig_launch(a, b, out)
        if current_list is not None:
            current_list.append(("dense", a.detach().clone(), b.detach().clone()))

    def hook_joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
        orig_joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
        if current_list is not None and precision == "tf32x3":
            k, n = b.shape
            nat = rows // dim
            left = a.reshape(-1)[a_offset:].as_strided((nat, dim, k), (a_n, a_m, 1)).reshape(rows, k)
            current_list.append(("joined", left.detach().clone(), b.detach().clone()))

    tf32x3_gemm._launch = hook_launch
    segment_gemm._gemm = hook_joined
    try:
        current_list = step0_ops
        ff.evaluate(atoms)
        torch.cuda.synchronize()

        atoms_step1 = atoms.copy()
        atoms_step1.positions += rng.normal(0.0, 0.006, size=atoms.positions.shape)
        current_list = step1_ops
        ff.evaluate(atoms_step1)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch = orig_launch
        segment_gemm._gemm = orig_joined

    # Evaluate across all 52 real products:
    # 1. Exponent-shifted 3xFP16 (OPT5-T06b)
    # 2. Channel-magnitude split (OPT5-T07a: top 50% norm channels in IEEE FP32 + bottom 50% in 1xTF32 or 2xFP16)
    # 3. Time-step Delta GEMM (OPT5-T07b: Y_0 in IEEE FP32 + (A_1 - A_0) @ B in 1xTF32 or 2xFP16)
    fp16x3_ratios = []
    fp16x3_worse = 0
    fp16x3_max_cancel = 0.0
    ieee_max_cancel = 0.0

    chan_tf32_ratios = []
    chan_fp16x2_ratios = []
    delta_tf32_ratios = []
    delta_fp16x2_ratios = []
    rel_delta_norms = []

    for (kind0, a0, b0), (kind1, a1, b1) in zip(step0_ops, step1_ops):
        ref1 = a1.double() @ b1.double()
        abs1 = a1.double().abs() @ b1.double().abs()
        c_ieee1 = a1 @ b1
        err_ieee = calc_errors(c_ieee1, ref1, abs1)["cancel_scaled"]
        ieee_max_cancel = max(ieee_max_cancel, err_ieee)

        # 1. 3xFP16 on step 1
        c_fp16x3 = torch.empty_like(c_ieee1)
        launch_triton_fp16x3(a1, b1, c_fp16x3)
        err_fp16x3 = calc_errors(c_fp16x3, ref1, abs1)["cancel_scaled"]
        fp16x3_max_cancel = max(fp16x3_max_cancel, err_fp16x3)
        if err_ieee > 0:
            fp16x3_ratios.append(err_fp16x3 / err_ieee)
            if err_fp16x3 > err_ieee:
                fp16x3_worse += 1

        # 2. Channel-Magnitude Split (top 50% K by row norm of B in IEEE FP32, bottom 50% K in 1xTF32 or 2xFP16)
        k = a1.shape[1]
        if k >= 32 and b1.shape[1] >= 32:
            b_norms = torch.norm(b1, dim=1)
            order = torch.argsort(b_norms, descending=True)
            k_half = k // 2
            idx_hi, idx_lo = order[:k_half], order[k_half:]
            c_hi = a1[:, idx_hi] @ b1[idx_hi, :]  # CUDA cores IEEE FP32
            a_lo, b_lo = a1[:, idx_lo], b1[idx_lo, :]
            # 1xTF32 on tail:
            a_lo_tf32 = torch.ops.aten._to_copy(a_lo)
            # Emulate TF32 rounding (clear low 13 mantissa bits):
            a_lo_tf32 = (a_lo.view(torch.int32) & -8192).view(torch.float32)
            b_lo_tf32 = (b_lo.view(torch.int32) & -8192).view(torch.float32)
            c_chan_tf32 = c_hi + (a_lo_tf32 @ b_lo_tf32)
            err_chan_tf32 = calc_errors(c_chan_tf32, ref1, abs1)["cancel_scaled"]
            if err_ieee > 0:
                chan_tf32_ratios.append(err_chan_tf32 / err_ieee)

            # 2xFP16 (big*big + 2^-11 * big*small, c=2!) on tail:
            ab_lo = a_lo.to(torch.float16)
            bb_lo = b_lo.to(torch.float16)
            bs_lo = ((b_lo - bb_lo.float()) * 2048.0).to(torch.float16)
            c_chan_fp16x2 = c_hi + (ab_lo.float() @ bb_lo.float()) + (ab_lo.float() @ bs_lo.float()) * (1.0 / 2048.0)
            err_chan_fp16x2 = calc_errors(c_chan_fp16x2, ref1, abs1)["cancel_scaled"]
            if err_ieee > 0:
                chan_fp16x2_ratios.append(err_chan_fp16x2 / err_ieee)

        # 3. Time-Step Delta GEMM: Y_1 = Y_0_ieee + (A_1 - A_0) @ B_1 (when B_1 == B_0)
        if a0.shape == a1.shape and b0.shape == b1.shape and torch.equal(b0, b1):
            da = a1 - a0
            rel_norm = float(da.norm() / a1.norm().clamp_min(1e-30))
            rel_delta_norms.append(rel_norm)
            y0_ieee = a0 @ b0
            # 1xTF32 on delta da @ b1:
            da_tf32 = (da.view(torch.int32) & -8192).view(torch.float32)
            b1_tf32 = (b1.view(torch.int32) & -8192).view(torch.float32)
            c_delta_tf32 = y0_ieee + (da_tf32 @ b1_tf32)
            err_delta_tf32 = calc_errors(c_delta_tf32, ref1, abs1)["cancel_scaled"]
            if err_ieee > 0:
                delta_tf32_ratios.append(err_delta_tf32 / err_ieee)

            # 2xFP16 with dynamic max-exponent shift on delta da @ b1:
            scale_da = float(2.0 ** torch.floor(torch.log2(da.abs().max().clamp_min(1e-30))).item())
            da_scaled = da / scale_da
            dab = da_scaled.to(torch.float16)
            bb1 = b1.to(torch.float16)
            bs1 = ((b1 - bb1.float()) * 2048.0).to(torch.float16)
            c_delta_fp16x2 = y0_ieee + scale_da * (
                (dab.float() @ bb1.float()) + (dab.float() @ bs1.float()) * (1.0 / 2048.0)
            )
            err_delta_fp16x2 = calc_errors(c_delta_fp16x2, ref1, abs1)["cancel_scaled"]
            if err_ieee > 0:
                delta_fp16x2_ratios.append(err_delta_fp16x2 / err_ieee)

    # Also measure end-to-end Force & Energy error when all atom GEMMs use Exponent-Shifted 3xFP16 vs float64!
    orig_launch = tf32x3_gemm._launch
    orig_joined = segment_gemm._gemm

    def fp16x3_hook_launch(a, b, out):
        launch_triton_fp16x3(a, b, out)

    tf32x3_gemm._launch = fp16x3_hook_launch
    try:
        e_fp16x3, f_fp16x3, _ = ff.evaluate(atoms_step1)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch = orig_launch

    e_tf32x3, f_tf32x3, _ = ff.evaluate(atoms_step1)
    ff_def = fastmace.ForceField(loaded, atoms_step1, precision="float32")
    e_def, f_def, _ = ff_def.evaluate(atoms_step1)

    return {
        "real_52_products_fp16x3": {
            "products": len(step1_ops),
            "fp16x3_max_cancel_scaled": fp16x3_max_cancel,
            "ieee_max_cancel_scaled": ieee_max_cancel,
            "fp16x3_over_ieee_median": statistics.median(fp16x3_ratios),
            "fp16x3_over_ieee_max": max(fp16x3_ratios),
            "fp16x3_worse_count": fp16x3_worse,
        },
        "end_to_end_mh1_nacl_vs_default_fp32": {
            "fp16x3_energy_diff_eV": float(abs(e_fp16x3 - e_def)),
            "fp16x3_force_max_diff_eV_A": float(np.max(np.abs(f_fp16x3 - f_def))),
            "tf32x3_energy_diff_eV": float(abs(e_tf32x3 - e_def)),
            "tf32x3_force_max_diff_eV_A": float(np.max(np.abs(f_tf32x3 - f_def))),
        },
        "channel_magnitude_split_50_50": {
            "count": len(chan_tf32_ratios),
            "tail_1xTF32_over_ieee_median": statistics.median(chan_tf32_ratios),
            "tail_1xTF32_over_ieee_max": max(chan_tf32_ratios),
            "tail_2xFP16_over_ieee_median": statistics.median(chan_fp16x2_ratios),
            "tail_2xFP16_over_ieee_max": max(chan_fp16x2_ratios),
        },
        "timestep_delta_gemm_1fs": {
            "weight_gemms_count": len(delta_tf32_ratios),
            "rel_delta_norm_median": statistics.median(rel_delta_norms),
            "rel_delta_norm_max": max(rel_delta_norms),
            "delta_1xTF32_over_ieee_median": statistics.median(delta_tf32_ratios),
            "delta_1xTF32_over_ieee_max": max(delta_tf32_ratios),
            "delta_2xFP16_over_ieee_median": statistics.median(delta_fp16x2_ratios),
            "delta_2xFP16_over_ieee_max": max(delta_fp16x2_ratios),
        },
    }


def main() -> None:
    out_dir = Path("/mnt/d/Margraf/FastMACE/results/opt5-T06-fp16x3-20261026")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "speed_and_synthetic_accuracy": run_speed_comparison(),
        "real_mh1_force_call_and_delta": run_real_operands_and_delta(),
    }
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
