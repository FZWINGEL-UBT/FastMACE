"""Dense GEMM rates on this GPU by format (arXiv 2609.37693 follow-up): FP64, FP32 (TF32 off), TF32, FP16, BF16,
INT8 (torch._int_mm), FP8 E4M3 and FP4 E2M1 (torch._scaled_mm), on a large square shape and on MH-1's shapes.
Each format in try/except (not every one has a path here). One JSON line: TFLOPS (TOPS for INT8) per shape."""
import json
import time

import torch

SHAPES = [(8192, 8192, 8192), (13608, 512, 512), (1944, 2048, 384), (56320, 64, 1280)]


def timed(fn, flops):
    fn()
    torch.cuda.synchronize()
    reps = max(3, min(200, int(2e12 / flops)))
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return flops * reps / (time.perf_counter() - t) / 1e12


def run(m, k, n):
    out, f = {}, 2.0 * m * k * n
    a32, b32 = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
    for name, fn in (
            ("fp64", lambda: a32.double() @ b32.double()),
    ):
        a64, b64 = a32.double(), b32.double()
        out["fp64"] = timed(lambda: a64 @ b64, f)
    torch.backends.cuda.matmul.allow_tf32 = False
    out["fp32"] = timed(lambda: a32 @ b32, f)
    torch.backends.cuda.matmul.allow_tf32 = True
    out["tf32"] = timed(lambda: a32 @ b32, f)
    torch.backends.cuda.matmul.allow_tf32 = False
    a16, b16 = a32.half(), b32.half()
    out["fp16"] = timed(lambda: a16 @ b16, f)
    ab, bb = a32.bfloat16(), b32.bfloat16()
    out["bf16"] = timed(lambda: ab @ bb, f)
    try:
        ai = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        bi = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
        out["int8"] = timed(lambda: torch._int_mm(ai, bi), f)
    except Exception as e:
        out["int8"] = repr(e)[:120]
    try:
        a8 = a32.to(torch.float8_e4m3fn)
        b8 = b32.t().contiguous().to(torch.float8_e4m3fn).t()  # column-major B
        one = torch.ones((), device="cuda")
        out["fp8_e4m3"] = timed(lambda: torch._scaled_mm(a8, b8, one, one, out_dtype=torch.bfloat16), f)
    except Exception as e:
        out["fp8_e4m3"] = repr(e)[:160]
    try:  # NVFP4: E2M1 packed two per byte, E4M3 scales per 16 elements along K
        a4 = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
        b4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2).t()
        sa = torch.ones(m * (k // 16), device="cuda").to(torch.float8_e4m3fn)
        sb = torch.ones(n * (k // 16), device="cuda").to(torch.float8_e4m3fn)
        out["fp4_e2m1"] = timed(lambda: torch._scaled_mm(a4, b4, sa, sb, out_dtype=torch.bfloat16), f)
    except Exception as e:
        out["fp4_e2m1"] = repr(e)[:200]
    return out


print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "rates": {f"{m}x{k}x{n}": run(m, k, n) for m, k, n in SHAPES}}), flush=True)
