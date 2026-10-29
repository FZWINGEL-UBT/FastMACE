"""cuBLAS FP32 emulation (BF16x9) probe: the same SGEMM with and without CUBLAS_EMULATE_SINGLE_PRECISION in this
process's environment (set by the caller); prints the time, the error against float64, a hash of the result and the
kernels cuBLAS ran. TF32 screen, 29 October 2026."""
import hashlib
import json
import os
import time

import torch

torch.backends.cuda.matmul.fp32_precision = "ieee"
g = torch.Generator(device="cuda").manual_seed(1)
out = {"env": os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION"), "gpu": torch.cuda.get_device_name(),
       "cc": torch.cuda.get_device_capability(), "torch": torch.__version__, "cuda": torch.version.cuda}
for n in (4096, 1080):
    k = 4096 if n == 4096 else 512
    a = torch.randn(n, k, generator=g, device="cuda")
    b = torch.randn(k, n if n == 4096 else 512, generator=g, device="cuda")
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    us = (time.perf_counter() - t) / 20 * 1e6
    ref = a.double() @ b.double()
    err = float((c.double() - ref).abs().max() / ref.abs().max())
    out[f"{n}x{k}"] = dict(us=us, tflops=2 * a.shape[0] * k * b.shape[1] / us / 1e6, max_rel=err,
                           sha=hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()[:16])
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    c = a @ b
    torch.cuda.synchronize()
out["kernels"] = sorted({e.name[:80] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})
print(json.dumps(out))
