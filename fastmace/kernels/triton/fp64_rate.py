"""The GPU's float64 : float32 multiply-add rate, measured (f64-T01, owner decision 63).

``EngineConfig.fp64_emulation``'s rule needs to know whether a GPU has full-rate float64 units (data-centre parts:
1/2 of float32) or reduced-rate ones (consumer and workstation parts: 1/64). The compute capability does not tell:
sm_80, sm_90 and sm_100 are full rate, but so is no x.0 consumer part after sm_120 (12.0, the RTX 50 series). So the
rate is measured: a chain of dependent-free fused multiply-adds per element, once in float32 and once in float64, in
Triton (no cuBLAS: the measurement must not create a cuBLAS handle before ``configure_fp64_emulation``). About
50 ms on first use per device, cached for the process.
"""

from __future__ import annotations

from functools import cache

STEPS = 256
ELEMENTS = 1 << 22
BLOCK = 256


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fma_chain(a_ptr, out_ptr, n, STEPS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        a = tl.load(a_ptr + i, mask=mask, other=1.0)
        y0 = a
        y1 = a + 1.0
        y2 = a + 2.0
        y3 = a + 3.0
        for _ in tl.static_range(STEPS):
            y0 = tl.fma(y0, a, 0.5)
            y1 = tl.fma(y1, a, 0.25)
            y2 = tl.fma(y2, a, 0.125)
            y3 = tl.fma(y3, a, 0.0625)
        tl.store(out_ptr + i, y0 + y1 + y2 + y3, mask=mask)

    return fma_chain


def _seconds(dtype, device):
    import torch

    kernel = _kernel()
    a = torch.full((ELEMENTS,), 0.5, dtype=dtype, device=device)
    out = torch.empty_like(a)
    grid = ((ELEMENTS + BLOCK - 1) // BLOCK,)
    kernel[grid](a, out, ELEMENTS, STEPS=STEPS, BLOCK=BLOCK)  # compile and warm
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        kernel[grid](a, out, ELEMENTS, STEPS=STEPS, BLOCK=BLOCK)
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop) / 1e3)
    return best


@cache
def _ratio(device_index):
    import torch

    device = torch.device("cuda", device_index)
    return _seconds(torch.float32, device) / _seconds(torch.float64, device)


def fp64_rate_ratio(device) -> float:
    """float64 : float32 fused-multiply-add throughput of CUDA ``device`` (about 0.5 on A100/H100, about 1/64 on
    consumer and workstation GPUs)."""
    import torch

    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _ratio(index)


__all__ = ["fp64_rate_ratio"]
