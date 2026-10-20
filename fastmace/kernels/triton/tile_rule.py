"""The shape rule of FastMACE's own GEMMs (optimisation round 1: OPT-T02 stage B, OPT-T03).

One function decides, from the GEMM's M, N and K and the device's number of SMs only, which backend runs
a product and with which fixed tile configuration. No table of GPUs or shapes is kept and nothing is
measured at run time (SPEC §9.1). The thresholds are in waves, tiles over SMs, so the same rule holds on
36 and on 48 SMs.

- **Backend.** The gathered-row Triton GEMM (``segment_gemm``) runs at cuBLAS speed once its grid is
  large. OPT-T02 B's pre-check on murphy measured 0.98–1.02× of cuBLAS for 9,720–13,608 rows. A
  contiguous Triton GEMM on small M was 38 % slower (OPT-T01 bench). Below ``MIN_TRITON_WAVES`` waves of
  ``BLOCK_M``-row tiles per SM, cuBLAS keeps the product.
- **Tiles.** One configuration, the one the pre-check picked most often: 64 × 64 × 32, 4 warps, 3 stages.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 64, 32, 4, 3
# Triton only where M covers at least this many waves of BLOCK_M-row tiles over the SMs (one column of tiles).
MIN_TRITON_WAVES = 2


@dataclass(frozen=True)
class GemmChoice:
    """How one product runs: ``backend`` "cublas" or "triton", and the Triton tile configuration."""

    backend: str
    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    block_k: int = BLOCK_K
    num_warps: int = NUM_WARPS
    num_stages: int = NUM_STAGES
    reason: str = ""


def row_waves(m: int, sms: int, block_m: int = BLOCK_M) -> float:
    """Waves of ``block_m``-row tiles of one tile column over ``sms`` SMs."""
    return math.ceil(m / block_m) / max(sms, 1)


def segment_gemm_choice(m: int, n: int, k: int, sms: int | None) -> GemmChoice:
    """The backend and tiles of a gathered-row product ``[m, k] @ [k, n]`` on a device with ``sms`` SMs
    (None: not a CUDA device, so cuBLAS/PyTorch)."""
    if not sms:
        return GemmChoice("cublas", reason="not a CUDA device")
    m, n, k, sms = int(m), int(n), int(k), int(sms)
    waves = float(row_waves(m, sms))
    if waves < MIN_TRITON_WAVES:
        return GemmChoice("cublas", reason=f"{m} rows are {waves:.2f} waves of {BLOCK_M}-row tiles on {sms} SMs "
                                           f"(< {MIN_TRITON_WAVES})")
    return GemmChoice("triton", reason=f"{m} rows are {waves:.2f} waves of {BLOCK_M}-row tiles on {sms} SMs")


def device_sms(device) -> int | None:
    """The SM count of a CUDA ``device``, else None (cached per device: forwards ask it while tracing)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return None
    return _sms(device.index if device.index is not None else torch.cuda.current_device())


@functools.lru_cache(maxsize=None)
def _sms(index: int) -> int:
    import torch

    return torch.cuda.get_device_properties(index).multi_processor_count


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "GemmChoice", "MIN_TRITON_WAVES", "NUM_STAGES", "NUM_WARPS",
           "device_sms", "row_waves", "segment_gemm_choice"]
