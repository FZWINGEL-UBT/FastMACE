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
- **Split-K** (``split_k_parts``, record ``split_k``, OPT2-T05, ``docs/design/opt2-T05-split-k.md``): a product
  whose nominal 64 × 64 output tiles do not fill one wave of the SMs, and whose K is not deeper than its N,
  is split along K into a power of two of parts, until it reaches ``SPLIT_K_TARGET_WAVES`` waves, while every
  part keeps at least ``SPLIT_K_MIN_PART`` of K and K divides evenly. A deep K (K > N) stays with cuBLAS, whose
  heuristics split it themselves (opt-T03b: ``[16, 1, 8]`` grids for the K = 2048 inputs); splitting it again
  lost 24-36 % in the micro-benchmark (``tools/split_k_bench.py``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 64, 32, 4, 3
# Triton only where M covers at least this many waves of BLOCK_M-row tiles over the SMs (one column of tiles).
MIN_TRITON_WAVES = 2
# Split-K: the waves a split product aims for, and the smallest K of one part (a real GEMM, not a reduction).
SPLIT_K_TARGET_WAVES = 1  # 2 parts on NaCl's l = 0 products: the micro-benchmark's best (4 and 8 parts gain less)
SPLIT_K_MIN_PART = 128


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


def uses_triton(m, sms) -> bool:
    """Whether ``segment_gemm_choice`` picks Triton for ``m`` rows on ``sms`` SMs, in integer arithmetic only
    (it is evaluated while torch.compile traces a forward, where ``m`` may be symbolic)."""
    if not sms:
        return False
    return (m + BLOCK_M - 1) // BLOCK_M >= MIN_TRITON_WAVES * sms


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


def split_k_parts(m, n, k, sms) -> int:
    """How many parts of K the product ``[m, k] @ [k, n]`` is split into on ``sms`` SMs (1: not split).

    Only where its 64 × 64 output tiles are fewer than one wave and K is at most N; then doubled while the
    tiles times the parts stay below ``SPLIT_K_TARGET_WAVES`` waves, K divides evenly and every part keeps at
    least ``SPLIT_K_MIN_PART``. Integer arithmetic only (the forward evaluates it while torch.compile traces).
    """
    if not sms or k > n:
        return 1
    tiles = ((m + BLOCK_M - 1) // BLOCK_M) * ((n + BLOCK_N - 1) // BLOCK_N)
    if tiles >= sms:
        return 1
    parts = 1
    while (tiles * parts < SPLIT_K_TARGET_WAVES * sms and k % (2 * parts) == 0
           and k // (2 * parts) >= SPLIT_K_MIN_PART):
        parts *= 2
    return int(parts)


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
           "SPLIT_K_MIN_PART", "SPLIT_K_TARGET_WAVES", "device_sms", "row_waves", "segment_gemm_choice",
           "split_k_parts"]
