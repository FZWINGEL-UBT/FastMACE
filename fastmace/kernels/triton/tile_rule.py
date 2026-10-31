"""The shape rule of FastMACE's own GEMMs (optimisation round 1: OPT-T02 stage B, OPT-T03).

One function decides, from the GEMM's M, N and K and the device's number of SMs only, which backend runs
a product and with which fixed tile configuration; a second, from A's bytes and the device's L2 size, the
order of its tiles. No table of GPUs or shapes is kept and nothing is measured at run time (SPEC §9.1).
The thresholds are in waves, tiles over SMs, so the same rule holds on 36 and on 48 SMs.

- **Backend.** The gathered-row Triton GEMM (``segment_gemm``) runs at cuBLAS speed once its grid is
  large. OPT-T02 B's pre-check on murphy measured 0.98–1.02× of cuBLAS for 9,720–13,608 rows. A
  contiguous Triton GEMM on small M was 38 % slower (OPT-T01 bench). Below ``MIN_TRITON_WAVES`` waves of
  ``BLOCK_M``-row tiles per SM, cuBLAS keeps the product.
- **Tiles.** One configuration, the one the pre-check picked most often: 64 × 64 × 32, 4 warps, 3 stages.
- **Tile order** (``columns_together``). Launched row tile first, the column tiles of one row tile run a
  whole pass apart, so an A larger than L2 is read from DRAM once per column tile. Where A does not fit in L2
  the column tiles of a row tile run next to each other instead, and A is read about once. The same tiles
  and the same K order, so the results are bitwise equal. MH-1's ``linear_2`` products at NbCl5 9,216
  atoms on the RTX 5060 Ti (32 MiB L2): they read A 8.5 times and ran at 91 % of memory throughput; side
  by side, 0.80–0.84 times the time for A of 57–132 MB, but 1.06 times for 19 MB, which fits
  (docs/validation/segment-gemm-order.md). A default only on ``COLUMN_ORDER_GPUS`` (decision 66).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 64, 32, 4, 3
# Triton only where M covers at least this many waves of BLOCK_M-row tiles over the SMs (one column of tiles).
MIN_TRITON_WAVES = 2
# The Triton products' tile rows and warps (the waves rule above keeps BLOCK_M). 0: by GPU, 128 rows with 8 warps
# on LARGE_TILE_GPUS, else BLOCK_M and NUM_WARPS; a positive value is an A/B side's override on every GPU.
TRITON_BLOCK_M = 0
TRITON_NUM_WARPS = 0
# Decision 66 (owner, 29 October 2026): 128-row tiles with 8 warps (bitwise equal) only where same-process whole-step
# A/Bs gained in both build orders (docs/validation/ab-constants-audit-20261030.md, "tiles128"). RTX 4000 Ada: NbCl5
# 9,216 atoms 0.987 / 0.986 (column order), 1,944 atoms 0.989 / 0.989, NaNbO3 0.995 / 0.994. The RTX 5060 Ti lost
# at NbCl5 1,944 atoms (1.020 / 1.007), the A100-SXM4-40GB at 9,216 atoms (1.021 / 1.021).
LARGE_TILE_GPUS = ("NVIDIA RTX 4000 Ada Generation",)
# The tile order: "auto" (columns_together's L2 rule), or "rows" / "columns" for an A/B side.
TILE_ORDER = "auto"
# Decision 66 (owner, 29 October 2026): the column order is a default only where same-process whole-step A/Bs in
# both build orders measured a speed-up, by exact device name (docs/validation/segment-gemm-order.md). RTX 5060 Ti:
# NbCl5 9,216 atoms 0.940 / 0.938, NaNbO3 8,640 atoms 0.959 / 0.955. RTX 4000 Ada: see the report. The
# A100-SXM4-80GB measured no gain (NbCl5 9,216 atoms 1.0017 / 1.0015; the products 1.003 times the time).
COLUMN_ORDER_GPUS = ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")


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
    column_fastest: bool = False


def row_waves(m: int, sms: int, block_m: int = BLOCK_M) -> float:
    """Waves of ``block_m``-row tiles of one tile column over ``sms`` SMs."""
    return math.ceil(m / block_m) / max(sms, 1)


def uses_triton(m, sms) -> bool:
    """Whether ``segment_gemm_choice`` picks Triton for ``m`` rows on ``sms`` SMs, in integer arithmetic only
    (it is evaluated while torch.compile traces a forward, where ``m`` may be symbolic)."""
    if not sms:
        return False
    return (m + BLOCK_M - 1) // BLOCK_M >= MIN_TRITON_WAVES * sms


def segment_gemm_choice(m: int, n: int, k: int, sms: int | None, gpu: str | None = None) -> GemmChoice:
    """The backend and tiles of a gathered-row product ``[m, k] @ [k, n]`` on a device with ``sms`` SMs
    (None: not a CUDA device, so cuBLAS/PyTorch) named ``gpu`` (the tile shape, ``LARGE_TILE_GPUS``)."""
    if not sms:
        return GemmChoice("cublas", reason="not a CUDA device")
    m, n, k, sms = int(m), int(n), int(k), int(sms)
    waves = float(row_waves(m, sms))
    if waves < MIN_TRITON_WAVES:
        return GemmChoice("cublas", reason=f"{m} rows are {waves:.2f} waves of {BLOCK_M}-row tiles on {sms} SMs "
                                           f"(< {MIN_TRITON_WAVES})")
    large = gpu in LARGE_TILE_GPUS
    block_m = TRITON_BLOCK_M if TRITON_BLOCK_M > 0 else (128 if large else BLOCK_M)
    num_warps = TRITON_NUM_WARPS if TRITON_NUM_WARPS > 0 else (8 if large else NUM_WARPS)
    return GemmChoice("triton", block_m=block_m, num_warps=num_warps,
                      reason=f"{m} rows are {waves:.2f} waves of {BLOCK_M}-row tiles on {sms} SMs")


def columns_together(m: int, k: int, element_bytes: int, l2_bytes: int | None, gpu: str | None = None) -> bool:
    """Whether the column tiles of a row tile run next to each other: on a GPU named in ``COLUMN_ORDER_GPUS``, A
    (``m * k`` elements) is larger than the device's L2. Without an L2 size, or on another GPU, the row-first order."""
    if TILE_ORDER != "auto":
        return TILE_ORDER == "columns"
    return gpu in COLUMN_ORDER_GPUS and bool(l2_bytes) and int(m) * int(k) * int(element_bytes) > int(l2_bytes)


def device_name(device) -> str | None:
    """The name of a CUDA ``device``, else None (cached per device)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return None
    return _name(device.index if device.index is not None else torch.cuda.current_device())


@functools.lru_cache(maxsize=None)
def _name(index: int) -> str:
    import torch

    return torch.cuda.get_device_name(index)


def device_l2_bytes(device) -> int | None:
    """The L2 size in bytes of a CUDA ``device``, else None (cached per device)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return None
    return _l2(device.index if device.index is not None else torch.cuda.current_device())


@functools.lru_cache(maxsize=None)
def _l2(index: int) -> int:
    import torch

    return int(torch.cuda.get_device_properties(index).L2_cache_size)


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


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "COLUMN_ORDER_GPUS", "GemmChoice", "LARGE_TILE_GPUS", "MIN_TRITON_WAVES",
           "NUM_STAGES", "NUM_WARPS", "TILE_ORDER", "columns_together", "device_l2_bytes", "device_name", "device_sms",
           "row_waves", "segment_gemm_choice"]
