"""The POLAR k-space kernels in CUDA C++ (``kspace_megakernel.cu``), the backend of ``kspace_megakernel.CUDA =
True`` (polar-megakernel, 7 November 2026).

Built offline into one cubin per compute capability (``tools/build_kspace_megakernel.sh``; the build record is
``kspace_megakernel.build.txt``) and loaded with the CUDA driver API, like ``fastmace.kernels.cutlass.tf32x3``.
Each entry is a persistent kernel with grid barriers: its grid is the device's resident blocks (the occupancy of
that entry times the SMs), the same for every launch of the entry, with one arrival counter per entry and device
(never reset: each barrier waits for the next multiple of the grid). ``field_forward`` / ``field_backward`` give
the same results as ``kspace_megakernel.kspace_field_forward`` / ``_backward`` (rounding); ``structure_factor``
gives ``fourier.structure_factor``'s ``(Yc, Ys)`` with its own backward. Shapes the cubin has no entry for, other
dtypes and devices: ``supports`` is False and the callers take their other path.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import torch

HERE = Path(__file__).parent
CUBINS = {(8, 0): HERE / "kspace_megakernel_sm80.cubin", (8, 9): HERE / "kspace_megakernel_sm89.cubin",
          (9, 0): HERE / "kspace_megakernel_sm90.cubin", (12, 0): HERE / "kspace_megakernel_sm120.cubin"}
FIELDS = {(1, 4, 8), (2, 4, 8)}  # (C, J, F) with entries
SF_WIDTHS = {1, 2, 4, 8, 16}
S0 = 2.0 / (2.0 * math.pi) ** 3
MAX_BLOCKS_PER_SM = 4
ITEMS_PER_BLOCK = 2  # work items per resident block (each sum is split until it has this many)
_KERNELS: dict = {}  # device index -> _Kernels


def cubin_for(capability):
    """The cubin for a compute capability: SASS runs on its own major version at the same or a higher minor one, so
    sm_86/87 (RTX A4000, A10, Orin) load the sm_80 cubin and sm_121 the sm_120 one. None without a cubin."""
    major, minor = capability
    fits = [c for c in CUBINS if c[0] == major and c[1] <= minor]
    return CUBINS[max(fits)] if fits else None


class _Kernels:
    def __init__(self, index):
        import numpy as np
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        self.index = index
        with torch.cuda.device(index):
            torch.zeros(1, device=f"cuda:{index}")
            self.context = _check(driver.cuCtxGetCurrent())
            cubin = cubin_for(torch.cuda.get_device_capability(index))
            self.module = _check(driver.cuModuleLoadData(cubin.read_bytes()))
            pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_kspace_info"))
            info = np.zeros(size // 4, dtype=np.int32)
            _check(driver.cuMemcpyDtoH(info, pointer, size))
        self.threads, self.rows, self.warps = (int(v) for v in info[:3])
        self.sms = torch.cuda.get_device_properties(index).multi_processor_count
        self.entries = {}  # name -> (function, grid, counter)

    def entry(self, name):
        if name not in self.entries:
            from cuda.bindings import driver

            from fastmace.kernels.oeq_streaming.runtime import _check
            function = _check(driver.cuModuleGetFunction(self.module, name.encode()))
            per_sm = _check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, self.threads, 0))
            if per_sm < 1:
                raise RuntimeError(f"{name}: no block fits on an SM")
            grid = self.sms * min(int(per_sm), MAX_BLOCKS_PER_SM)
            counter = torch.zeros(1, dtype=torch.int64, device=f"cuda:{self.index}")
            self.entries[name] = (function, grid, counter)
        return self.entries[name]

    def grid(self, name):
        return self.entry(name)[1]

    def counter(self, name):
        return self.entry(name)[2]


def _kernels(index):
    if index not in _KERNELS:
        _KERNELS[index] = _Kernels(index)
    return _KERNELS[index]


def _index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def available(device) -> bool:
    """Whether the cubin loads on ``device`` (CUDA, a cubin for its capability, the file present)."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = _index(device)
    cubin = cubin_for(tuple(torch.cuda.get_device_capability(index)))
    if cubin is None or not cubin.is_file():
        return False
    try:
        _kernels(index)
    except (ImportError, RuntimeError):
        return False
    return True


def supports_field(*tensors, C, J, F) -> bool:
    return ((C, J, F) in FIELDS and all(t.is_cuda and t.dtype == torch.float32 for t in tensors)
            and available(tensors[0].device))


def supports_sf(*tensors, width) -> bool:
    return (width in SF_WIDTHS and all(t.is_cuda and t.dtype == torch.float32 for t in tensors)
            and available(tensors[0].device))


def _splits(tiles, length, grid, warps):
    """(splits, chunk) of ``length`` so that ``tiles * splits`` is at least ``ITEMS_PER_BLOCK`` items per block,
    with at least ``warps`` entries per chunk; a function of the shapes and the grid only (deterministic)."""
    splits = max(1, -(-ITEMS_PER_BLOCK * grid // max(tiles, 1)))
    splits = min(splits, max(1, -(-length // warps)))
    chunk = -(-length // splits)
    return -(-length // chunk), chunk


_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def _launch(kernels, name, pointers, ints, s0, device):
    values = tuple(t.data_ptr() if isinstance(t, torch.Tensor) else t for t in pointers)
    values += (kernels.counter(name).data_ptr(),) + tuple(ints)
    types = (_P,) * (len(pointers) + 1) + (_I,) * len(ints)
    if s0 is not None:
        values += (s0,)
        types += (_F,)
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    function, grid, _ = kernels.entry(name)
    current = _check(driver.cuCtxGetCurrent())
    if int(current) != int(kernels.context):
        _check(driver.cuCtxSetCurrent(kernels.context))
    stream = driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    _check(driver.cuLaunchKernel(function, grid, 1, 1, kernels.threads, 1, 1, 0, stream, (values, types), 0))


def field_forward(k, r, mask, x, dr, di, w, pr, pi):
    """``(out [N, C, F], yc, ys [K, C, J], a, b [K, C, F])``; ``supports_field`` holds and K, N > 0."""
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]
    kernels = _kernels(_index(k.device))
    name = f"fm_kspace_field_forward_{C}_{J}_{F}"
    grid = kernels.grid(name)
    s1, c1 = _splits(-(-K // kernels.rows), N, grid, kernels.warps)
    s2, c2 = _splits(-(-N // kernels.rows), K, grid, kernels.warps)
    out = r.new_empty((N, C, F))
    yc, ys = k.new_empty((K, C, J)), k.new_empty((K, C, J))
    a, b = k.new_empty((K, C, F)), k.new_empty((K, C, F))
    part1 = k.new_empty((s1, K, 2 * C * J))
    part2 = k.new_empty((s2, N, C * F))
    args = [t.contiguous() for t in (k, r, mask, x, dr, di, w, pr, pi)]
    _launch(kernels, name, args + [out, yc, ys, a, b, part1, part2], (K, N, s1, c1, s2, c2), S0, k.device)
    return out, yc, ys, a, b


def field_backward(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi, factors, grad_k):
    """``(dk, dr_pos, dx, d_dr, d_di, dw, dpr, dpi)`` as ``kspace_field_backward``; K, N > 0."""
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]
    kernels = _kernels(_index(k.device))
    name = f"fm_kspace_field_backward_{C}_{J}_{F}_{int(factors)}{int(grad_k)}"
    grid = kernels.grid(name)
    s1, c1 = _splits(-(-K // kernels.rows), N, grid, kernels.warps)
    tiles = -(-N // kernels.rows)
    s2, c2 = _splits(tiles, K, grid, kernels.warps)
    dyc, dys = k.new_empty((K, C, J)), k.new_empty((K, C, J))
    part1 = k.new_empty((s1, K, 2 * C * F))
    part2 = k.new_empty((s2, N, C * J + 3))
    gk_part = k.new_empty((tiles, K, 3) if grad_k else (1,))
    dk = k.new_empty((K, 3) if grad_k else (0,))
    gr, dx = r.new_empty((N, 3)), r.new_empty((N, C, J))
    if factors:
        ddr, ddi, dpr, dpi, dw = k.new_empty((K, J)), k.new_empty((K, J)), k.new_empty((K, F)), \
            k.new_empty((K, F)), k.new_empty((K,))
    else:
        ddr, ddi, dpr, dpi, dw = (k.new_empty((1,)) for _ in range(5))
    args = [t.contiguous() for t in (k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi)]
    _launch(kernels, name, args + [dyc, dys, part1, part2, gk_part, dk if grad_k else gk_part, gr, dx, ddr, ddi, dw,
                                   dpr, dpi], (K, N, s1, c1, s2, c2), S0, k.device)
    if factors:
        return dk, gr, dx, ddr, ddi, dw, dpr, dpi
    return (dk, gr, dx, *(k.new_empty((0,)) for _ in range(5)))


# --------------------------------------------------------------------------- the structure factor (Coulomb energy)


def _dense_sf(k, r, mask, x):
    theta = k @ r.t()
    return mask[:, None] * (torch.cos(theta) @ x), mask[:, None] * (torch.sin(theta) @ x)


def _dense_sf_backward(k, r, x, dyc, dys):
    theta = k @ r.t()
    cos, sin = torch.cos(theta), torch.sin(theta)
    dx = cos.t() @ dyc + sin.t() @ dys
    dtheta = cos * (dys @ x.t()) - sin * (dyc @ x.t())
    return dtheta @ r, dtheta.t() @ k, dx


@torch.library.custom_op("fastmace::kspace_sf_forward", mutates_args=())
def sf_forward(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor,
               x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(Yc, Ys)`` [K, W]: ``Yc = mask * cos(k r^T) x`` (and sin)."""
    K, N, W = k.shape[0], r.shape[0], x.shape[1]
    if not supports_sf(k, r, mask, x, width=W) or K == 0 or N == 0:
        return tuple(t.contiguous() for t in _dense_sf(k, r, mask, x))
    kernels = _kernels(_index(k.device))
    name = f"fm_kspace_sf_forward_{W}"
    s1, c1 = _splits(-(-K // kernels.rows), N, kernels.grid(name), kernels.warps)
    yc, ys = k.new_empty((K, W)), k.new_empty((K, W))
    part1 = k.new_empty((s1, K, 2 * W))
    _launch(kernels, name, [t.contiguous() for t in (k, r, mask, x)] + [yc, ys, part1], (K, N, s1, c1), None,
            k.device)
    return yc, ys


@sf_forward.register_fake
def _(k, r, mask, x):
    return k.new_empty((k.shape[0], x.shape[1])), k.new_empty((k.shape[0], x.shape[1]))


@torch.library.custom_op("fastmace::kspace_sf_backward", mutates_args=())
def sf_backward(k: torch.Tensor, r: torch.Tensor, x: torch.Tensor, dyc: torch.Tensor, dys: torch.Tensor,
                grad_k: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dk, dr, dx)`` for the masked output gradients ``dyc, dys`` [K, W]; ``dk`` empty without ``grad_k``."""
    K, N, W = k.shape[0], r.shape[0], x.shape[1]
    if not supports_sf(k, r, x, dyc, dys, width=W) or K == 0 or N == 0:
        dk, dr, dx = _dense_sf_backward(k, r, x, dyc, dys)
        return (dk.contiguous() if grad_k else k.new_empty((0,))), dr.contiguous(), dx.contiguous()
    kernels = _kernels(_index(k.device))
    name = f"fm_kspace_sf_backward_{W}_{int(grad_k)}"
    tiles = -(-N // kernels.rows)
    s2, c2 = _splits(tiles, K, kernels.grid(name), kernels.warps)
    dx, gr = r.new_empty((N, W)), r.new_empty((N, 3))
    dk = k.new_empty((K, 3) if grad_k else (0,))
    part2 = k.new_empty((s2, N, W + 3))
    gk_part = k.new_empty((tiles, K, 3) if grad_k else (1,))
    _launch(kernels, name, [t.contiguous() for t in (k, r, x, dyc, dys)] + [dx, gr, dk if grad_k else gk_part,
                                                                            part2, gk_part],
            (K, N, s2, c2), None, k.device)
    return dk, gr, dx


@sf_backward.register_fake
def _(k, r, x, dyc, dys, grad_k):
    return k.new_empty((k.shape[0], 3) if grad_k else (0,)), r.new_empty((r.shape[0], 3)), r.new_empty(x.shape)


class _StructureFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, r, mask, x):
        ctx.save_for_backward(k, r, mask, x)
        return sf_forward(k, r, mask, x)

    @staticmethod
    def backward(ctx, grad_cos, grad_sin):
        k, r, mask, x = ctx.saved_tensors
        need = ctx.needs_input_grad
        grad_cos = torch.zeros((k.shape[0], x.shape[1]), dtype=x.dtype, device=x.device) if grad_cos is None \
            else grad_cos
        grad_sin = torch.zeros_like(grad_cos) if grad_sin is None else grad_sin
        dk, dr, dx = sf_backward(k, r, x, (mask[:, None] * grad_cos).contiguous(),
                                 (mask[:, None] * grad_sin).contiguous(), bool(need[0]))
        return (dk if need[0] else None, dr if need[1] else None, None, dx if need[3] else None)


def structure_factor(k, r, mask, x):
    """``(Yc, Ys)`` [K, W] of ``x`` [N, W]: ``fourier.structure_factor``'s values, differentiable in ``k``, ``r``
    and ``x``."""
    return _StructureFactor.apply(k, r, mask, x)


__all__ = ["available", "field_backward", "field_forward", "structure_factor", "supports_field", "supports_sf"]
