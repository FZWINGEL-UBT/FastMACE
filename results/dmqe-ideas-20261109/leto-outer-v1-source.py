"""Receiver-owned CUDA lab for the first interaction's tabled outer sums.

This is an isolated lab.  It does not change FastMACE production dispatch.  The
baseline calls the current ``tabled_outer_sum_forward`` and
``tabled_outer_sum_backward`` Triton custom ops.  The candidate is a small
hand-written CUDA implementation compiled with NVRTC:

* one CTA owns one receiver and all sender-species slots;
* the forward accumulates ``S[receiver, species, lm, k]`` without atomics or a
  global zero fill;
* the backward reuses a slot's ``G[lm,k]`` for its contiguous edge run and
  computes ``grad_y`` and ``dr`` with warp-shuffle butterfly reductions.

The lab uses the real OMAT-0 dimensions (D=16, K=64, 3 slots, degree-5
tables), jittered SrTiO3 edge ordering, and random values with the production
table layout.  It reports NVRTC register/spill data and a PTX instruction
census when a CUDA environment is available.

Usage (hybrid environment, checkout on PYTHONPATH)::

    python tools/dmqe_outer_lab.py --output results/dmqe-outer-local --rounds 31

The tool is intentionally self-contained so that a workstation run cannot
alter any production module constants or source files.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CUDA_SOURCE = r"""
extern "C" __device__ __forceinline__ float cheb_value(
    const float* __restrict__ coef, long base, int terms, int columns, int k, float x) {
  float v1 = 0.0f, v2 = 0.0f;
  for (int t = terms - 1; t >= 1; --t) {
    float c = coef[base + (long)t * columns + k];
    float v = fmaf(2.0f * x, v1, c - v2);
    v2 = v1; v1 = v;
  }
  return fmaf(x, v1, coef[base + k] - v2);
}

extern "C" __device__ __forceinline__ float cheb_scalar(
    const float* __restrict__ coef, long base, int terms, int columns, int k, float x) {
  return cheb_value(coef, base, terms, columns, k, x);
}

extern "C" __device__ __forceinline__ float edge_x(
    float r, const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, int n_bins, float r_lo, float bin_scale, float r_max,
    int* piece_out, bool* inside_out) {
  float clamped = fmaxf(r, r_lo);
  int index = min((int)((clamped - r_lo) * bin_scale), n_bins - 1);
  int piece = bins[index];
  *piece_out = piece;
  *inside_out = r < r_max;
  return (clamped - starts[piece]) * scales[piece] - 1.0f;
}

// Receiver-owned forward.  One CTA writes every value of one receiver's
// [slots,D,K] tile; all output elements therefore have a single writer.
extern "C" __global__ void dmqe_outer_forward(
    const float* __restrict__ r, const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ starts,
    const float* __restrict__ scales, const int* __restrict__ bins,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ out, float* __restrict__ density_edge, float* __restrict__ density_receiver,
    int nodes, int slots, int pieces, int n_bins, int terms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  extern __shared__ float sm[];
  float* acc = sm;                         // slots * D * K
  float* aval = acc + slots * D * K;       // K
  float* yval = aval + K;                  // D
  int receiver = (int)blockIdx.x;
  if (receiver >= nodes) return;
  int tid = (int)threadIdx.x;
  int width = D * K;
  for (int i = tid; i < slots * width; i += blockDim.x) acc[i] = 0.0f;
  if (tid == 0 && density_mode) density_receiver[receiver] = 0.0f;
  __syncthreads();

  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  for (int e = begin; e < end; ++e) {
    int slot = (int)(key[e] % slots);
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long base = ((long)p * pieces + piece) * terms * columns;
    if (tid < K) {
      aval[tid] = inside ? cheb_value(coef, base, terms, columns, tid, x) : 0.0f;
    }
    if (tid < D) yval[tid] = y[(long)e * D + tid];
    if (tid == 0) {
      float den = inside ? cheb_scalar(coef, base, terms, columns, columns - 1, x) : 0.0f;
      if (density_mode) density_receiver[receiver] += den;
      else density_edge[e] = den;
    }
    __syncthreads();
    for (int q = tid; q < width; q += blockDim.x) {
      int d = q / K, k = q - d * K;
      acc[slot * width + q] = fmaf(yval[d], aval[k], acc[slot * width + q]);
    }
    __syncthreads();
  }
  for (int i = tid; i < slots * width; i += blockDim.x) {
    int slot = i / width, q = i - slot * width;
    int d = q / K, k = q - d * K;
    long dst = inner == 1
        ? (long)(receiver * slots + slot) * width + q
        : (long)receiver * slots * width + (long)d * slots * K + (long)slot * K + k;
    out[dst] = acc[i];
  }
}

// Receiver-owned fused backward.  The four warps cover four lm components
// each; every eight-lane subgroup reduces its K=64 dot by shuffle butterfly.
extern "C" __global__ void dmqe_outer_backward(
    const float* __restrict__ grad, const float* __restrict__ r,
    const int* __restrict__ pair, const long long* __restrict__ key,
    const float* __restrict__ coef, const float* __restrict__ dcoef,
    const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, const float* __restrict__ grad_density,
    const float* __restrict__ y, const int* __restrict__ row_ptr,
    float* __restrict__ dr, float* __restrict__ grad_y,
    int nodes, int slots, int pieces, int n_bins, int terms, int dterms, int columns,
    float r_lo, float bin_scale, float r_max, int D, int K, int density_mode, int inner) {
  extern __shared__ float sm[];
  float* g = sm;                       // D*K
  float* aval = g + D*K;               // K
  float* slope = aval + K;             // K
  float* yval = slope + K;             // D
  float* part = yval + D;              // 4 warp partials
  int receiver = (int)blockIdx.x, tid = (int)threadIdx.x;
  if (receiver >= nodes) return;
  int lane = tid & 31, warp = tid >> 5;
  int width = D * K;
  int begin = row_ptr[receiver], end = row_ptr[receiver + 1];
  long long current = -1;
  for (int e = begin; e < end; ++e) {
    long long edge_key = key[e];
    if (edge_key != current) {
      int slot = (int)(edge_key % slots);
      for (int i = tid; i < width; i += blockDim.x) {
        int d = i / K, k = i - d * K;
        long src = inner == 1
            ? (long)(receiver * slots + slot) * width + i
            : (long)receiver * slots * width + (long)d * slots * K + (long)slot * K + k;
        g[i] = grad[src];
      }
      current = edge_key;
    }
    int p = pair[e];
    int piece; bool inside;
    float x = edge_x(r[e], starts, scales, bins, n_bins, r_lo, bin_scale, r_max, &piece, &inside);
    long vbase = ((long)p * pieces + piece) * terms * columns;
    long dbase = ((long)p * pieces + piece) * dterms * columns;
    if (tid < K) {
      aval[tid] = inside ? cheb_value(coef, vbase, terms, columns, tid, x) : 0.0f;
      slope[tid] = inside ? cheb_value(dcoef, dbase, dterms, columns, tid, x) : 0.0f;
    }
    if (tid < D) yval[tid] = y[(long)e * D + tid];
    __syncthreads();

    // Each warp handles four d rows, with four independent 8-lane groups.
    float dr_local = 0.0f;
    for (int group = 0; group < 4; ++group) {
      int d = warp * 4 + group;
      if (d >= D) continue;
      int sub = lane >> 3, sublane = lane & 7;
      float gy = 0.0f, rd = 0.0f;
      for (int k = sublane; k < K; k += 8) {
        float gv = g[d * K + k];
        gy = fmaf(gv, aval[k], gy);
        rd = fmaf(gv * yval[d], slope[k], rd);
      }
      unsigned mask = 0xffu << (sub * 8);
      for (int off = 4; off > 0; off >>= 1) {
        gy += __shfl_down_sync(mask, gy, off, 8);
        rd += __shfl_down_sync(mask, rd, off, 8);
      }
      if (sublane == 0) {
        grad_y[(long)e * D + d] = gy;
        dr_local += rd;
      }
    }
    // One partial per warp, then warp 0 performs the final butterfly.
    if (lane == 0) part[warp] = dr_local;
    __syncthreads();
    if (warp == 0) {
      float v = lane < 4 ? part[lane] : 0.0f;
      unsigned mask = 0xffffffffu;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
      if (lane == 0) {
        float gd = density_mode ? grad_density[receiver] : grad_density[e];
        // The density column is the final table column.  Its derivative is
        // evaluated by lane 0 with the same Chebyshev recurrence.
        float ds = inside ? cheb_scalar(dcoef, dbase, dterms, columns, columns - 1, x) : 0.0f;
        dr[e] = inside ? v + gd * ds : 0.0f;
      }
    }
    __syncthreads();
  }
}
"""


def _check(result, name):
    err = result[0] if isinstance(result, tuple) else result
    if int(err) != 0:
        raise RuntimeError(f"{name}: CUDA driver error {err!r}")
    return result[1] if isinstance(result, tuple) and len(result) == 2 else None


def _compile(source: str, arch: str):
    """Compile source with NVRTC and return loaded functions plus a census."""
    from cuda.bindings import driver, nvrtc

    program = _check(nvrtc.nvrtcCreateProgram(source.encode(), b"dmqe_outer_lab.cu", 0, [], []), "create")
    options = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", b"-default-device",
               b"--ptxas-options=-v"]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    log_size = _check(nvrtc.nvrtcGetProgramLogSize(program), "log size")
    log_buf = b" " * int(log_size)
    _check(nvrtc.nvrtcGetProgramLog(program, log_buf), "log")
    log = log_buf.decode(errors="replace")
    if int(result[0]) != 0:
        raise RuntimeError(log[-5000:])
    cubin_size = _check(nvrtc.nvrtcGetCUBINSize(program), "cubin size")
    cubin = b" " * int(cubin_size)
    _check(nvrtc.nvrtcGetCUBIN(program, cubin), "cubin")
    ptx = ""
    try:
        ptx_size = _check(nvrtc.nvrtcGetPTXSize(program), "ptx size")
        ptx_buf = b" " * int(ptx_size)
        _check(nvrtc.nvrtcGetPTX(program, ptx_buf), "ptx")
        ptx = ptx_buf.decode(errors="replace")
    except Exception:
        pass
    module = _check(driver.cuModuleLoadData(cubin), "module")
    fwd = _check(driver.cuModuleGetFunction(module, b"dmqe_outer_forward"), "forward function")
    bwd = _check(driver.cuModuleGetFunction(module, b"dmqe_outer_backward"), "backward function")
    regs = [int(x) for x in re.findall(r"Used (\d+) registers", log)]
    spills = [int(x) for x in re.findall(r"(\d+) bytes spill stores", log)]
    # PTX is a source-level census, useful even when cuobjdump is unavailable.
    inst = {}
    for line in ptx.splitlines():
        m = re.search(r"\s+([a-z][a-z0-9_.]+)\s", line)
        if m and not m.group(1).startswith((".param", ".reg", ".loc", ".pragma")):
            op = m.group(1)
            inst[op] = inst.get(op, 0) + 1
    return {"module": module, "forward": fwd, "backward": bwd,
            "registers": max(regs) if regs else None, "spill_store_bytes": max(spills) if spills else 0,
            "nvrtc_log": log, "ptx_instruction_counts": dict(sorted(inst.items()))}


def _launch(driver, fn, grid, block, shared, stream, args):
    values = []
    types = []
    for value, kind in args:
        if kind == "ptr":
            # ``torch.Tensor`` does not define ``__int__`` for CUDA tensors;
            # use its device address explicitly.  Integer addresses are kept
            # accepted for small standalone probes.
            address = value.data_ptr() if hasattr(value, "data_ptr") else int(value)
            values.append(ctypes.c_void_p(int(address)))
            types.append(ctypes.c_void_p)
        elif kind == "i32":
            values.append(ctypes.c_int(int(value)))
            types.append(ctypes.c_int)
        elif kind == "f32":
            values.append(ctypes.c_float(float(value)))
            types.append(ctypes.c_float)
        else:
            raise ValueError(kind)
    argv = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
    _check(driver.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shared, stream, ctypes.addressof(argv), 0), "launch")


def _edges(structure, cutoff, jitter, seed=0):
    import numpy as np
    from ase.io import read
    from ase.neighborlist import neighbor_list

    path, _, repeat = structure.partition("@")
    atoms = read(path)
    if repeat:
        atoms = atoms.repeat(tuple(int(v) for v in repeat.split("x")))
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, jitter, atoms.positions.shape)
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    species = np.unique(atoms.numbers, return_inverse=True)[1]
    slots = int(species.max()) + 1
    order = np.lexsort((species[i], j))
    i, j, d = i[order], j[order], d[order]
    key = j * slots + species[i]
    pair = species[j] * slots + species[i]
    counts = np.bincount(j, minlength=len(atoms)).astype("int32")
    row_ptr = np.concatenate(([0], np.cumsum(counts, dtype="int64"))).astype("int32")
    return key, pair, d.astype("float32"), row_ptr, len(atoms), slots, int(slots * slots)


def _err(a, b):
    import torch
    scale = max(float(b.abs().max()), 1e-30)
    return float((a - b).abs().max()) / scale


def _time(fn, rounds):
    import fastmace  # noqa: F401 (soft file locks before torch)
    import numpy as np
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); stop.record(); stop.synchronize()
        vals.append(start.elapsed_time(stop) * 1_000.0)
    return float(np.median(vals)), float(np.mean(vals))


def _source_commit():
    """Return the checkout snapshot id, including unpacked snapshots without .git."""
    root = Path(__file__).resolve().parents[1]
    marker = root / ".source_commit"
    if marker.is_file():
        value = marker.read_text().strip()
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _capture_and_alternate(torch, baseline, candidate, rounds):
    """Capture both paths in one process, then alternate graph A/B replay timings."""
    # Triton's first invocation may compile lazily.  Compile/warm each path
    # before graph capture so the measured replays contain no setup work.
    baseline()
    candidate()
    torch.cuda.synchronize()
    graphs = []
    for fn in (baseline, candidate):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs.append(graph)
    torch.cuda.synchronize()
    values = {"baseline": [], "candidate": []}
    for _ in range(rounds):
        for label, graph in (("baseline", graphs[0]), ("candidate", graphs[1])):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            stop.record()
            stop.synchronize()
            values[label].append(start.elapsed_time(stop) * 1_000.0)
    import numpy as np
    return {
        label: {"median_us": float(np.median(samples)), "mean_us": float(np.mean(samples)),
                "samples_us": samples}
        for label, samples in values.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--jitter", type=float, default=0.07)
    parser.add_argument("--pieces", type=int, default=896)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-run", action="store_true", help="compile/source census only")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dmqe_outer_lab.cu").write_text(CUDA_SOURCE, newline="\n")

    if args.no_run:
        print("wrote CUDA source; --no-run does not compile without a device", flush=True)
        return

    import numpy as np
    import torch

    key_np, pair_np, dist_np, row_ptr_np, nodes, slots, pairs = _edges(args.structure, args.cutoff, args.jitter)
    dev = torch.device("cuda")
    torch.cuda.init()
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    rng = torch.Generator(device=dev).manual_seed(3)
    edges, D, K, terms, J = len(key_np), 16, 64, 6, 66
    n_bins = 4 * args.pieces
    width = (args.cutoff - 0.3) / args.pieces
    r_lo = 0.3
    coef = torch.randn((pairs, args.pieces, terms, J), generator=rng, device=dev) * 0.05
    # Keep the coefficient/derivative relationship physically meaningful for
    # the optional finite-difference check: dcoef stores d/dr, so Chebyshev
    # derivatives in x are multiplied by each piece's scale.  Production's
    # table builder does the same (the baseline parity check would also pass
    # with independent random dcoef, but that would not test a derivative).
    coef_cpu = coef.detach().cpu().numpy()
    dcoef_cpu = np.polynomial.chebyshev.chebder(coef_cpu, axis=2)
    starts = (r_lo + width * torch.arange(args.pieces, device=dev)).float()
    scales = torch.full((args.pieces,), 2.0 / width, device=dev)
    dcoef_cpu *= scales.detach().cpu().numpy()[None, :, None, None]
    dcoef = torch.from_numpy(dcoef_cpu).to(device=dev, dtype=coef.dtype).contiguous()
    bins = (torch.arange(n_bins, device=dev) * args.pieces // n_bins).to(torch.int32)
    r = torch.from_numpy(dist_np).to(dev)
    pair = torch.from_numpy(pair_np).to(dev)
    key = torch.from_numpy(key_np).to(dev)
    row_ptr = torch.from_numpy(row_ptr_np).to(dev)
    y = torch.randn((edges, D), generator=rng, device=dev)
    grad = torch.randn((nodes * slots, D * K), generator=rng, device=dev)
    grad_density = torch.randn((edges,), generator=rng, device=dev)

    from fastmace.kernels.triton import tabled_outer_sum as tos

    def baseline_forward(inner=1):
        return tos.tabled_outer_sum_forward(r, pair, coef, starts, scales, bins, r_lo,
                                            n_bins / (args.cutoff - r_lo), args.cutoff, K, y, key,
                                            nodes * slots, inner)

    def baseline_backward(inner=1, grad_in=None):
        if grad_in is None:
            grad_in = grad
        return tos.tabled_outer_sum_backward(grad_in, torch.empty((0, 0), device=dev), y, key, r, pair, dcoef,
                                             starts, scales, bins, grad_density, r_lo,
                                             n_bins / (args.cutoff - r_lo), args.cutoff, inner, coef, K)

    module_info = _compile(CUDA_SOURCE, arch)
    from cuda.bindings import driver
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    p = lambda t: [(t.data_ptr(), "ptr")]
    common_f = [(r, "ptr"), (pair, "ptr"), (key, "ptr"), (coef, "ptr"), (starts, "ptr"),
                (scales, "ptr"), (bins, "ptr"), (y, "ptr"), (row_ptr, "ptr")]
    f_out = torch.empty((nodes * slots, D * K), device=dev)
    f_den = torch.empty((edges,), device=dev)
    f_rden = torch.empty((nodes,), device=dev)
    f_args = common_f + [(f_out, "ptr"), (f_den, "ptr"), (f_rden, "ptr"), (nodes, "i32"), (slots, "i32"),
                         (args.pieces, "i32"), (n_bins, "i32"), (terms, "i32"), (J, "i32"),
                         (r_lo, "f32"), (n_bins / (args.cutoff - r_lo), "f32"), (args.cutoff, "f32"),
                         (D, "i32"), (K, "i32"), (0, "i32"), (1, "i32")]
    def candidate_forward():
        _launch(driver, module_info["forward"], nodes, 128, slots * D * K * 4 + (K + D) * 4, stream, f_args)
        return f_out, f_den

    b_out_dr = torch.empty((edges,), device=dev)
    b_out_gy = torch.empty_like(y)
    b_args = [(grad, "ptr"), (r, "ptr"), (pair, "ptr"), (key, "ptr"), (coef, "ptr"), (dcoef, "ptr"),
              (starts, "ptr"), (scales, "ptr"), (bins, "ptr"), (grad_density, "ptr"), (y, "ptr"),
              (row_ptr, "ptr"), (b_out_dr, "ptr"), (b_out_gy, "ptr"), (nodes, "i32"), (slots, "i32"),
              (args.pieces, "i32"), (n_bins, "i32"), (terms, "i32"), (terms - 1, "i32"), (J, "i32"),
              (r_lo, "f32"), (n_bins / (args.cutoff - r_lo), "f32"), (args.cutoff, "f32"), (D, "i32"),
              (K, "i32"), (0, "i32"), (1, "i32")]
    def candidate_backward():
        _launch(driver, module_info["backward"], nodes, 128, (D * K + 2 * K + D + 4) * 4, stream, b_args)
        return b_out_dr, b_out_gy

    ref_f = tuple(v.clone() for v in baseline_forward())
    ref_b = tuple(v.clone() for v in baseline_backward())
    cand_f = candidate_forward(); cand_b = candidate_backward()
    torch.cuda.synchronize()
    forward_err = {"S": _err(cand_f[0], ref_f[0]), "density_edge": _err(cand_f[1], ref_f[1].reshape(-1))}
    backward_err = {"dr": _err(cand_b[0], ref_b[0]), "grad_y": _err(cand_b[1], ref_b[1])}

    # Exercise the production JOINED layout as well as the default D-major
    # layout.  The same data and derivative-consistent dcoef are used; only
    # the slot/channel stride interpretation changes.
    grad_inner = torch.randn_like(grad)
    f_args[-1] = (3, "i32")
    ref_f_inner = tuple(v.clone() for v in baseline_forward(3))
    cand_f_inner = candidate_forward()
    b_args[0] = (grad_inner, "ptr")
    ref_b_inner = tuple(v.clone() for v in baseline_backward(3, grad_inner))
    cand_b_inner = candidate_backward()
    torch.cuda.synchronize()
    inner_errors = {
        "S": _err(cand_f_inner[0], ref_f_inner[0]),
        "density_edge": _err(cand_f_inner[1], ref_f_inner[1].reshape(-1)),
        "dr": _err(cand_b_inner[0], ref_b_inner[0]),
        "grad_y": _err(cand_b_inner[1], ref_b_inner[1]),
    }
    # Restore the default layout and its original gradient pointer before the
    # receiver-density test and timing captures.
    b_args[0] = (grad, "ptr")
    f_args[-1] = (1, "i32")
    # Receiver-density mode is a pure work-removal side: compare the reduction
    # against the baseline edge density and leave it available for later VJP use.
    f_rden.zero_()
    f_args[-2] = (1, "i32")
    candidate_forward()
    torch.cuda.synchronize()
    receivers = torch.div(key, slots, rounding_mode="floor").to(torch.int64)
    ref_rden = torch.zeros((nodes,), device=dev).index_add(0, receivers, ref_f[1].reshape(-1))
    receiver_density_err = _err(f_rden, ref_rden)

    # Same-process alternating A/B timings.  The baseline allocates its normal
    # outputs; candidate writes its preallocated outputs.
    f_args[-2] = (0, "i32")
    timing = {}
    capture_error = None
    try:
        timing["forward_us"] = _capture_and_alternate(torch, baseline_forward, candidate_forward, args.rounds)
        timing["backward_us"] = _capture_and_alternate(torch, baseline_backward, candidate_backward, args.rounds)
    except Exception as exc:
        # Some driver/Triton combinations do not permit custom-op capture.
        # Keep a useful same-process A/B fallback and preserve the exact reason
        # in the JSON rather than silently mixing timing modes.
        capture_error = f"{type(exc).__name__}: {exc}"
        timing = {"mode": "direct-event-fallback", "forward_us": _time(baseline_forward, args.rounds),
                  "backward_us": _time(baseline_backward, args.rounds)}
    result = {
        "tool": "dmqe_outer_lab", "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "capability": capability, "torch": torch.__version__,
        "structure": args.structure, "jitter": args.jitter, "nodes": nodes, "edges": edges,
        "slots": slots, "pairs": pairs, "D": D, "K": K, "pieces": args.pieces, "terms": terms,
        "forward_error": forward_err, "backward_error": backward_err, "inner3_error": inner_errors,
        "receiver_density_error": receiver_density_err, "timing": timing,
        "source_commit": _source_commit(),
        "ncu": shutil.which("ncu") or None,
        "census": {k: v for k, v in module_info.items() if k not in ("module", "forward", "backward", "nvrtc_log")},
        "census_note": "PTX/NVRTC candidate census only; current Triton kernel census and NCU counters were not collected.",
    }
    if capture_error is not None:
        result["timing_capture_error"] = capture_error
    (args.output / "dmqe_outer_lab.json").write_text(json.dumps(result, indent=1) + "\n")
    (args.output / "nvrtc.log").write_text(module_info["nvrtc_log"], newline="\n")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
