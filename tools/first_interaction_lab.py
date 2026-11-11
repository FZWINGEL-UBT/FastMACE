"""Falsifiers for the first interaction's two tabled kernels (next-steps C1 and C2, 11 November 2026).

Ideas 18 and 19 of ``plans/ideas-20261111-round3.md``. Times ``tabled_outer_sum``'s forward and fused backward on the
real key order and edge lengths (SrTiO3 10x10x10, jittered, edges sorted by receiver and sender species; OMAT-0
medium's interaction-1 shapes: a random table [pairs, 896 pieces, 6 terms, 66 columns], K = 64, D = 16), against
copies of the production kernels with timing-only changes. Numbers of the falsifiers are wrong on purpose.

- ``prod``: the production op (``tabled_outer_sum_forward``: zero fill + kernel; ``tabled_outer_sum_backward``).
- ``copy``: this file's copy of the kernel with every switch off (checked bitwise against ``prod``).
- **C1 falsifier** ``noslot``: the forward without its slot traffic: the per-run ``atomic_add`` of the slot block
  becomes a store whose mask is false at run time (the sums stay live), and the output is ``empty`` (no zero fill).
- ``byslot-wN``: a prototype of idea 18's shape (not the falsifier the plan names; timing and a rounding check):
  one program of N warps per slot over that slot's edges, the [D, K] sums in registers, one plain store per slot.
- **C2 falsifier** ``oneload``: one table load per column and edge (and one for the density column) in place of the
  six value rows (forward, the backward's recompute) and five slope rows (the backward without POWER_BACKWARD).
  The same arithmetic on the one row.

The backward runs both ways on every GPU: ``recompute`` (Clenshaw value + slope rows; the A100's default) and
``power`` (dual Horner from the power basis; POWER_BACKWARD, the H200's and the 5060 Ti's default).

``--census``: no GPU; every variant compiled with ``triton.compile`` for sm_80, sm_90 and sm_120 with its PTX/SASS
global loads, registers and spills (``ptxas -v``).

Watchdog (local 5060 Ti rule, 11 November 2026; ``--watchdog``): a CUDA memory fraction of 0.9, and the process exits
if its reserved memory passes 14.8 GiB, the card's used memory passes 16,000 MiB, a new "soft lockup" or "rcu ...
detected stalls" kernel line appears, or a timed call takes over 3x its variant's first median.

Usage::

    python tools/first_interaction_lab.py --output DIR [--watchdog]
    CUDA_VISIBLE_DEVICES= python tools/first_interaction_lab.py --census --output DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ----------------------------------------------------------------------------------------------------------------------
# Kernels (copies of fastmace/kernels/triton/tabled_outer_sum.py's unpacked walks, with timing-only switches)


@cache
def kernels():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()

    @triton.jit
    def forward(y_ptr, key_ptr, r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, out_ptr, density_ptr,
                num_edges, n_pieces, n_bins, r_lo, bin_scale, r_max, live, K: tl.constexpr, D: tl.constexpr,
                INNER: tl.constexpr, TERMS: tl.constexpr, J: tl.constexpr, CHUNK: tl.constexpr,
                BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, SLOTS: tl.constexpr, ONE: tl.constexpr):
        """tabled_outer_sum._forward_kernel (unpacked). SLOTS False: no slot atomics (a store masked off by
        ``live`` = 0); ONE: one load per column (row 0 for every term)."""
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
        current = tl.load(key_ptr + start).to(tl.int64)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:
                if SLOTS:
                    tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")
                else:
                    tl.store(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask & (live != 0))
                acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
                current = key
            row = edge.to(tl.int64)
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            base = (pair * n_pieces + piece) * TERMS * J
            inside = r < r_max
            v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            c1 = 0.0 * x
            c2 = 0.0 * x
            if ONE:
                c0 = tl.load(coef_ptr + base + k, mask=k < K, other=0.0)
                cd0 = tl.load(coef_ptr + base + (J - 1))
            for t in tl.static_range(TERMS - 1, 0, -1):
                if ONE:
                    c = c0
                    cd = cd0
                else:
                    c = tl.load(coef_ptr + base + t * J + k, mask=k < K, other=0.0)
                    cd = tl.load(coef_ptr + base + t * J + (J - 1))
                v = c + 2.0 * x * v1 - v2
                v2 = v1
                v1 = v
                cn = cd + 2.0 * x * c1 - c2
                c2 = c1
                c1 = cn
            if ONE:
                first = c0
                first_d = cd0
            else:
                first = tl.load(coef_ptr + base + k, mask=k < K, other=0.0)
                first_d = tl.load(coef_ptr + base + (J - 1))
            a = tl.where(inside, first + x * v1 - v2, 0.0)
            density = tl.where(inside, first_d + x * c1 - c2, 0.0)
            tl.store(density_ptr + row, density)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            acc += y[:, None] * a[None, :]
        if SLOTS:
            tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")
        else:
            tl.store(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask & (live != 0))

    @triton.jit
    def by_slot(y_ptr, slot_start_ptr, r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, out_ptr,
                density_ptr, n_pieces, n_bins, r_lo, bin_scale, r_max, K: tl.constexpr, D: tl.constexpr,
                TERMS: tl.constexpr, J: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        """Idea 18's shape (prototype, INNER = 1): one program per slot over its edges, sums in registers, one
        plain store. ``slot_start [slots + 1]``: the slot's first edge (keys sorted)."""
        slot = tl.program_id(0)
        lo = tl.load(slot_start_ptr + slot)
        hi = tl.load(slot_start_ptr + slot + 1)
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
        for edge in range(lo, hi):
            row = edge.to(tl.int64)
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            base = (pair * n_pieces + piece) * TERMS * J
            inside = r < r_max
            v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            c1 = 0.0 * x
            c2 = 0.0 * x
            for t in tl.static_range(TERMS - 1, 0, -1):
                c = tl.load(coef_ptr + base + t * J + k, mask=k < K, other=0.0)
                v = c + 2.0 * x * v1 - v2
                v2 = v1
                v1 = v
                cd = tl.load(coef_ptr + base + t * J + (J - 1))
                cn = cd + 2.0 * x * c1 - c2
                c2 = c1
                c1 = cn
            a = tl.where(inside, tl.load(coef_ptr + base + k, mask=k < K, other=0.0) + x * v1 - v2, 0.0)
            density = tl.where(inside, tl.load(coef_ptr + base + (J - 1)) + x * c1 - c2, 0.0)
            tl.store(density_ptr + row, density)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            acc += y[:, None] * a[None, :]
        tl.store(out_ptr + slot.to(tl.int64) * (D * K) + d[:, None] * K + k[None, :], acc, mask=mask)

    @triton.jit
    def backward(grad_ptr, y_ptr, key_ptr, r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr,
                 grad_density_ptr, dr_ptr, grad_y_ptr, num_edges, E_DENSITY, n_pieces, n_bins, r_lo, bin_scale,
                 r_max, K: tl.constexpr, D: tl.constexpr, INNER: tl.constexpr, TERMS: tl.constexpr, J: tl.constexpr,
                 CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, coef_ptr, VTERMS: tl.constexpr,
                 POWER: tl.constexpr, ONE: tl.constexpr):
        """tabled_outer_sum._kernel with RECOMPUTE (unpacked; POWER as there). ONE: one load per column and edge
        (row 0 of the value table for every term of the value and the slope series)."""
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        current = tl.load(key_ptr + start).to(tl.int64)
        block = tl.load(grad_ptr + _base(current, D, K, INNER) + offsets, mask=mask, other=0.0)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:
                block = tl.load(grad_ptr + _base(key, D, K, INNER) + offsets, mask=mask, other=0.0)
                current = key
            row = edge.to(tl.int64)
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            vbase = (pair * n_pieces + piece) * VTERMS * J
            if ONE:
                c0 = tl.load(coef_ptr + vbase + k, mask=k < K, other=0.0)
                cd0 = tl.load(coef_ptr + vbase + (J - 1))
            if POWER:
                if ONE:
                    pa = c0
                    qa = cd0
                else:
                    pa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + k, mask=k < K, other=0.0)
                    qa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + (J - 1))
                pd = tl.zeros((BLOCK_K,), dtype=tl.float32)
                qd = 0.0 * x
                for t in tl.static_range(VTERMS - 2, -1, -1):
                    if ONE:
                        c = c0
                        cd = cd0
                    else:
                        c = tl.load(coef_ptr + vbase + t * J + k, mask=k < K, other=0.0)
                        cd = tl.load(coef_ptr + vbase + t * J + (J - 1))
                    pd = pd * x + pa
                    pa = pa * x + c
                    qd = qd * x + qa
                    qa = qa * x + cd
                a = tl.where(r < r_max, pa, 0.0)
            else:
                v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                for t in tl.static_range(VTERMS - 1, 0, -1):
                    if ONE:
                        c = c0
                    else:
                        c = tl.load(coef_ptr + vbase + t * J + k, mask=k < K, other=0.0)
                    v = c + 2.0 * x * v1 - v2
                    v2 = v1
                    v1 = v
                if ONE:
                    a = c0 + x * v1 - v2
                else:
                    a = tl.load(coef_ptr + vbase + k, mask=k < K, other=0.0) + x * v1 - v2
                a = tl.where(r < r_max, a, 0.0)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            tl.store(grad_y_ptr + row * D + d, tl.sum(block * a[None, :], axis=1), mask=d < D)
            grad_a = tl.sum(block * y[:, None], axis=0)
            if POWER:
                sc = tl.load(scales_ptr + piece)
                slope = pd * sc
                slope_density = qd * sc
            else:
                base = (pair * n_pieces + piece) * TERMS * J
                b1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                b2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                c1 = 0.0 * x
                c2 = 0.0 * x
                for t in tl.static_range(TERMS - 1, 0, -1):
                    if ONE:
                        c = c0
                        cd = cd0
                    else:
                        c = tl.load(dcoef_ptr + base + t * J + k, mask=k < K, other=0.0)
                        cd = tl.load(dcoef_ptr + base + t * J + (J - 1))
                    b = c + 2.0 * x * b1 - b2
                    b2 = b1
                    b1 = b
                    cn = cd + 2.0 * x * c1 - c2
                    c2 = c1
                    c1 = cn
                if ONE:
                    slope = c0 + x * b1 - b2
                    slope_density = cd0 + x * c1 - c2
                else:
                    slope = tl.load(dcoef_ptr + base + k, mask=k < K, other=0.0) + x * b1 - b2
                    slope_density = tl.load(dcoef_ptr + base + (J - 1)) + x * c1 - c2
            grad_density = tl.load(grad_density_ptr + row, mask=edge < E_DENSITY, other=0.0)
            total = tl.sum(grad_a * slope, axis=0) + grad_density * slope_density
            total = tl.where((r < r_max) & (r >= r_lo), total, 0.0)
            tl.store(dr_ptr + row, total)

    return forward, by_slot, backward


# ----------------------------------------------------------------------------------------------------------------------
# Census (no GPU)

POINTER_TYPES = {"pair_ptr": "*i32", "bins_ptr": "*i32", "key_ptr": "*i64", "slot_start_ptr": "*i64"}
INTS = ("num_edges", "E_DENSITY", "n_pieces", "n_bins", "live")
ALIGNED_INTS = ("n_pieces", "n_bins")
FLOATS = ("r_lo", "bin_scale", "r_max")


def _source(kernel, constexprs):
    from triton.compiler import ASTSource

    signature, attrs = {}, {}
    for index, name in enumerate(kernel.arg_names):
        if name in constexprs:
            signature[name] = "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = POINTER_TYPES.get(name, "*fp32")
            attrs[(index,)] = [["tt.divisibility", 16]]
        elif name in INTS:
            signature[name] = "i32"
            if name in ALIGNED_INTS:
                attrs[(index,)] = [["tt.divisibility", 16]]
        elif name in FLOATS:
            signature[name] = "fp32"
        else:
            raise ValueError(f"no type for argument {name}")
    return ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)


def census_variants(k=64, d=16, terms=6, columns=66):
    from fastmace.kernels.triton import tabled_outer_sum as tos

    forward, by_slot, backward = kernels()
    common = dict(K=k, D=d, J=columns, BLOCK_K=64, BLOCK_D=16)
    fwd = dict(common, INNER=1, TERMS=terms, CHUNK=tos.FORWARD_CHUNK)
    bwd = dict(common, INNER=1, TERMS=terms - 1, CHUNK=tos.CHUNK, VTERMS=terms)
    rows = [("forward-copy", forward, dict(fwd, SLOTS=True, ONE=False), tos.FORWARD_WARPS),
            ("forward-noslot", forward, dict(fwd, SLOTS=False, ONE=False), tos.FORWARD_WARPS),
            ("forward-oneload", forward, dict(fwd, SLOTS=True, ONE=True), tos.FORWARD_WARPS)]
    rows += [(f"byslot-w{w}", by_slot, dict(common, TERMS=terms), w) for w in (1, 2, 4)]
    for power in (False, True):
        name = "power" if power else "recompute"
        rows += [(f"backward-{name}-copy", backward, dict(bwd, POWER=power, ONE=False), tos.WARPS),
                 (f"backward-{name}-oneload", backward, dict(bwd, POWER=power, ONE=True), tos.WARPS)]
    return rows


def census(output):
    import triton
    from triton.backends.compiler import GPUTarget

    import packed_coef_lab as pcl

    rows = []
    for capability in (80, 90, 120):
        for name, kernel, constexprs, warps in census_variants():
            row = {"sm": capability, "variant": name, "warps": warps}
            try:
                compiled = triton.compile(_source(kernel, constexprs), target=GPUTarget("cuda", capability, 32),
                                          options={"num_warps": warps})
                ptx = compiled.asm["ptx"]
                row["ptx_loads"] = pcl.ptx_loads(ptx)
                row["ptx_atomics"] = len(re.findall(r"\batom\.global|\bred\.global", ptx))
                row["ptx_instructions"] = sum(1 for line in ptx.splitlines()
                                              if re.match(r"\s+[a-z@]", line) and not line.strip().startswith("//"))
                row["shared_mem"] = compiled.metadata.shared
                usage = pcl.ptxas_usage(ptx, capability)
                row["registers"], row["spill_stores"], row["spill_loads"] = usage if usage else (None, None, None)
                try:
                    sass = compiled.asm["sass"]
                    row["sass_loads"] = pcl.sass_loads(sass)
                    row["sass_instructions"] = len(re.findall(r"^[-0-9a-fY:]+\t[A-Z]", sass, re.M))
                except Exception as error:
                    row["sass_loads"] = f"unavailable ({type(error).__name__})"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"[:400]
            rows.append(row)
            print(json.dumps(row), flush=True)
    (output / "census.json").write_text(json.dumps({"tool": "first_interaction_lab --census",
                                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                                    "triton": triton.__version__, "rows": rows}, indent=1))


# ----------------------------------------------------------------------------------------------------------------------
# Watchdog

PATTERN = re.compile(r"soft lockup|rcu.*detected stalls")


def _kernel_lines():
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    return sum(1 for line in out.splitlines() if PATTERN.search(line))


def _card_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(v) for v in out.split())
    except Exception:
        return None


def start_watchdog(log):
    import torch

    torch.cuda.set_per_process_memory_fraction(0.9)
    baseline = _kernel_lines()
    log(f"watchdog: memory fraction 0.9; kernel-log baseline {baseline} matching lines"
        + (" (dmesg unreadable: that check is off)" if baseline is None else ""))

    def watch():
        while True:
            reserved = torch.cuda.memory_reserved() / 2 ** 30
            used = _card_used_mib()
            lines = _kernel_lines() if baseline is not None else None
            reason = None
            if reserved > 14.8:
                reason = f"process reserved {reserved:.2f} GiB > 14.8"
            elif used is not None and used > 16000:
                reason = f"card used {used} MiB > 16000"
            elif lines is not None and lines > baseline:
                reason = "new soft lockup / rcu stall kernel line"
            if reason:
                log(f"WATCHDOG KILL: {reason}")
                os._exit(86)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


# ----------------------------------------------------------------------------------------------------------------------
# GPU timing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--jitter", type=float, default=0.07)
    p.add_argument("--pieces", type=int, default=896)
    p.add_argument("--rounds", type=int, default=41)
    p.add_argument("--repeat", type=int, default=10,
                   help="calls per timed sample, back to back between two events (keeps host launch time out)")
    p.add_argument("--census", action="store_true")
    p.add_argument("--watchdog", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    logfile = (args.output / ("census.log" if args.census else "lab.log")).open("a")

    def log(text):
        line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    if args.census:
        census(args.output)
        return

    import numpy as np
    import torch
    import triton

    from fastmace.kernels.triton import tabled_outer_sum as tos
    from tabled_sum_lab import edges_of

    if args.watchdog:
        start_watchdog(log)
    forward_k, by_slot_k, backward_k = kernels()

    key, pair, dist, slots, pairs = edges_of(args.structure, args.cutoff, args.jitter)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    r_lo, r_max, n_bins = 0.5, args.cutoff, 4 * args.pieces
    width = (r_max - r_lo) / args.pieces
    coef = torch.randn((pairs, args.pieces, 6, 66), generator=g).to(dev)
    pcoef = torch.randn((pairs, args.pieces, 6, 66), generator=g).to(dev)
    dcoef = torch.randn((pairs, args.pieces, 5, 66), generator=g).to(dev)
    starts = (r_lo + width * torch.arange(args.pieces, dtype=torch.float64)).float().to(dev)
    scales = torch.full((args.pieces,), 2.0 / width).to(dev)
    bins = (torch.arange(n_bins) * args.pieces // n_bins).to(torch.int32).to(dev)
    bin_scale = n_bins / (r_max - r_lo)
    edges, d, k = len(key), 16, 64
    r = torch.from_numpy(dist.astype(np.float32)).reshape(-1, 1).to(dev)
    r_flat = r.reshape(-1).contiguous()
    pair_t = torch.from_numpy(pair.astype(np.int32)).to(dev)
    key_t = torch.from_numpy(key.astype(np.int64)).to(dev)
    slot_start = torch.searchsorted(key_t, torch.arange(slots + 1, device=dev, dtype=torch.int64)).contiguous()
    y = torch.randn((edges, d), generator=g).to(dev)
    grad = torch.randn((slots, d * k), generator=g).to(dev)
    grad_density = torch.randn((edges, 1), generator=g).to(dev).reshape(-1).contiguous()
    table = (r, pair_t, coef, starts, scales, bins, r_lo, bin_scale, r_max)
    log(f"{torch.cuda.get_device_name()}: {edges} edges, {slots} slots, {pairs} pairs, "
        f"{edges / max(1, int((slot_start[1:] > slot_start[:-1]).sum())):.1f} edges per filled slot")
    fwd_args = (n_bins, float(r_lo), float(bin_scale), float(r_max))

    def fwd_copy(slots_on=True, one=False, fill=True):
        out = (torch.zeros if fill else torch.empty)((slots, d * k), device=dev)
        density = r.new_empty((edges, 1))
        forward_k[(triton.cdiv(edges, tos.FORWARD_CHUNK),)](
            y, key_t, r_flat, pair_t, coef, starts, scales, bins, out, density, edges, coef.shape[1], *fwd_args, 0,
            K=k, D=d, INNER=1, TERMS=6, J=66, CHUNK=tos.FORWARD_CHUNK, BLOCK_K=64, BLOCK_D=16, SLOTS=slots_on,
            ONE=one, num_warps=tos.FORWARD_WARPS)
        return out, density

    def fwd_by_slot(warps):
        out = torch.empty((slots, d * k), device=dev)
        density = r.new_empty((edges, 1))
        by_slot_k[(slots,)](y, slot_start, r_flat, pair_t, coef, starts, scales, bins, out, density, coef.shape[1],
                            *fwd_args, K=k, D=d, TERMS=6, J=66, BLOCK_K=64, BLOCK_D=16, num_warps=warps)
        return out, density

    def bwd_copy(power, one=False):
        dr = r_flat.new_empty((edges,))
        grad_y = torch.empty_like(y)
        backward_k[(triton.cdiv(edges, tos.CHUNK),)](
            grad, y, key_t, r_flat, pair_t, dcoef, starts, scales, bins, grad_density, dr, grad_y, edges, edges,
            coef.shape[1], *fwd_args, K=k, D=d, INNER=1, TERMS=5, J=66, CHUNK=tos.CHUNK, BLOCK_K=64, BLOCK_D=16,
            coef_ptr=pcoef if power else coef, VTERMS=6, POWER=power, ONE=one, num_warps=tos.WARPS)
        return dr, grad_y

    def bwd_prod(power):
        return tos.tabled_outer_sum_backward(grad, y.new_empty((0, 0)), y, key_t, r, pair_t, dcoef, starts, scales,
                                             bins, grad_density, r_lo, bin_scale, r_max, 1,
                                             pcoef if power else coef, k, power)

    variants = {
        "fwd-prod": lambda: tos.tabled_outer_sum_forward(*table, k, y, key_t, slots),
        "fwd-copy": lambda: fwd_copy(),
        "fwd-fill-only": lambda: torch.zeros((slots, d * k), device=dev),
        "fwd-noslot": lambda: fwd_copy(slots_on=False, fill=False),
        "fwd-noslot-fill": lambda: fwd_copy(slots_on=False, fill=True),
        "fwd-oneload": lambda: fwd_copy(one=True),
        **{f"fwd-byslot-w{w}": (lambda w=w: fwd_by_slot(w)) for w in (1, 2, 4)},
        "bwd-recompute-prod": lambda: bwd_prod(False),
        "bwd-recompute-copy": lambda: bwd_copy(False),
        "bwd-recompute-oneload": lambda: bwd_copy(False, True),
        "bwd-power-prod": lambda: bwd_prod(True),
        "bwd-power-copy": lambda: bwd_copy(True),
        "bwd-power-oneload": lambda: bwd_copy(True, True),
    }

    def err(new, ref):
        return max(float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
                   for a, b in zip(new, ref))

    # Faithfulness of the copies, and the prototype's rounding.
    checks = {}
    ref_f = variants["fwd-prod"]()
    for name in ("fwd-copy", "fwd-byslot-w1", "fwd-byslot-w2", "fwd-byslot-w4"):
        checks[name] = err(variants[name](), ref_f)
    for name in ("recompute", "power"):
        ref_b = variants[f"bwd-{name}-prod"]()
        checks[f"bwd-{name}-copy"] = err(variants[f"bwd-{name}-copy"](), ref_b)
    torch.cuda.synchronize()
    log(f"checks (max |d| / max |ref| against prod): {json.dumps(checks)}")

    times = {name: [] for name in variants}
    reference = {}
    for name, fn in variants.items():  # warm-up and compile
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    for round_ in range(args.rounds):
        order = list(variants) if round_ % 2 == 0 else list(reversed(variants))
        for name in order:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeat):
                variants[name]()
            stop.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(stop) * 1e3 / args.repeat
            times[name].append(us)
            if round_ == 4:
                reference[name] = float(np.median(times[name]))
            if args.watchdog and name in reference and us > 3 * reference[name] and us > 200:
                log(f"WATCHDOG KILL: {name} took {us:.0f} us > 3x {reference[name]:.0f}")
                os._exit(87)
    rows = {}
    for name, values in times.items():
        values = np.array(values[3:])
        rows[name] = {"median_us": float(np.median(values)), "p10_us": float(np.percentile(values, 10)),
                      "p90_us": float(np.percentile(values, 90))}
        log(f"{name:24s} {rows[name]['median_us']:8.1f} us  (p10 {rows[name]['p10_us']:.1f}, "
            f"p90 {rows[name]['p90_us']:.1f})")

    def m(name):
        return rows[name]["median_us"]

    summary = {
        "C1_noslot_saving_us": m("fwd-copy") - m("fwd-noslot"),
        "C1_noslot_kernel_saving_us": m("fwd-copy") - m("fwd-noslot-fill"),
        "C1_byslot_best_saving_us": m("fwd-copy") - min(m(f"fwd-byslot-w{w}") for w in (1, 2, 4)),
        "C2_forward_saving_us": m("fwd-copy") - m("fwd-oneload"),
        "C2_backward_recompute_saving_us": m("bwd-recompute-copy") - m("bwd-recompute-oneload"),
        "C2_backward_power_saving_us": m("bwd-power-copy") - m("bwd-power-oneload"),
    }
    log(f"summary: {json.dumps(summary)}")
    meta = {"tool": "first_interaction_lab", "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
            "edges": edges, "slots": slots, "pairs": pairs, "pieces": args.pieces, "structure": args.structure,
            "rounds": args.rounds, "repeat": args.repeat, "source_commit": os.environ.get("FASTMACE_SOURCE_COMMIT")}
    (args.output / "lab.json").write_text(json.dumps({"meta": meta, "checks": checks, "rows": rows,
                                                      "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
