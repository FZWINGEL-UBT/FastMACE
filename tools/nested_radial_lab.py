"""The nested radial basis's real kernel shape, timed (step 3 of next-steps rev2 §1, 11 November 2026).

Step 2 (``first_interaction_lab.py --only 'copy|k32'``) bounded the gain with every edge at 32 columns and a [16, 32]
slot block. The real kernel keeps the [16, 64] slot block (near and far edges share slots) and branches per edge:

- ``split``: the K = 64 columns as two halves of ``HALF`` = 32, each with its own [D, HALF] accumulator (forward) or
  gradient block (backward). Every edge evaluates the low half (its value series, and in the backward its slope
  series) and the density column. An edge with ``r < r_far`` (near) also evaluates the high half. The branch is a
  scalar ``if`` per edge (one warp, no divergence).
- Forward: a run's low block is added to its slot always, the high block only if the run had a near edge.
- Backward: the slot's low gradient block is loaded per run, the high block only when the run's first near edge
  needs it.

Check: on a table whose columns ``HALF:K`` are zero on every piece from ``r_far`` (``r_far`` set to a piece start),
``split`` equals the K = 64 copy (``first_interaction_lab``'s kernels) to rounding.

Timing as ``first_interaction_lab``: SrTiO3 10x10x10 (368,752 edges), random tables of the production shape, 10 calls
back to back per sample, alternating order, medians. Gain = (copy - split) per kernel, at the real far fraction.

Usage::

    python tools/nested_radial_lab.py --output DIR [--watchdog] [--r-far 3.15]
    CUDA_VISIBLE_DEVICES= python tools/nested_radial_lab.py --census --output DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))


@cache
def kernels():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()

    @triton.jit
    def _values(coef_ptr, base, x, col, J: tl.constexpr, TERMS: tl.constexpr):
        """Clenshaw sum of columns ``col`` (a [HALF] index vector) of the value rows at ``base``."""
        v1 = tl.zeros(col.shape, dtype=tl.float32)
        v2 = tl.zeros(col.shape, dtype=tl.float32)
        for t in tl.static_range(TERMS - 1, 0, -1):
            c = tl.load(coef_ptr + base + t * J + col)
            v = c + 2.0 * x * v1 - v2
            v2 = v1
            v1 = v
        return tl.load(coef_ptr + base + col) + x * v1 - v2

    @triton.jit
    def _horner(coef_ptr, base, x, col, J: tl.constexpr, TERMS: tl.constexpr):
        """(p, dp/dx) of columns ``col`` by dual Horner from the power-basis rows at ``base``."""
        pa = tl.load(coef_ptr + base + (TERMS - 1) * J + col)
        pd = tl.zeros(col.shape, dtype=tl.float32)
        for t in tl.static_range(TERMS - 2, -1, -1):
            c = tl.load(coef_ptr + base + t * J + col)
            pd = pd * x + pa
            pa = pa * x + c
        return pa, pd

    @triton.jit
    def forward_split(y_ptr, key_ptr, r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, out_ptr,
                      density_ptr, num_edges, n_pieces, n_bins, r_lo, bin_scale, r_max, r_far, K: tl.constexpr,
                      D: tl.constexpr, INNER: tl.constexpr, TERMS: tl.constexpr, J: tl.constexpr,
                      CHUNK: tl.constexpr, HALF: tl.constexpr, BLOCK_D: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        h = tl.arange(0, HALF)
        mask = (d < D)[:, None] & (h < HALF)[None, :]
        offsets = d[:, None] * (INNER * K) + h[None, :]
        lo = tl.zeros((BLOCK_D, HALF), dtype=tl.float32)
        hi = tl.zeros((BLOCK_D, HALF), dtype=tl.float32)
        near_run = start * 0
        current = tl.load(key_ptr + start).to(tl.int64)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:  # a run ends: its low block always, its high block if it had a near edge
                slot = out_ptr + _base(current, D, K, INNER) + offsets
                tl.atomic_add(slot, lo, mask=mask, sem="relaxed")
                if near_run != 0:
                    tl.atomic_add(slot + HALF, hi, mask=mask, sem="relaxed")
                lo = tl.zeros((BLOCK_D, HALF), dtype=tl.float32)
                hi = tl.zeros((BLOCK_D, HALF), dtype=tl.float32)
                near_run = start * 0
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
            c1 = 0.0 * x
            c2 = 0.0 * x
            for t in tl.static_range(TERMS - 1, 0, -1):
                cd = tl.load(coef_ptr + base + t * J + (J - 1))
                cn = cd + 2.0 * x * c1 - c2
                c2 = c1
                c1 = cn
            density = tl.where(inside, tl.load(coef_ptr + base + (J - 1)) + x * c1 - c2, 0.0)
            tl.store(density_ptr + row, density)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            a = tl.where(inside, _values(coef_ptr, base, x, h, J, TERMS), 0.0)
            lo += y[:, None] * a[None, :]
            if r < r_far:  # near: the high half too
                b = tl.where(inside, _values(coef_ptr, base, x, h + HALF, J, TERMS), 0.0)
                hi += y[:, None] * b[None, :]
                near_run = start * 0 + 1
        slot = out_ptr + _base(current, D, K, INNER) + offsets
        tl.atomic_add(slot, lo, mask=mask, sem="relaxed")
        if near_run != 0:
            tl.atomic_add(slot + HALF, hi, mask=mask, sem="relaxed")

    @triton.jit
    def backward_split(grad_ptr, y_ptr, key_ptr, r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr,
                       grad_density_ptr, dr_ptr, grad_y_ptr, num_edges, E_DENSITY, n_pieces, n_bins, r_lo, bin_scale,
                       r_max, r_far, K: tl.constexpr, D: tl.constexpr, INNER: tl.constexpr, TERMS: tl.constexpr,
                       J: tl.constexpr, CHUNK: tl.constexpr, HALF: tl.constexpr, BLOCK_D: tl.constexpr, coef_ptr,
                       VTERMS: tl.constexpr, POWER: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        h = tl.arange(0, HALF)
        mask = (d < D)[:, None] & (h < HALF)[None, :]
        offsets = d[:, None] * (INNER * K) + h[None, :]
        current = tl.load(key_ptr + start).to(tl.int64)
        block = tl.load(grad_ptr + _base(current, D, K, INNER) + offsets, mask=mask, other=0.0)
        upper = tl.zeros((BLOCK_D, HALF), dtype=tl.float32)
        upper_key = current * 0 - 1  # the slot whose high block is in ``upper`` (-1: none)
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
            sc = tl.load(scales_ptr + piece)
            vbase = (pair * n_pieces + piece) * VTERMS * J
            base = (pair * n_pieces + piece) * TERMS * J
            inside = r < r_max
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            # the density column's slope
            if POWER:
                qa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + (J - 1))
                qd = 0.0 * x
                for t in tl.static_range(VTERMS - 2, -1, -1):
                    cd = tl.load(coef_ptr + vbase + t * J + (J - 1))
                    qd = qd * x + qa
                    qa = qa * x + cd
                slope_density = qd * sc
            else:
                c1 = 0.0 * x
                c2 = 0.0 * x
                for t in tl.static_range(TERMS - 1, 0, -1):
                    cd = tl.load(dcoef_ptr + base + t * J + (J - 1))
                    cn = cd + 2.0 * x * c1 - c2
                    c2 = c1
                    c1 = cn
                slope_density = tl.load(dcoef_ptr + base + (J - 1)) + x * c1 - c2
            # the low half
            if POWER:
                pa, pd = _horner(coef_ptr, vbase, x, h, J, VTERMS)
                a = tl.where(inside, pa, 0.0)
                slope = pd * sc
            else:
                a = tl.where(inside, _values(coef_ptr, vbase, x, h, J, VTERMS), 0.0)
                slope = _values(dcoef_ptr, base, x, h, J, TERMS)
            grad_y = tl.sum(block * a[None, :], axis=1)
            total = tl.sum(tl.sum(block * y[:, None], axis=0) * slope, axis=0)
            if r < r_far:  # near: the high half, its gradient block loaded once per slot
                if upper_key != current:
                    upper = tl.load(grad_ptr + _base(current, D, K, INNER) + offsets + HALF, mask=mask, other=0.0)
                    upper_key = current
                if POWER:
                    pa, pd = _horner(coef_ptr, vbase, x, h + HALF, J, VTERMS)
                    b = tl.where(inside, pa, 0.0)
                    slope_b = pd * sc
                else:
                    b = tl.where(inside, _values(coef_ptr, vbase, x, h + HALF, J, VTERMS), 0.0)
                    slope_b = _values(dcoef_ptr, base, x, h + HALF, J, TERMS)
                grad_y += tl.sum(upper * b[None, :], axis=1)
                total += tl.sum(tl.sum(upper * y[:, None], axis=0) * slope_b, axis=0)
            tl.store(grad_y_ptr + row * D + d, grad_y, mask=d < D)
            grad_density = tl.load(grad_density_ptr + row, mask=edge < E_DENSITY, other=0.0)
            total = total + grad_density * slope_density
            total = tl.where((r < r_max) & (r >= r_lo), total, 0.0)
            tl.store(dr_ptr + row, total)

    return forward_split, backward_split


def census_variants(k=64, d=16, terms=6, columns=66):
    from fastmace.kernels.triton import tabled_outer_sum as tos

    forward_split, backward_split = kernels()
    common = dict(K=k, D=d, J=columns, HALF=k // 2, BLOCK_D=16, INNER=1)
    rows = [("forward-split", forward_split, dict(common, TERMS=terms, CHUNK=tos.FORWARD_CHUNK), tos.FORWARD_WARPS)]
    for power in (False, True):
        name = "power" if power else "recompute"
        rows += [(f"backward-{name}-split", backward_split,
                  dict(common, TERMS=terms - 1, CHUNK=tos.CHUNK, VTERMS=terms, POWER=power), tos.WARPS)]
    return rows


def census(output):
    import first_interaction_lab as fil
    import packed_coef_lab as pcl
    import triton
    from triton.backends.compiler import GPUTarget

    fil.FLOATS = (*fil.FLOATS, "r_far")
    rows = []
    for capability in (80, 90, 120):
        for name, kernel, constexprs, warps in census_variants():
            row = {"sm": capability, "variant": name, "warps": warps}
            try:
                compiled = triton.compile(fil._source(kernel, constexprs), target=GPUTarget("cuda", capability, 32),
                                          options={"num_warps": warps})
                ptx = compiled.asm["ptx"]
                row["ptx_atomics"] = len(re.findall(r"\batom\.global|\bred\.global", ptx))
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
                row["error"] = f"{type(error).__name__}: {error}"[:600]
            rows.append(row)
            print(json.dumps(row), flush=True)
    (output / "census.json").write_text(json.dumps({"tool": "nested_radial_lab --census",
                                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                                    "triton": triton.__version__, "rows": rows}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--jitter", type=float, default=0.07)
    p.add_argument("--pieces", type=int, default=896)
    p.add_argument("--r-far", type=float, default=3.15)
    p.add_argument("--rounds", type=int, default=41)
    p.add_argument("--repeat", type=int, default=10)
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

    import first_interaction_lab as fil
    import numpy as np
    import torch
    import triton
    from tabled_sum_lab import edges_of

    from fastmace.kernels.triton import tabled_outer_sum as tos

    if args.watchdog:
        fil.start_watchdog(log)
    forward_k, _, backward_k, _, _ = fil.kernels()
    forward_s, backward_s = kernels()

    key, pair, dist, slots, pairs = edges_of(args.structure, args.cutoff, args.jitter)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    r_lo, r_max, n_bins = 0.5, args.cutoff, 4 * args.pieces
    width = (r_max - r_lo) / args.pieces
    # r_far on a piece start, so a table zero in columns HALF: from r_far is exactly what the split kernel assumes
    first_far = int(np.ceil((args.r_far - r_lo) / width - 1e-9))
    r_far = float(np.float32(r_lo + width * first_far))
    k, d, half = 64, 16, 32
    coef = torch.randn((pairs, args.pieces, 6, 66), generator=g)
    pcoef = torch.randn((pairs, args.pieces, 6, 66), generator=g)
    dcoef = torch.randn((pairs, args.pieces, 5, 66), generator=g)
    for table in (coef, pcoef, dcoef):
        table[:, first_far:, :, half:k] = 0.0
    coef, pcoef, dcoef = coef.to(dev), pcoef.to(dev), dcoef.to(dev)
    starts = (r_lo + width * torch.arange(args.pieces, dtype=torch.float64)).float().to(dev)
    scales = torch.full((args.pieces,), 2.0 / width).to(dev)
    bins = (torch.arange(n_bins) * args.pieces // n_bins).to(torch.int32).to(dev)
    bin_scale = n_bins / (r_max - r_lo)
    edges = len(key)
    r = torch.from_numpy(dist.astype(np.float32)).to(dev).contiguous()
    pair_t = torch.from_numpy(pair.astype(np.int32)).to(dev)
    key_t = torch.from_numpy(key.astype(np.int64)).to(dev)
    y = torch.randn((edges, d), generator=g).to(dev)
    grad = torch.randn((slots, d * k), generator=g).to(dev)
    grad_density = torch.randn((edges,), generator=g).to(dev)
    far_fraction = float((r >= r_far).float().mean())
    log(f"{torch.cuda.get_device_name()}: {edges} edges, {slots} slots, r_far {r_far:.4f} (piece {first_far}), "
        f"far fraction {far_fraction:.4f}")
    fwd_args = (n_bins, float(r_lo), float(bin_scale), float(r_max))

    def fwd(split):
        out = torch.zeros((slots, d * k), device=dev)
        density = r.new_empty((edges,))
        grid = (triton.cdiv(edges, tos.FORWARD_CHUNK),)
        common = (y, key_t, r, pair_t, coef, starts, scales, bins, out, density, edges, args.pieces, *fwd_args)
        if split:
            forward_s[grid](*common, r_far, K=k, D=d, INNER=1, TERMS=6, J=66, CHUNK=tos.FORWARD_CHUNK, HALF=half,
                            BLOCK_D=16, num_warps=tos.FORWARD_WARPS)
        else:
            forward_k[grid](*common, 0, K=k, D=d, INNER=1, TERMS=6, J=66, CHUNK=tos.FORWARD_CHUNK, BLOCK_K=64,
                            BLOCK_D=16, SLOTS=True, ONE=False, num_warps=tos.FORWARD_WARPS)
        return out, density

    def bwd(power, split):
        dr = r.new_empty((edges,))
        grad_y = torch.empty_like(y)
        grid = (triton.cdiv(edges, tos.CHUNK),)
        common = (grad, y, key_t, r, pair_t, dcoef, starts, scales, bins, grad_density, dr, grad_y, edges, edges,
                  args.pieces, *fwd_args)
        vc = pcoef if power else coef
        if split:
            backward_s[grid](*common, r_far, K=k, D=d, INNER=1, TERMS=5, J=66, CHUNK=tos.CHUNK, HALF=half,
                             BLOCK_D=16, coef_ptr=vc, VTERMS=6, POWER=power, num_warps=tos.WARPS)
        else:
            backward_k[grid](*common, K=k, D=d, INNER=1, TERMS=5, J=66, CHUNK=tos.CHUNK, BLOCK_K=64, BLOCK_D=16,
                             coef_ptr=vc, VTERMS=6, POWER=power, ONE=False, num_warps=tos.WARPS)
        return dr, grad_y

    variants = {
        "fwd-copy": lambda: fwd(False),
        "fwd-split": lambda: fwd(True),
        "bwd-recompute-copy": lambda: bwd(False, False),
        "bwd-recompute-split": lambda: bwd(False, True),
        "bwd-power-copy": lambda: bwd(True, False),
        "bwd-power-split": lambda: bwd(True, True),
    }

    def err(new, ref):
        return max(float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
                   for a, b in zip(new, ref))

    checks = {name: err(variants[f"{name}-split"](), variants[f"{name}-copy"]())
              for name in ("fwd", "bwd-recompute", "bwd-power")}
    torch.cuda.synchronize()
    log(f"checks (split against the K = 64 copy, max |d| / max |ref|): {json.dumps(checks)}")

    times = {name: [] for name in variants}
    reference = {}
    for fn in variants.values():
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
    summary = {f"{name}_saving_us": rows[f"{name}-copy"]["median_us"] - rows[f"{name}-split"]["median_us"]
               for name in ("fwd", "bwd-recompute", "bwd-power")}
    log(f"summary: {json.dumps(summary)}")
    meta = {"tool": "nested_radial_lab", "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
            "edges": edges, "slots": slots, "pairs": pairs, "pieces": args.pieces, "r_far": r_far,
            "far_fraction": far_fraction, "rounds": args.rounds, "repeat": args.repeat,
            "source_commit": os.environ.get("FASTMACE_SOURCE_COMMIT")}
    (args.output / "lab.json").write_text(json.dumps({"meta": meta, "checks": checks, "rows": rows,
                                                      "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
