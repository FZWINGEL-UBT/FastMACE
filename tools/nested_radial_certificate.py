"""Nested radial basis for the first interaction: step 1, the certificate (next-steps rev2 §1, 11 November 2026).

Idea 1 of the second outside note (``D:\\Margraf\\inference-more-20261111\\ideas.md``). Per ordered species pair, an
orthonormal basis ``Q`` (64 x 64) of the first table's 64 radial columns (scaled by ``D``, each column's largest value
from 1 A) is built so that its first ``rank`` directions span the radial functions beyond ``R_FAR`` to ``tolerance``
(values and slopes; seeded with the value and slope at ``R_FAR``, so both sides meet there). The rotated table
``z = a D^-1 Q^T`` keeps all 64 columns below ``R_FAR`` and its first ``rank`` columns from ``R_FAR``; the node side
takes ``Q D`` composed into its matrices (here: into the folded weight, ``C = float32(Q D F^T)``).

This tool fits the first generic interaction's table as production does (``radial.fit``, float32), builds the bases
from it, rotates the table (float32 coefficients), evaluates it in float32 (``radial_table.evaluate``, the kernels'
recurrence), truncates from ``R_FAR`` and runs the radial certificate's comparison (``radial.certify``'s: table error
over the float32 network's error, against the float64 chain, values and slopes, whole domain and from 1 A, w columns
and the density) per pair. The unrotated table through the same path reproduces the production certificate.

CPU only. Usage::

    CUDA_VISIBLE_DEVICES= python tools/nested_radial_certificate.py --checkpoint mace-omat-0-medium.model \
        --elements Sr Ti O --tolerances 1e-7 5e-8 2e-8 --output DIR
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

R_FAR = 3.15
COLUMNS = 64
FIT_POINTS = 8193
TEST_POINTS = 8209


def _eval64(table, pair, r, derivative=False):
    """The float32 table's leading 64 columns evaluated in float64 at ``r`` (numpy float64)."""
    import torch

    from fastmace.kernels.triton.radial_table import evaluate

    coef = (table["dcoef"] if derivative else table["coef"]).double()
    rr = torch.as_tensor(r, dtype=torch.float64)
    out = evaluate(rr, torch.full(rr.shape, pair, dtype=torch.int32), coef, table["starts"].double(),
                   table["scales"].double(), table["bins"], table["r_lo"], table["bin_scale"], table["r_max"],
                   derivative=derivative)
    return out[:, :COLUMNS].numpy()


def nested_basis(table, pair, tolerance, r_far=R_FAR):
    """(Q [64, 64] orthonormal rows, D [64], far rank, far errors (value, slope)): the probe's construction
    (``nested_radial_basis.py``) with two bands, [r_lo, r_far) and [r_far, r_max)."""
    import numpy as np

    lo, hi = table["r_lo"], table["r_max"]
    rfit = np.linspace(lo, hi, FIT_POINTS, endpoint=False)
    rtest = lo + (np.arange(TEST_POINTS) + 0.38196601125) * (hi - lo) / TEST_POINTS
    fv, fd = _eval64(table, pair, rfit), _eval64(table, pair, rfit, True)
    tv, td = _eval64(table, pair, rtest), _eval64(table, pair, rtest, True)
    physical = rfit >= 1.0
    vscale = np.maximum(np.abs(fv[physical]).max(0), 1e-20)
    dscale = np.maximum(np.abs(fd[physical]).max(0), 1e-20)
    boundary = np.array([r_far])
    seed = np.vstack((_eval64(table, pair, boundary) / vscale, _eval64(table, pair, boundary, True) / vscale))
    seed /= np.maximum(np.linalg.norm(seed, axis=1, keepdims=True), 1e-300)
    _, s, b = np.linalg.svd(seed, full_matrices=False)
    base = b[: int(np.sum(s > s[0] * 1e-11))]
    rank, errors = COLUMNS, (0.0, 0.0)
    for band, (a, z) in enumerate(((r_far, hi), (lo, r_far))):
        m, mm = (rfit >= a) & (rfit < z), (rtest >= a) & (rtest < z)
        v, d = fv[m] / vscale, fd[m] / vscale
        gamma = np.linalg.norm(v) / max(np.linalg.norm(d), 1e-300)
        work = np.vstack((v, gamma * d))
        residual = work - (work @ base.T) @ base
        _, _, new = np.linalg.svd(residual, full_matrices=False)
        candidate = np.vstack((base, new[: COLUMNS - len(base)]))
        q, _ = np.linalg.qr(candidate.T)
        candidate = q.T
        if band == 1:  # the near band keeps every direction
            base = candidate
            break
        vv, dd = tv[mm] / vscale, td[mm] / vscale
        dnorm = dscale / vscale  # slopes relative to each column's largest slope from 1 A
        for k in range(len(base), COLUMNS + 1):
            bb = candidate[:k]
            e1 = float(np.max(np.abs((vv @ bb.T) @ bb - vv)))
            e2 = float(np.max(np.abs(((dd @ bb.T) @ bb - dd) / dnorm)))
            if max(e1, e2) <= tolerance:
                rank, errors = k, (e1, e2)
                break
        base = candidate[:rank]
    if base.shape[0] < COLUMNS:  # complete to a full basis
        q, _ = np.linalg.qr(np.vstack((base, np.eye(COLUMNS))).T)
        base = q.T[:COLUMNS]
    return base, vscale, rank, errors


def certificate(table, chain64, chain32, pairs, transforms, r_far=R_FAR, seed=0):
    """``radial.certify``'s comparison for the table rotated by ``transforms`` (per pair ``(Q, D, rank)`` or None:
    unrotated), evaluated in float32 and truncated to ``rank`` columns from ``r_far``; the node side composes
    ``C = float32(Q D F^T)`` (F: the folded weight's first 64 columns)."""
    import torch

    from fastmace.kernels.triton.radial_table import evaluate
    from fastmace.rewrites import radial

    generator = torch.Generator().manual_seed(seed)
    r_lo, r_max = table["r_lo"], table["r_max"]
    n = radial.CERTIFY_POINTS
    r64 = torch.cat([torch.linspace(r_lo, r_max, n + 1, dtype=torch.float64)[:-1],
                     r_lo + (r_max - r_lo) * torch.rand(n, generator=generator, dtype=torch.float64)])
    r64 = r64.to(torch.float32).to(torch.float64)
    r32 = r64.to(torch.float32)
    physical = r64 >= radial.R_PHYSICAL
    far = r32 >= r_far
    folded64 = chain64.folded  # [W, 65]
    args = (table["starts"], table["scales"], table["bins"], table["r_lo"], table["bin_scale"], table["r_max"])
    out, worst = {}, 0.0
    for index, (s, t) in enumerate(pairs):
        truth, dtruth = chain64.with_derivative(r64, s, t)
        net, dnet = chain32.with_derivative(r32, s, t)
        pair = torch.full(r32.shape, index, dtype=torch.int32)
        transform = transforms[index]
        if transform is None:
            coef, dcoef = table["coef"], table["dcoef"]
        else:
            q, d, rank = transform
            rot = torch.as_tensor(q.T / d[:, None], dtype=torch.float64)  # a D^-1 Q^T: [64, 64]
            coef = table["coef"].clone()
            dcoef = table["dcoef"].clone()
            coef[..., :COLUMNS] = (table["coef"][..., :COLUMNS].double() @ rot).float()
            dcoef[..., :COLUMNS] = (table["dcoef"][..., :COLUMNS].double() @ rot).float()
        tab = evaluate(r32, pair, coef, *args)  # float32 arithmetic, [n, 66]
        dtab = evaluate(r32, pair, dcoef, *args, derivative=True)
        if transform is not None:
            q, d, rank = transform
            compose = torch.as_tensor((q * d[None, :]) @ folded64[:, :COLUMNS].double().numpy().T).float().double()
            for g in (tab, dtab):
                g[far, rank:COLUMNS] = 0.0
            ws = [g[:, :COLUMNS].double() @ compose for g in (tab, dtab)]
        else:
            ws = [g[:, :COLUMNS].double() @ folded64[:, :COLUMNS].T for g in (tab, dtab)]

        def to_w(g):
            g = g.detach().to(torch.float64).cpu()
            return torch.cat([g[:, :-1] @ folded64.T, g[:, -1:]], dim=1)

        tabled = [torch.cat([w, g[:, -1:].double()], dim=1) for w, g in zip(ws, (tab, dtab))]
        ratios, parts = [], {}
        for name, exact, network, tw in (("value", truth, net, tabled[0]), ("slope", dtruth, dnet, tabled[1])):
            exact, network = to_w(exact), to_w(network)
            for region_name, region in (("all", slice(None)), ("physical", physical)):
                e = exact[region]
                scale = e.abs().amax(dim=0).clamp_min(1e-300)
                net_error = (network[region] - e).abs() / scale
                tab_error = (tw[region] - e).abs() / scale
                for cname, columns in (("w", slice(0, -1)), ("density", slice(-1, None))):
                    err_net, err_tab = net_error[:, columns].max().item(), tab_error[:, columns].max().item()
                    ratio = 0.0 if err_tab == 0.0 else (err_tab / err_net if err_net > 0 else float("inf"))
                    ratios.append(ratio)
                    parts[f"{name}/{region_name}/{cname}"] = {"table": err_tab, "network": err_net, "ratio": ratio}
        out[f"{s}-{t}"] = {"worst": max(ratios), "parts": parts}
        worst = max(worst, max(ratios))
    return {"worst": worst, "pairs": out}


def main(argv=None):
    import torch
    from ase.data import atomic_numbers

    from fastmace.rewrites import radial

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--elements", nargs="+", default=["Sr", "Ti", "O"])
    p.add_argument("--tolerances", nargs="+", type=float, default=[1e-7, 5e-8, 2e-8])
    p.add_argument("--r-far", type=float, default=R_FAR)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    log = (args.output / "certificate.log").open("a")

    def say(text):
        line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    zs = [int(z) for z in model.atomic_numbers]
    columns = [zs.index(atomic_numbers[e]) for e in args.elements]
    per_pair = radial._species_radial(model)
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))] if per_pair else [(0, 0)]
    r_max = float(model.r_max)
    index, interaction = radial._generic_interactions(model)[0]
    chain64 = radial._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
    chain32 = radial._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
    start = time.perf_counter()
    table = radial.fit(chain64, pairs, r_max, table_dtype=torch.float32)
    say(f"interaction {index}: fitted {table['coef'].shape} in {time.perf_counter() - start:.0f} s, "
        f"fit error {table['fit_error']:.2e}, per_pair {per_pair}, folded {tuple(chain64.folded.shape)}")
    module = radial._table_module(table, columns, torch.device("cpu"), single=not per_pair)
    production = radial.certify(module, chain64, chain32, pairs, r_max, seed=index)
    say(f"production certificate (radial.certify): worst {production['worst']:.4f}")
    base = certificate(table, chain64, chain32, pairs, [None] * len(pairs), args.r_far, seed=index)
    say(f"this path, unrotated: worst {base['worst']:.4f}")
    report = {"tool": "nested_radial_certificate", "created_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": args.checkpoint, "elements": args.elements, "pairs": [f"{s}-{t}" for s, t in pairs],
              "r_far": args.r_far, "pieces": int(table["coef"].shape[1]), "fit_error": table["fit_error"],
              "production_certificate": production["worst"], "unrotated": base, "rows": []}
    for tolerance in args.tolerances:
        transforms, ranks, errors = [], [], []
        for pair in range(len(pairs)):
            q, d, rank, err = nested_basis(table, pair, tolerance, args.r_far)
            transforms.append((q, d, rank))
            ranks.append(rank)
            errors.append(err)
        cert = certificate(table, chain64, chain32, pairs, transforms, args.r_far, seed=index)
        # the same bases with no truncation: the rotation's own float32 rounding
        full = certificate(table, chain64, chain32, pairs, [(q, d, COLUMNS) for q, d, _ in transforms],
                           args.r_far, seed=index)
        row = {"tolerance": tolerance, "ranks": ranks, "max_rank": max(ranks), "projection_errors": errors,
               "certificate": cert["worst"], "certificate_untruncated": full["worst"],
               "per_pair": {k: v["worst"] for k, v in cert["pairs"].items()}, "detail": cert["pairs"]}
        report["rows"].append(row)
        say(f"tolerance {tolerance:g}: far ranks {ranks} (max {max(ranks)}); certificate {cert['worst']:.4f} "
            f"(untruncated rotation {full['worst']:.4f}; production {production['worst']:.4f})")
        (args.output / "certificate.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
