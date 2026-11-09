"""Standalone ZBL geometry-tail falsifier.

This is deliberately a lab tool, not a FastMACE rewrite.  It compares the
current MACE ``ZBLBasis`` contract (one energy per receiver atom, with the
directed-edge 1/2 factor) and autograd's radial derivative with a one-pass
analytic value-plus-derivative implementation.  The formula is the one in
``mace/modules/radial.py``: ZBL's four exponentials and the p=6 polynomial
envelope whose radius is the sum of the ASE covalent radii.

The interesting candidate is the force-only path: it keeps the per-atom
energy output for callers that need it, but does not reduce that output to a
scalar and does not retain an autograd graph just to obtain the radial
derivative.  This file measures both forms, because omitting an energy
reduction is not a valid production change when an engine/MC caller still
consumes energy.

No FastMACE model, checkpoint, or production module is changed.  The default
input is a deterministic synthetic SrTiO3 supercell and all species are
frozen (Sr=38, Ti=22, O=8).  CUDA graph timing is optional and is skipped on a
CPU-only workstation.

Examples (in an environment containing torch)::

    python tools/dmqe_geometry_lab.py --device cpu --cells 2 --rounds 5
    python tools/dmqe_geometry_lab.py --device cuda --cells 10 --rounds 31 \
        --output results/dmqe-geometry-lab

The current reference can optionally be compiled with ``torch.compile``;
``--no-compile`` is useful on older torch or CPU installations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# These are the three entries used by ASE's covalent_radii table for the
# frozen SrTiO3 species.  The source ZBLBasis indexes the complete ASE table;
# keeping only these entries makes this lab independent of ASE.
_COVALENT_RADIUS = {8: 0.66, 22: 1.60, 38: 1.95}
_SPECIES = (38, 22, 8)
_BASIS_SPECIES = (38, 22, 8, 8, 8)


def _torch():
    """Import torch lazily so ``--help`` and syntax checks work without it."""

    try:
        # isort: split — fastmace must load before torch in this repository.
        import fastmace  # noqa: F401, I001  (before torch: preserve the repo's soft file-lock ordering)
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the Windows checkout
        raise SystemExit("dmqe_geometry_lab needs torch; run it in the WSL/Ada environment") from exc
    return torch


def _probe_mace_source():
    """Record whether this workstation exposes the installed ZBLBasis source."""

    try:
        import mace.modules.radial as radial

        source = inspect.getsource(radial.ZBLBasis)
        return {
            "status": "ok",
            "module": getattr(radial, "__file__", None),
            "zblbasis_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "class_source_has_p6_envelope": "PolynomialCutoff.calculate_envelope" in source,
            "class_source_has_half_factor": "0.5 * v_edges" in source,
        }
    except Exception as exc:  # pragma: no cover - depends on the WSL environment
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def synthetic_srtio3(cells: int, cutoff: float, *, dtype, device):
    """Make a periodic perovskite graph without ASE or a neighbour-list package.

    The five-atom cubic basis is Sr at the corner, Ti at the body centre and
    three O atoms on the face centres.  Periodic cell shifts in {-1,0,1} are
    enough for the six-Angstrom lab cutoff at the default 3.905-Angstrom cell.
    Edges are directed and sorted by receiver; the ZBL 1/2 factor then avoids
    double-counting the directed pair, matching ``ZBLBasis``.
    """

    torch = _torch()
    if cells < 2:
        raise ValueError("cells must be >= 2 so periodic image shifts are unique")
    a0 = 3.905
    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=dtype,
        device=device,
    )
    species_basis = torch.tensor(_BASIS_SPECIES, dtype=torch.int64, device=device)
    grid = torch.arange(cells, dtype=torch.int64, device=device)
    gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing="ij")
    cell = torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3)
    ncell = int(cell.shape[0])
    positions = (cell[:, None, :].to(dtype) + basis[None, :, :]) * a0
    positions = positions.reshape(-1, 3)
    species = species_basis.repeat(ncell)

    edge_receivers = []
    edge_senders = []
    edge_distances = []
    # The source convention is edge_index[0] = sender and edge_index[1] =
    # receiver.  ``receiver`` is deliberately the second index here.
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-2, -1, 0, 1, 2):
                shift = torch.tensor((dx, dy, dz), dtype=torch.int64, device=device)
                sender_cell = (cell + shift) % cells
                for br in range(5):
                    receiver = torch.arange(ncell, device=device, dtype=torch.int64) * 5 + br
                    for bs in range(5):
                        sender = sender_cell.new_tensor(sender_cell[:, 0] * (cells * cells) * 5)
                        sender = sender + sender_cell[:, 1] * cells * 5 + sender_cell[:, 2] * 5 + bs
                        # ``shift`` is the requested sender-cell offset, not
                        # a whole-box image.  This remains correct for both
                        # interior and wrapped receiver cells.
                        dr = (shift.to(dtype) + basis[bs] - basis[br]) * a0
                        dr = dr.expand(ncell, 3)
                        dist = torch.linalg.vector_norm(dr, dim=1)
                        keep = (dist > 1.0e-9) & (dist < cutoff)
                        if bool(keep.any()):
                            edge_receivers.append(receiver[keep])
                            edge_senders.append(sender[keep])
                            edge_distances.append(dist[keep])
    receiver = torch.cat(edge_receivers)
    sender = torch.cat(edge_senders)
    distance = torch.cat(edge_distances)
    # Receiver-major order is the production list's useful geometry order.
    order = torch.argsort(receiver * positions.shape[0] + sender, stable=True)
    receiver, sender, distance = receiver[order], sender[order], distance[order]
    z_sender, z_receiver = species[sender], species[receiver]
    return positions, species, sender, receiver, distance, z_sender, z_receiver


def _pair_radii(z_sender, z_receiver, *, dtype):
    torch = _torch()
    # The lab intentionally rejects unknown species rather than silently using
    # a different covalent-radius table from a user's installed ASE version.
    values = torch.tensor([_COVALENT_RADIUS[int(z)] for z in _SPECIES], dtype=dtype, device=z_sender.device)
    lookup = {z: values[i] for i, z in enumerate(_SPECIES)}
    return torch.stack((lookup[8] * (z_sender == 8), lookup[22] * (z_sender == 22), lookup[38] * (z_sender == 38))).sum(
        dim=0
    ) + torch.stack(
        (lookup[8] * (z_receiver == 8), lookup[22] * (z_receiver == 22), lookup[38] * (z_receiver == 38))
    ).sum(dim=0)


def zbl_edge_value_and_dr(r, z_sender, z_receiver, *, p: int = 6):
    """Return ``(v_edges, d v_edges / d r)`` in the current ZBL formula.

    The constants and operation order mirror ``mace.modules.radial.ZBLBasis``:
    ``a = .4543*.529/(Z_u^.3+Z_v^.3)``, four exponentials, ``14.3996`` and
    the per-pair p=6 envelope.  ``dr`` is analytic and represents upstream
    gradient one.  It is zero on the inactive side of the envelope, as the
    source's ``envelope * (x < r_max)`` is.
    """

    torch = _torch()
    dtype = r.dtype
    zu = z_sender.to(dtype)
    zv = z_receiver.to(dtype)
    a = (0.4543 * 0.529) / (zu.pow(0.300) + zv.pow(0.300))
    r_over_a = r / a
    coeff = (0.1818, 0.5099, 0.2802, 0.02817)
    decay = (3.2, 0.9423, 0.4028, 0.2016)
    phi = sum(c * torch.exp(-b * r_over_a) for c, b in zip(coeff, decay))
    dphi = sum(c * (-b / a) * torch.exp(-b * r_over_a) for c, b in zip(coeff, decay))
    # ``ZBLBasis`` multiplies the int64 atomic numbers before converting the
    # product to the model dtype; retain that order rather than doing two
    # floating multiplies.
    charge_product = (z_sender * z_receiver).to(dtype)
    coulomb = (14.3996 * charge_product) / r
    dcoulomb = -(14.3996 * charge_product) / r.pow(2)
    bare = coulomb * phi
    dbare = dcoulomb * phi + coulomb * dphi

    r_max = _pair_radii(z_sender, z_receiver, dtype=dtype)
    q = r / r_max
    # PolynomialCutoff.calculate_envelope(x, r_max, p), specialized only by
    # the integer p while retaining the source's operation order.
    env = 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * q.pow(p)
    env = env + p * (p + 2.0) * q.pow(p + 1)
    env = env - (p * (p + 1.0) / 2.0) * q.pow(p + 2)
    denv = (
        -((p + 1.0) * (p + 2.0) / 2.0) * p * q.pow(p - 1)
        + p * (p + 2.0) * (p + 1.0) * q.pow(p)
        - (p * (p + 1.0) / 2.0) * (p + 2.0) * q.pow(p + 1)
    ) / r_max
    active = r < r_max
    # MACE's neighbour list is directed, so each edge carries half of the
    # pair energy.  ``where`` also makes the candidate's inactive derivative
    # exactly zero, matching the source's multiplication by a boolean mask.
    value = 0.5 * torch.where(active, bare * env, torch.zeros_like(bare))
    dr = 0.5 * torch.where(active, dbare * env + bare * denv, torch.zeros_like(bare))
    return value, dr


def current_node_energy(r, z_sender, z_receiver, receiver, n_atoms):
    """Current reference contract: edge ZBL values reduced to receiver atoms."""

    torch = _torch()
    value, _ = zbl_edge_value_and_dr(r, z_sender, z_receiver)
    node = torch.zeros((n_atoms,), dtype=r.dtype, device=r.device)
    return node.index_add(0, receiver, value)


def reference_value_and_autograd(
    r, z_sender, z_receiver, receiver, n_atoms, node_fn=current_node_energy, *, upstream_scale: float = 1.0
):
    """Current value plus autograd's radial derivative (upstream scalar 1)."""

    rr = r.detach().clone().requires_grad_(True)
    node = node_fn(rr, z_sender, z_receiver, receiver, n_atoms)
    # ScaleShiftMACE applies this scalar to pair+interaction node energies
    # before its graph reduction.  The default one is the isolated constant-1
    # ZBL claim; a checkpoint scale can be passed to falsify the production
    # upstream instead of accidentally assuming it is one.
    total = node.sum() * upstream_scale
    dr = _torch().autograd.grad(total, rr, create_graph=False, retain_graph=False)[0]
    return node.detach(), dr.detach(), total.detach()


def fused_value_and_dr(
    r, z_sender, z_receiver, receiver, n_atoms, *, reduce_total: bool = False, upstream_scale: float = 1.0
):
    """Candidate one-pass value+radial-derivative path.

    ``reduce_total=False`` is the force-only contract.  ``reduce_total=True``
    retains the energy reduction so its cost can be separated from derivative
    work; it is not a claim that production can remove the reduction.
    """

    torch = _torch()
    value, dr = zbl_edge_value_and_dr(r, z_sender, z_receiver)
    node = torch.zeros((n_atoms,), dtype=r.dtype, device=r.device).index_add(0, receiver, value)
    total = node.sum() * upstream_scale if reduce_total else None
    return node, dr * upstream_scale, total


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]


def _time_cpu(fn: Callable[[], object], rounds: int, warmup: int):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return {
        "median_us": statistics.median(samples),
        "p10_us": _percentile(samples, 0.1),
        "p90_us": _percentile(samples, 0.9),
        "samples_us": samples,
    }


def _time_cuda(fn: Callable[[], object], rounds: int, warmup: int):
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop) * 1e3))
    return {
        "median_us": statistics.median(samples),
        "p10_us": _percentile(samples, 0.1),
        "p90_us": _percentile(samples, 0.9),
        "samples_us": samples,
    }


def _try_compile(node_fn):
    torch = _torch()
    if not hasattr(torch, "compile"):
        return node_fn, "unavailable"
    try:
        compiled = torch.compile(node_fn, fullgraph=True, dynamic=False)
        # Trigger compilation outside the measurement.  The caller invokes it
        # once with real operands immediately after this function returns.
        return compiled, "requested"
    except Exception as exc:  # pragma: no cover - version/device dependent
        return node_fn, f"failed: {type(exc).__name__}: {exc}"


def _cuda_graph_ab(
    r, z_sender, z_receiver, receiver, n_atoms, *, rounds: int, node_fn, fused_fn, upstream_scale: float
):
    """Capture current autograd A and fused analytic B, then replay A/B."""

    torch = _torch()
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable"}
    # A and B have independent static inputs so that capture and replay do not
    # accidentally share autograd state.  The graph output buffers make the
    # contract visible even though this lab does not attach forces to atoms.
    ra = r.detach().clone().requires_grad_(True)
    rb = r.detach().clone()
    ref_energy = torch.empty((n_atoms,), dtype=r.dtype, device=r.device)
    ref_dr = torch.empty_like(ra)
    fused_energy = torch.empty_like(ref_energy)
    fused_dr = torch.empty_like(rb)
    ref_total = torch.empty((), dtype=r.dtype, device=r.device)
    fused_total = torch.empty_like(ref_total)
    ga, gb, gc = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()

    def current_graph_body():
        node = node_fn(ra, z_sender, z_receiver, receiver, n_atoms)
        total = node.sum() * upstream_scale
        grad = torch.autograd.grad(total, ra, create_graph=False, retain_graph=False)[0]
        ref_energy.copy_(node)
        ref_dr.copy_(grad)
        ref_total.copy_(total)

    def fused_graph_body(preserve_total=False):
        with torch.no_grad():
            node, dr = fused_fn(rb, z_sender, z_receiver, receiver, n_atoms)
        fused_energy.copy_(node)
        fused_dr.copy_(dr)
        if preserve_total:
            fused_total.copy_(node.sum() * upstream_scale)

    try:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                current_graph_body()
                fused_graph_body()
                fused_graph_body(True)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(ga, stream=capture_stream):
            current_graph_body()
        with torch.cuda.graph(gb, stream=capture_stream):
            fused_graph_body()
        with torch.cuda.graph(gc, stream=capture_stream):
            fused_graph_body(True)
        # Alternate replay order; these are lab graphs, not independent model builds.
        torch.cuda.synchronize()
        samples = {"current": [], "force_only": [], "energy_preserved": []}
        entries = [("current", ga), ("force_only", gb), ("energy_preserved", gc)]
        for index in range(rounds):
            for name, graph in entries if index % 2 == 0 else entries[::-1]:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graph.replay()
                stop.record()
                stop.synchronize()
                samples[name].append(float(start.elapsed_time(stop) * 1e3 / 50))
        samples_a, samples_b = samples["current"], samples["force_only"]
        return {
            "status": "ok",
            "current_autograd_median_us": statistics.median(samples_a),
            "fused_force_only_median_us": statistics.median(samples_b),
            "fused_energy_preserved_median_us": statistics.median(samples["energy_preserved"]),
            "energy_preserved_samples_us": samples["energy_preserved"],
            "replays_per_sample": 50,
            "current_samples_us": samples_a,
            "fused_samples_us": samples_b,
            "outputs_max_abs": {
                "energy": float((ref_energy - fused_energy).abs().max().item()),
                "dr": float((ref_dr - fused_dr).abs().max().item()),
                "total": float((ref_total - fused_total).abs().item()),
            },
        }
    except Exception as exc:  # pragma: no cover - graph support varies by torch/GPU
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cells", type=int, default=4, help="per-axis cells (5*cells^3 atoms; default 320)")
    parser.add_argument("--cutoff", type=float, default=6.0, help="synthetic graph search cutoff in Angstrom")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument(
        "--upstream-scale",
        type=float,
        default=1.0,
        help="constant upstream multiplier (ScaleShiftMACE scale; 1 isolates ZBL)",
    )
    parser.add_argument("--no-compile", action="store_true", help="do not try torch.compile for current reference")
    parser.add_argument("--no-graphs", action="store_true", help="skip the CUDA graph A/B even on CUDA")
    parser.add_argument("--output", type=Path, help="optional directory for dmqe_geometry_lab.json")
    args = parser.parse_args(argv)
    torch = _torch()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    dtype = torch.float32
    positions, species, sender, receiver, distance, z_sender, z_receiver = synthetic_srtio3(
        args.cells, args.cutoff, dtype=dtype, device=device
    )
    n_atoms, edges = int(positions.shape[0]), int(distance.numel())
    # Include near-boundary and near-origin values in a small correctness set;
    # the production-like graph remains the measured input.
    checks = torch.tensor([0.20, 0.45, 0.90, 1.30, 2.25, 3.0, 4.0, 5.8], dtype=torch.float64, device=device)
    check_zu = torch.tensor([38, 38, 22, 8, 8, 22, 38, 8], dtype=torch.int64, device=device)
    check_zv = torch.tensor([8, 22, 8, 8, 22, 38, 38, 22], dtype=torch.int64, device=device)
    check_r = checks.detach().requires_grad_(True)
    ref_value, ref_dr, _ = reference_value_and_autograd(
        check_r,
        check_zu,
        check_zv,
        torch.arange(len(checks), device=device),
        len(checks),
        upstream_scale=args.upstream_scale,
    )
    fused_value, fused_dr = zbl_edge_value_and_dr(checks, check_zu, check_zv)
    fused_dr = fused_dr * args.upstream_scale
    correctness = {
        "value_max_abs": float((ref_value - fused_value).abs().max().item()),
        "value_max_rel": float(((ref_value - fused_value).abs() / ref_value.abs().clamp_min(1e-30)).max().item()),
        "dr_max_abs": float((ref_dr - fused_dr).abs().max().item()),
        "dr_max_rel": float(((ref_dr - fused_dr).abs() / ref_dr.abs().clamp_min(1e-30)).max().item()),
        "dtype": "float64",
        "p6": True,
        "species": {"sender": check_zu.tolist(), "receiver": check_zv.tolist()},
    }
    correctness["pass"] = correctness["value_max_abs"] <= 5e-11 and correctness["dr_max_abs"] <= 5e-10

    r = distance.detach().clone().requires_grad_(True)
    # Prefer the actually installed MACE ZBLBasis for the current side.  The
    # closed-form mirror remains the fallback so the lab is still useful in a
    # minimal torch-only environment (and is the candidate implementation).
    node_fn = current_node_energy
    reference_implementation = "closed_form_mirror"
    try:
        from mace.modules.radial import ZBLBasis

        zbl_module = ZBLBasis(p=6).to(device=device, dtype=dtype)
        node_attrs = torch.stack([(species == z).to(dtype) for z in _SPECIES], dim=1)
        edge_index = torch.stack((sender, receiver), dim=0)
        atomic_numbers = torch.tensor(_SPECIES, dtype=torch.int64, device=device)

        def mace_node_energy(rr, _zs, _zr, _recv, _n):
            # MACE's source consumes lengths with a trailing singleton dim;
            # its output is already the per-receiver node-energy contract.
            return zbl_module(rr.unsqueeze(-1), node_attrs, edge_index, atomic_numbers)

        node_fn = mace_node_energy
        reference_implementation = "mace.modules.radial.ZBLBasis"
    except Exception as exc:  # pragma: no cover - environment dependent
        reference_implementation = f"closed_form_mirror ({type(exc).__name__}: {exc})"
    uncompiled_node_fn = node_fn
    fused_fn = None
    compile_status = "disabled"
    fused_compile_status = "disabled"
    if not args.no_compile:
        node_fn, compile_status = _try_compile(node_fn)

        def _fused_compilable(rr, zs, zr, recv, n):
            node, dr, _ = fused_value_and_dr(
                rr, zs, zr, recv, n, reduce_total=False, upstream_scale=args.upstream_scale
            )
            return node, dr

        fused_fn, fused_compile_status = _try_compile(_fused_compilable)
    # Force compilation once, before timing or graph capture.  A compile
    # failure is reported but does not prevent the direct CPU/GPU falsifier.
    if compile_status == "requested":
        try:
            node_fn(r, z_sender, z_receiver, receiver, n_atoms)
            if device.type == "cuda":
                torch.cuda.synchronize()
            compile_status = "ok"
        except Exception as exc:  # pragma: no cover - torch version dependent
            node_fn, compile_status = uncompiled_node_fn, f"failed: {type(exc).__name__}: {exc}"
    if fused_compile_status == "requested":
        try:
            fused_fn(r, z_sender, z_receiver, receiver, n_atoms)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fused_compile_status = "ok"
        except Exception as exc:  # pragma: no cover - torch version dependent
            fused_fn, fused_compile_status = None, f"failed: {type(exc).__name__}: {exc}"
    if fused_fn is None:

        def fused_fn(rr, zs, zr, recv, n):
            node, dr, _ = fused_value_and_dr(
                rr, zs, zr, recv, n, reduce_total=False, upstream_scale=args.upstream_scale
            )
            return node, dr

    def current_call():
        return reference_value_and_autograd(
            r, z_sender, z_receiver, receiver, n_atoms, node_fn=node_fn, upstream_scale=args.upstream_scale
        )

    def fused_force_call():
        node, dr = fused_fn(r, z_sender, z_receiver, receiver, n_atoms)
        return node, dr, None

    def fused_energy_call():
        node, dr = fused_fn(r, z_sender, z_receiver, receiver, n_atoms)
        return node, dr, node.sum() * args.upstream_scale

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    timings = {
        "current_autograd_value_plus_dr": timer(current_call, args.rounds, args.warmup),
        "fused_force_only_value_plus_dr": timer(fused_force_call, args.rounds, args.warmup),
        "fused_with_total_reduction": timer(fused_energy_call, args.rounds, args.warmup),
    }
    # Compare production-like graph outputs once in the same dtype.  Autograd
    # gets a fresh leaf in the helper, so this does not retain measurement state.
    ref_node, ref_graph_dr, ref_total = current_call()
    candidate_node, candidate_dr, candidate_total = fused_energy_call()
    output_check = {
        "node_energy_max_abs": float((ref_node - candidate_node).abs().max().item()),
        "node_energy_max_rel": float(
            ((ref_node - candidate_node).abs() / ref_node.abs().clamp_min(1e-30)).max().item()
        ),
        "radial_dr_max_abs": float((ref_graph_dr - candidate_dr).abs().max().item()),
        "radial_dr_max_rel": float(
            ((ref_graph_dr - candidate_dr).abs() / ref_graph_dr.abs().clamp_min(1e-30)).max().item()
        ),
        "total_energy_max_abs": float((ref_total - candidate_total).abs().item()),
        "force_only_total_is_none": fused_force_call()[2] is None,
    }
    output_check["pass"] = (
        output_check["node_energy_max_abs"] <= 2e-4
        and output_check["radial_dr_max_abs"] <= 2e-3
        and output_check["total_energy_max_abs"] <= 2e-3
    )
    graphs = {"status": "skipped", "reason": "disabled" if args.no_graphs else "CPU"}
    if device.type == "cuda" and not args.no_graphs:
        graphs = _cuda_graph_ab(
            r.detach(),
            z_sender,
            z_receiver,
            receiver,
            n_atoms,
            rounds=args.rounds,
            node_fn=node_fn,
            fused_fn=fused_fn,
            upstream_scale=args.upstream_scale,
        )

    result = {
        "tool": "dmqe_geometry_lab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "formula": {
            "source": "mace.modules.radial.ZBLBasis",
            "p": 6,
            "coeff": [0.1818, 0.5099, 0.2802, 0.02817],
            "decay": [3.2, 0.9423, 0.4028, 0.2016],
            "a_prefactor": 0.4543,
            "bohr": 0.529,
            "coulomb": 14.3996,
            "covalent_radii_A": _COVALENT_RADIUS,
            "directed_edge_half_factor": 0.5,
        },
        "mace_source_probe": _probe_mace_source(),
        "upstream_scale": args.upstream_scale,
        "synthetic_srtio3": {
            "cells": args.cells,
            "atoms": n_atoms,
            "edges": edges,
            "cutoff_A": args.cutoff,
            "frozen_species": _SPECIES,
        },
        "compile_current": compile_status,
        "compile_fused": fused_compile_status,
        "current_reference": reference_implementation,
        "correctness_float64": correctness,
        "output_contract_float32": output_check,
        "timings_float32_us": timings,
        "cuda_graph_ab": graphs,
        "gate": {
            "float64_value_and_autograd": correctness["pass"],
            "float32_node_and_radial_contract": output_check["pass"],
            "speed_claim_allowed": correctness["pass"]
            and output_check["pass"]
            and graphs.get("status") in ("ok", "skipped"),
            "verdict": "READY_FOR_GPU_AB" if correctness["pass"] and output_check["pass"] else "BLOCKED_BY_CORRECTNESS",
        },
        "interpretation": {
            "force_only": (
                "fused_force_only_value_plus_dr returns raw per-atom ZBL energy and scaled per-edge dE/dr "
                "but intentionally no scalar total; this is not production-safe for energy/MC callers."
            ),
            "scale_shift": (
                "ScaleShiftMACE applies atomic_inter_scale to pair+interaction node energies before the scalar "
                "graph reduction; atomic_inter_shift has zero coordinate derivative. Multiple heads and "
                "nonconstant upstreams are outside this falsifier."
            ),
            "additional_forward_cost": (
                "compare fused_force_only_value_plus_dr against current_autograd_value_plus_dr and "
                "fused_with_total_reduction; the latter retains the scalar reduction so work removal is not hidden."
            ),
            "adoption": "lab/falsifier only; no production adoption or full geometry-tail fusion is claimed.",
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "dmqe_geometry_lab.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
