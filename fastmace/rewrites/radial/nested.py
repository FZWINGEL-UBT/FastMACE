"""The nested radial basis of the first interaction (next-steps rev2 §1, 11 November 2026;
``results/nested-radial-20261111``).

Beyond ``R_FAR`` the first interaction's 64 radial columns ``a(r)`` of a species pair are close to a 32-dimensional
space. Per table row (species pair), an orthonormal basis ``Q`` [64, 64] whose first ``FAR_COLUMNS`` directions span
the row's values and slopes from ``R_FAR`` (to ``TOLERANCE`` of each column's scale ``D``, the largest value from
``R_PHYSICAL``) rotates the row: ``z = a D^-1 Q^T``, ``a = z M`` with ``M = Q D``. The table stores ``z`` (its
first 64 channels rotated; the c column and the density column as they were). The tabled outer sum sums ``y z^T``
per slot and, from ``R_FAR``, only ``z``'s first ``FAR_COLUMNS`` channels (``tabled_outer_sum``'s split walks); the
node side composes ``M`` into ``compose_skip``'s per-receiver-species weights at setup, so the message is unchanged
up to the truncation and float32 rounding.

The construction is the probe's (``nested_radial_basis.py``; ``tools/nested_radial_certificate.py``): seeded with the
row's value and slope at ``R_FAR`` (both sides meet there), then the far band's residual SVD, completed to 64. The
far width is fixed at ``FAR_COLUMNS`` (the kernel's half width); ``install`` refuses a table where any row's far
projection error exceeds ``TOLERANCE``. Records: certificate 0.3011 against the production 0.3011 (OMAT-0 medium,
Sr Ti O, every row at 32 far columns, far error at most 5.4e-8).

Rounding class (a truncation under the radial certificate): the force/stress gate decides adoption.
``NESTED_RADIAL``: None: on for ``NESTED_RADIAL_GPUS``; True / False: an A/B side's choice. Read at build
(``precision._route_compose_skip``), only where ``COMPOSE_SKIP`` runs and the tabled forward is on; with it the
tabled backward takes the power basis (``POWER_BACKWARD``'s walk), whose split keeps the gain.
"""

from __future__ import annotations

from fastmace.kernels.gpu_models import GPUModels

NESTED_RADIAL = None
NESTED_RADIAL_GPUS = GPUModels(())
R_FAR = 3.15  # Å
COLUMNS = 64
FAR_COLUMNS = 32
TOLERANCE = 1e-7
R_PHYSICAL = 1.0  # Å: the column scales' region (radial.R_PHYSICAL)
FIT_POINTS = 8193
TEST_POINTS = 8209


def nested_default(name=None) -> bool:
    """``NESTED_RADIAL`` (None: ``name`` in ``NESTED_RADIAL_GPUS``)."""
    on = NESTED_RADIAL
    if on is None:
        on = name in NESTED_RADIAL_GPUS
    return bool(on)


def _evaluate(coef, starts, scales, bins, r_lo, bin_scale, r_max, row, r, derivative=False):
    import torch

    from fastmace.kernels.triton.radial_table import evaluate

    rr = torch.as_tensor(r, dtype=torch.float64)
    out = evaluate(rr, torch.full(rr.shape, row, dtype=torch.int32), coef, starts, scales, bins, r_lo, bin_scale,
                   r_max, derivative=derivative)
    return out[:, :COLUMNS].numpy()


def basis(coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, row, r_far=R_FAR, far=FAR_COLUMNS):
    """``(Q [64, 64] orthonormal rows, D [64], (value error, slope error))`` of table row ``row`` (float64 CPU
    tensors in; numpy out): the first ``far`` rows of Q span the row's values and slopes from ``r_far``; the errors are
    the largest far projection errors, per column scale (slopes per each column's largest slope from 1 Å)."""
    import numpy as np

    args = (starts, scales, bins, r_lo, bin_scale, r_max, row)
    rfit = np.linspace(r_lo, r_max, FIT_POINTS, endpoint=False)
    rtest = r_lo + (np.arange(TEST_POINTS) + 0.38196601125) * (r_max - r_lo) / TEST_POINTS
    fv, fd = _evaluate(coef, *args, rfit), _evaluate(dcoef, *args, rfit, True)
    tv, td = _evaluate(coef, *args, rtest), _evaluate(dcoef, *args, rtest, True)
    physical = rfit >= R_PHYSICAL
    vscale = np.maximum(np.abs(fv[physical]).max(0), 1e-20)
    dscale = np.maximum(np.abs(fd[physical]).max(0), 1e-20)
    boundary = np.array([r_far])
    seed = np.vstack((_evaluate(coef, *args, boundary) / vscale, _evaluate(dcoef, *args, boundary, True) / vscale))
    seed /= np.maximum(np.linalg.norm(seed, axis=1, keepdims=True), 1e-300)
    _, s, b = np.linalg.svd(seed, full_matrices=False)
    base = b[: int(np.sum(s > s[0] * 1e-11))]
    errors = (0.0, 0.0)
    for band, (lo, hi) in enumerate(((r_far, r_max), (r_lo, r_far))):
        m, mm = (rfit >= lo) & (rfit < hi), (rtest >= lo) & (rtest < hi)
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
        bb = candidate[:far]
        vv, dd = tv[mm] / vscale, td[mm] / vscale
        e1 = float(np.max(np.abs((vv @ bb.T) @ bb - vv))) if len(vv) else 0.0
        e2 = float(np.max(np.abs(((dd @ bb.T) @ bb - dd) / (dscale / vscale)))) if len(dd) else 0.0
        errors = (e1, e2)
        base = bb
    if base.shape[0] < COLUMNS:  # complete to a full basis
        q, _ = np.linalg.qr(np.vstack((base, np.eye(COLUMNS))).T)
        base = q.T[:COLUMNS]
    return base, vscale, errors


def install(table, r_far=R_FAR):
    """Give ``table`` (``radial._table_module``) its nested copies, once: ``nested_coef``, ``nested_dcoef``,
    ``nested_pcoef`` (the rows rotated, in the table's dtype), ``nested_map`` (``M = Q D`` per row, [P, 64, 64]
    float64) and ``nested_r_far``. Returns whether the table serves: 64 or more channels, and every row's far error
    at most ``TOLERANCE`` (else no copies, and ``nested_errors`` says why)."""
    import torch

    from fastmace.rewrites.radial import power_basis

    if hasattr(table, "nested_ok"):
        return table.nested_ok
    coef = table.coef.detach().cpu().double()
    dcoef = table.dcoef.detach().cpu().double()
    starts, scales, bins = table.starts.cpu().double(), table.scales.cpu().double(), table.bins.cpu()
    table.nested_errors = []
    if coef.shape[-1] < COLUMNS + 2:
        table.nested_ok = False
        return False
    rows, maps, done = coef.shape[0], [], {}
    for row in range(rows):
        signature = coef[row].numpy().tobytes()  # mirror rows (SYMMETRIC_PAIRS) and repeats: one basis
        if signature not in done:
            done[signature] = basis(coef, dcoef, starts, scales, bins, table.r_lo, table.bin_scale, table.r_max, row,
                                    r_far)
        q, d, errors = done[signature]
        table.nested_errors.append(errors)
        maps.append((torch.from_numpy(q), torch.from_numpy(d)))
    if max(max(e) for e in table.nested_errors) > TOLERANCE:
        table.nested_ok = False
        return False
    q = torch.stack([m[0] for m in maps])  # [P, 64, 64]
    d = torch.stack([m[1] for m in maps])  # [P, 64]
    into = q.transpose(1, 2) / d[:, :, None]  # z = a (Q^T / D[k]): [P, k, k']
    rotated, drotated = coef.clone(), dcoef.clone()
    rotated[..., :COLUMNS] = torch.einsum("pqtk,pkl->pqtl", coef[..., :COLUMNS], into)
    drotated[..., :COLUMNS] = torch.einsum("pqtk,pkl->pqtl", dcoef[..., :COLUMNS], into)
    device, dtype = table.coef.device, table.coef.dtype
    table.register_buffer("nested_coef", rotated.to(device=device, dtype=dtype), persistent=False)
    table.register_buffer("nested_dcoef", drotated.to(device=device, dtype=dtype), persistent=False)
    table.register_buffer("nested_pcoef", power_basis(table.nested_coef), persistent=False)
    table.register_buffer("nested_map", (q * d[:, None, :]).to(device), persistent=False)  # a = z M
    table.nested_r_far = float(r_far)
    table.nested_ok = True
    return True


def rotation(table, receivers, senders):
    """``[receivers, senders * 64, senders * 64]`` float64 block-diagonal maps: block (z, z) of receiver species s is
    ``M`` of the row the table reads for edges z -> s (``table.pair``'s rule; superset indices)."""
    import torch

    m = table.nested_map
    out = torch.zeros((receivers, senders * COLUMNS, senders * COLUMNS), dtype=torch.float64, device=m.device)
    for s in range(receivers):
        for z in range(senders):
            if table.single:
                row = 0
            else:
                a, b = (min(z, s), max(z, s)) if table.symmetric else (z, s)
                row = a * table.species + b
            out[s, z * COLUMNS:(z + 1) * COLUMNS, z * COLUMNS:(z + 1) * COLUMNS] = m[row]
    return out


__all__ = ["COLUMNS", "FAR_COLUMNS", "NESTED_RADIAL", "NESTED_RADIAL_GPUS", "R_FAR", "TOLERANCE", "basis", "install",
           "nested_default", "rotation"]
