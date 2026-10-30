# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: ASE 3.27's Langevin step (ase/md/langevin.py) as Triton kernels, via fastmace/integrators/langevin.py;
# LGPL-2.1-or-later; see NOTICE.md.
"""The integrator stage of a fixed-cell Langevin step in a few kernels (opt-3 OPT3-T03; exact).

``GraphEngine``'s AC stage (and the pipelined ACP) is about 45 small PyTorch kernels on NaCl: the first half
of ``Langevin`` (noise terms, centre-of-mass correction, half kick, drift), the neighbour check (twice in
ACP), the compaction of the candidate pairs (gather, distance test, two cumsums, three scatters) and the
flags. ``FusedIntegrator.stage`` runs the same arithmetic in four Triton kernels around the three PyTorch
reductions whose order matters (the two centre-of-mass sums and the cumsum), so the results are bitwise
those of the PyTorch stage:

- every elementwise operation is the PyTorch expression's, in its order, in float64, with the
  coefficients loaded from float64 tensors (a Python float argument would be float32) and FMA
  contraction off (``enable_fp_fusion=False``); a division of a tensor by a Python number is a
  multiplication by its reciprocal, as PyTorch's CUDA kernel does it;
- a sum over the three components is ``(x + z) + y``, the order of PyTorch's ``sum(dim=1)`` over a
  width-3 float64 row on CUDA (``tools/opt3_t03_order_probe.py``); ``x ** 2`` is ``x * x``;
- the largest squared displacement is an exact maximum (atomic max of the bit patterns of
  non-negative doubles); the destinations of the compaction come from one cumsum, since
  ``cumsum(1 - inside) == i + 1 - cumsum(inside)`` in integers.

The random numbers are those ``Langevin.prepare_step`` copied into ``xi`` and ``eta`` (the host-drawn
batch, ASE's order), so the stream is unchanged. Fixed-cell ``Langevin`` on CUDA only; anything else keeps
the PyTorch stage.

``FixAtoms`` (plan task 4-T04): with a free mask (``state.free_mask``), the drift kernel's ``FIXED`` variant
puts the fixed atoms back at their old positions after the drift, as ``Langevin.first_half``'s ``where``,
before the velocity line; their squared displacement from the reference positions is then that of the old
positions. Without a mask the kernel is the one before 4-T04.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK = 256  # elements per program (a fixed size for every crystal and GPU)


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def noise_kernel(xi_ptr, eta_ptr, c3_ptr, c4_ptr, c5_ptr, kt_ptr, m_ptr, rp_ptr, rv_ptr, q_ptr, qp_ptr, max_ptr,
                     n, MASS_WEIGHTED: tl.constexpr, BLOCK: tl.constexpr):
        """rnd_pos = (c5 sqrt_kT) eta, rnd_vel = (c3 sqrt_kT) xi - (c4 sqrt_kT) eta and q = rnd_vel m (fix_com),
        with MASS_WEIGHTED also qp = rnd_pos m; program 0 also clears the displacement maximum."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = i < n
        atom = i // 3
        kt = tl.load(kt_ptr)
        xi = tl.load(xi_ptr + i, mask=live)
        eta = tl.load(eta_ptr + i, mask=live)
        s5 = tl.load(c5_ptr + atom, mask=live) * kt
        s3 = tl.load(c3_ptr + atom, mask=live) * kt
        s4 = tl.load(c4_ptr + atom, mask=live) * kt
        rp = s5 * eta
        rv = s3 * xi - s4 * eta
        tl.store(rp_ptr + i, rp, mask=live)
        tl.store(rv_ptr + i, rv, mask=live)
        m = tl.load(m_ptr + atom, mask=live)
        tl.store(q_ptr + i, rv * m, mask=live)
        if MASS_WEIGHTED:
            tl.store(qp_ptr + i, rp * m, mask=live)
        if tl.program_id(0) == 0:
            tl.store(max_ptr, 0)

    @triton.jit
    def drift_kernel(x_ptr, v_ptr, f_ptr, m_ptr, rp_raw_ptr, rv_raw_ptr, sp_ptr, sq_ptr, total_ptr, coef_ptr, rp_ptr,
                     rv_ptr, xold_ptr, mid_ptr, ref_ptr, max_ptr, free_ptr, natoms, FIX_COM: tl.constexpr,
                     MASS_WEIGHTED: tl.constexpr, SAVE_MID: tl.constexpr, FIXED: tl.constexpr, BLOCK: tl.constexpr):
        """Per atom: the centre-of-mass correction (ASE's, or with MASS_WEIGHTED the mass-weighted means
        sp / M and sq / M, a float64 division as PyTorch's tensor / tensor, no factor; 3-T22b),
        v += c1 f/m - c2 v + rnd_vel, x_old = x,
        x += dt v + rnd_pos (with FIXED, x of a fixed atom, free mask 0, stays x_old; 4-T04),
        v = (x - x_old - rnd_pos) / dt, the saved velocity, and the squared
        displacement from the reference positions into the maximum."""
        a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = a < natoms
        c1 = tl.load(coef_ptr)
        c2 = tl.load(coef_ptr + 1)
        dt = tl.load(coef_ptr + 2)
        nf = tl.load(coef_ptr + 3)  # float(n)
        factor = tl.load(coef_ptr + 4)
        inv_dt = tl.load(coef_ptr + 5)  # PyTorch's CUDA tensor / Python scalar is tensor * (1 / scalar)
        inv_n = tl.load(coef_ptr + 6)
        m = tl.load(m_ptr + a, mask=live, other=1.0)
        if FIXED:
            free = tl.load(free_ptr + a, mask=live, other=1.0) != 0.0
        den = m * nf
        s0 = tl.zeros([BLOCK], dtype=tl.float64)  # squared displacement components (static branches below)
        s1 = tl.zeros([BLOCK], dtype=tl.float64)
        s2 = tl.zeros([BLOCK], dtype=tl.float64)
        for c in tl.static_range(3):
            k = a * 3 + c
            rp = tl.load(rp_raw_ptr + k, mask=live)
            rv = tl.load(rv_raw_ptr + k, mask=live)
            if FIX_COM and MASS_WEIGHTED:
                total = tl.load(total_ptr)
                rp = rp - tl.load(sp_ptr + c) / total
                rv = rv - tl.load(sq_ptr + c) / total
            elif FIX_COM:
                rp = (rp - tl.load(sp_ptr + c) * inv_n) * factor
                rv = (rv - tl.load(sq_ptr + c) / den) * factor
            tl.store(rp_ptr + k, rp, mask=live)
            tl.store(rv_ptr + k, rv, mask=live)
            x = tl.load(x_ptr + k, mask=live)
            v = tl.load(v_ptr + k, mask=live)
            f = tl.load(f_ptr + k, mask=live)
            v = v + (((c1 * f) / m - c2 * v) + rv)
            xn = x + (dt * v + rp)
            if FIXED:
                xn = tl.where(free, xn, x)
            v = ((xn - x) - rp) * inv_dt
            tl.store(xold_ptr + k, x, mask=live)
            tl.store(x_ptr + k, xn, mask=live)
            tl.store(v_ptr + k, v, mask=live)
            if SAVE_MID:
                tl.store(mid_ptr + k, v, mask=live)
            d = xn - tl.load(ref_ptr + k, mask=live)
            if c == 0:
                s0 = d * d
            elif c == 1:
                s1 = d * d
            else:
                s2 = d * d
        s = (s0 + s2) + s1  # PyTorch's order for sum(dim=1) of width 3
        s = tl.where(live, s, 0.0)
        tl.atomic_max(max_ptr, tl.max(s, axis=0).to(tl.int64, bitcast=True))

    @triton.jit
    def inside_kernel(x_ptr, index_ptr, shifts64_ptr, unit_ptr, cell_ptr, inside_ptr, cut_ptr, capacity,
                      LEAN: tl.constexpr, BLOCK: tl.constexpr):
        """inside[e] = |x[receiver] - x[sender] + shift|^2 < cutoff^2 for every candidate pair.

        LEAN (``SkinNeighbourList.lean``): the shift is times_3x3(unit shift, build cell), in its order."""
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = e < capacity
        sender = tl.load(index_ptr + e, mask=live, other=0)
        receiver = tl.load(index_ptr + capacity + e, mask=live, other=0)
        s0 = tl.zeros([BLOCK], dtype=tl.float64)
        s1 = tl.zeros([BLOCK], dtype=tl.float64)
        s2 = tl.zeros([BLOCK], dtype=tl.float64)
        for c in tl.static_range(3):
            if LEAN:
                shift = (tl.load(unit_ptr + e * 3, mask=live).to(tl.float64) * tl.load(cell_ptr + c)
                         + tl.load(unit_ptr + e * 3 + 1, mask=live).to(tl.float64) * tl.load(cell_ptr + 3 + c)) \
                    + tl.load(unit_ptr + e * 3 + 2, mask=live).to(tl.float64) * tl.load(cell_ptr + 6 + c)
            else:
                shift = tl.load(shifts64_ptr + e * 3 + c, mask=live)
            d = (tl.load(x_ptr + receiver * 3 + c, mask=live) - tl.load(x_ptr + sender * 3 + c, mask=live)) \
                + shift
            if c == 0:
                s0 = d * d
            elif c == 1:
                s1 = d * d
            else:
                s2 = d * d
        s = (s0 + s2) + s1
        tl.store(inside_ptr + e, (s < tl.load(cut_ptr)).to(tl.int64), mask=live)

    @triton.jit
    def scatter_kernel(inside_ptr, cum_ptr, index_ptr, unit_ptr, shifts_ptr, edge_index_ptr, edge_unit_ptr,
                       edge_shifts_ptr, cell_ptr, count_ptr, max_ptr, thresholds_ptr, rebuild_ptr, flags_ptr, pipe_ptr,
                       capacity, LEAN: tl.constexpr, PIPELINED: tl.constexpr, BLOCK: tl.constexpr):
        """Move each candidate to its destination (inside pairs first, stable), set the count and the flags.

        LEAN: int32 candidate indices, and the shift times_3x3(unit shift, build cell) cast to the edge precision,
        as ``SkinNeighbourList.compact``."""
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = e < capacity
        count = tl.load(cum_ptr + capacity - 1)
        inside = tl.load(inside_ptr + e, mask=live, other=0)
        cum = tl.load(cum_ptr + e, mask=live, other=0)
        destination = tl.where(inside != 0, cum - 1, count + (e + 1 - cum) - 1)
        for row in tl.static_range(2):
            tl.store(edge_index_ptr + row * capacity + destination,
                     tl.load(index_ptr + row * capacity + e, mask=live).to(tl.int64), mask=live)
        if LEAN:
            u0 = tl.load(unit_ptr + e * 3, mask=live)
            u1 = tl.load(unit_ptr + e * 3 + 1, mask=live)
            u2 = tl.load(unit_ptr + e * 3 + 2, mask=live)
        for c in tl.static_range(3):
            tl.store(edge_unit_ptr + destination * 3 + c, tl.load(unit_ptr + e * 3 + c, mask=live), mask=live)
            if LEAN:
                shift = (u0.to(tl.float64) * tl.load(cell_ptr + c) + u1.to(tl.float64) * tl.load(cell_ptr + 3 + c)) \
                    + u2.to(tl.float64) * tl.load(cell_ptr + 6 + c)
                tl.store(edge_shifts_ptr + destination * 3 + c,
                         shift.to(edge_shifts_ptr.dtype.element_ty), mask=live)
            else:
                tl.store(edge_shifts_ptr + destination * 3 + c, tl.load(shifts_ptr + e * 3 + c, mask=live),
                         mask=live)
        if tl.program_id(0) == 0:
            largest = tl.load(max_ptr).to(tl.float64, bitcast=True)
            hard = largest > tl.load(thresholds_ptr)
            tl.store(count_ptr, count)
            tl.store(rebuild_ptr, hard)
            tl.store(flags_ptr, hard.to(tl.int64))
            tl.store(flags_ptr + 1, count)
            if PIPELINED:
                tl.store(pipe_ptr, hard.to(tl.int64))
                tl.store(pipe_ptr + 1, count)
                tl.store(pipe_ptr + 2, (largest > tl.load(thresholds_ptr + 1)).to(tl.int64))

    return triton, noise_kernel, drift_kernel, inside_kernel, scatter_kernel


def applicable(engine) -> str | None:
    """Why ``engine`` cannot take the fused stage, or None."""
    from fastmace.integrators.langevin import Langevin
    from fastmace.kernels import triton_available
    from fastmace.neighbors.skin import SkinNeighbourList

    if not engine.cuda:
        return "not a CUDA engine"
    if engine.variable_cell:
        return "variable cell (LangevinBAOAB keeps the PyTorch stage)"
    if type(engine.integrator) is not Langevin:
        return f"integrator {type(engine.integrator).__name__}, not Langevin"
    if not isinstance(getattr(engine.force_field, "neighbours", None), SkinNeighbourList):
        return "no skin neighbour list"
    if engine.state.positions.dtype != torch.float64 or engine.state.forces.dtype != torch.float64:
        return "positions or forces are not float64"
    if len(engine._mids) != 1:
        return "the integrator saves more than the velocities"
    available = triton_available()
    if not available.available:
        return available.reason
    return None


class FusedIntegrator:
    """The fused AC/ACP stage of one engine; ``stage(pipelined)`` does what ``_stage_ac`` (and, pipelined,
    the rest of ``_stage_acp``) does."""

    def __init__(self, engine):
        self.engine = engine
        device = engine.state.positions.device
        self.coefficients = self._coefficients(device)
        neighbours = engine.force_field.neighbours
        fraction = engine.config.pipeline_rebuild_fraction
        # The neighbour checks' thresholds as SkinNeighbourList.needs_rebuild forms them.
        self.thresholds = torch.tensor([(1.0 * neighbours.skin / 2) ** 2, (fraction * neighbours.skin / 2) ** 2],
                                       dtype=torch.float64, device=device)
        self.cutoff2 = torch.tensor([neighbours.cutoff ** 2], dtype=torch.float64, device=device)
        self.largest = torch.zeros((), dtype=torch.int64, device=device)
        # The total mass of the mass-weighted projection, as Langevin.first_half forms it (masses.sum()).
        self.total_mass = engine.state.masses.sum().reshape(1)

    def _coefficients(self, device):
        integrator = self.engine.integrator
        n = integrator.natoms
        factor = (n / (n - 1.0)) ** 0.5 if integrator.fix_com else 1.0
        # c1, c2, dt, float(n), factor, 1 / dt, 1 / n: float64 on the device (a Python float kernel argument
        # is float32). The reciprocals are the host's, as PyTorch forms them for a division by a scalar.
        return torch.tensor([integrator.c1, integrator.c2, integrator.dt, float(n), factor, 1.0 / integrator.dt,
                             1.0 / float(n)], dtype=torch.float64, device=device)

    def register_derived(self, registry) -> None:
        """``coefficients`` follow the friction (``c1``, ``c2``). Its refresh copies host floats to the device:
        a friction event is a host event that drops the graphs anyway, never replayed inside one (SPEC §8.5)."""
        registry.register("fused.coefficients", self.coefficients,
                          lambda: self.coefficients.copy_(self._coefficients(self.coefficients.device)),
                          depends_on=("friction",))
        masses = self.engine.state.masses
        registry.register("fused.total_mass", self.total_mass, lambda: self.total_mass.copy_(masses.sum().reshape(1)),
                          depends_on=("masses",))

    def stage(self, pipelined: bool) -> None:
        triton, noise_kernel, drift_kernel, inside_kernel, scatter_kernel = _kernels()
        engine = self.engine
        integrator, state = engine.integrator, engine.state
        neighbours = engine.force_field.neighbours
        natoms = integrator.natoms
        n3 = natoms * 3
        rp_raw = torch.empty_like(state.positions)
        rv_raw = torch.empty_like(state.positions)
        q = torch.empty_like(state.positions)
        weighted = bool(integrator.fix_com and integrator.mass_weighted_com)
        qp = torch.empty_like(state.positions) if weighted else q
        noise_kernel[(triton.cdiv(n3, BLOCK),)](
            integrator.xi, integrator.eta, integrator.c3_unit, integrator.c4_unit, integrator.c5_unit,
            integrator.sqrt_kT, state.masses, rp_raw, rv_raw, q, qp, self.largest, n3, MASS_WEIGHTED=weighted,
            BLOCK=BLOCK, enable_fp_fusion=False)
        if weighted:  # PyTorch's own reductions, as Langevin.first_half: (rnd_pos * masses).sum(dim=0)
            sp, sq = qp.sum(dim=0), q.sum(dim=0)
        elif integrator.fix_com:  # PyTorch's own reductions: their order is the baseline's
            sp, sq = rp_raw.sum(dim=0), q.sum(dim=0)
        else:
            sp = sq = rp_raw
        free = state.free_mask
        drift_kernel[(triton.cdiv(natoms, BLOCK),)](
            state.positions, state.velocities, state.forces, state.masses, rp_raw, rv_raw, sp, sq, self.total_mass,
            self.coefficients, integrator.rnd_pos, integrator.rnd_vel, integrator.x_old, engine._mids[0],
            neighbours.reference_positions, self.largest, state.masses if free is None else free, natoms,
            FIX_COM=bool(integrator.fix_com), MASS_WEIGHTED=weighted, SAVE_MID=pipelined, FIXED=free is not None,
            BLOCK=BLOCK, enable_fp_fusion=False)
        capacity = neighbours.capacity
        inside = torch.empty(capacity, dtype=torch.int64, device=state.positions.device)
        lean = bool(neighbours.lean)
        unit = neighbours.candidate_unit_shifts
        # Lean: the shifts come from the unit shifts and the build cell (the unused pointers get the unit shifts).
        cell = neighbours.build_cell if lean else unit
        shifts64 = unit if lean else neighbours.candidate_shifts64
        shifts = unit if lean else neighbours.candidate_shifts
        inside_kernel[(triton.cdiv(capacity, BLOCK),)](
            state.positions, neighbours.candidate_index, shifts64, unit, cell, inside, self.cutoff2,
            capacity, LEAN=lean, BLOCK=BLOCK, enable_fp_fusion=False)
        cum = torch.cumsum(inside, 0)
        scatter_kernel[(triton.cdiv(capacity, BLOCK),)](
            inside, cum, neighbours.candidate_index, unit, shifts,
            neighbours.edge_index, neighbours.unit_shifts, neighbours.shifts, cell, neighbours.count, self.largest,
            self.thresholds, engine.rebuild_needed, engine._flags, engine._pipe_flags, capacity,
            LEAN=lean, PIPELINED=pipelined, BLOCK=BLOCK, enable_fp_fusion=False)
        d3_needs_rebuild = getattr(engine.force_field, "d3_needs_rebuild", None)
        if d3_needs_rebuild is not None:  # the D3 list's own test, OR-ed into the flags the kernels wrote (3b-T04)
            flag = d3_needs_rebuild(state.positions)
            engine.rebuild_needed.logical_or_(flag)
            engine._flags[0].copy_(engine.rebuild_needed)
            if pipelined:
                engine._pipe_flags[0].copy_(engine.rebuild_needed)
                early = d3_needs_rebuild(state.positions, engine.config.pipeline_rebuild_fraction)
                engine._pipe_flags[2].copy_(engine._pipe_flags[2].bool() | early)


__all__ = ["BLOCK", "FusedIntegrator", "applicable"]
