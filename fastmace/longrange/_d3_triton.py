"""The Triton kernels of ``fastmace.longrange.d3_kernel`` (SPEC §11.2), imported on first GPU use.

One program per atom walks the atom's i segment and then its j segment of the sorted half list,
in blocks of ``BLOCK`` pairs, and sums in float64 registers. No atomics. ``F64`` selects the pair
precision. The helpers are module-level ``triton.jit`` functions because Triton resolves the
callees of a kernel through its module's globals.
"""

import triton
import triton.language as tl


@triton.jit
def pair_vector(pos_ptr, cell_ptr, a, b, n0, n1, n2, autoang, F64: tl.constexpr):
    """x_b + (n0, n1, n2) @ cell - x_a in bohr, in the pair precision."""
    dt = tl.float64 if F64 else tl.float32
    c00 = tl.load(cell_ptr + 0)
    c01 = tl.load(cell_ptr + 1)
    c02 = tl.load(cell_ptr + 2)
    c10 = tl.load(cell_ptr + 3)
    c11 = tl.load(cell_ptr + 4)
    c12 = tl.load(cell_ptr + 5)
    c20 = tl.load(cell_ptr + 6)
    c21 = tl.load(cell_ptr + 7)
    c22 = tl.load(cell_ptr + 8)
    n0 = n0.to(c00.dtype)
    n1 = n1.to(c00.dtype)
    n2 = n2.to(c00.dtype)
    a3 = a.to(tl.int64) * 3
    b3 = b.to(tl.int64) * 3
    x = (tl.load(pos_ptr + b3) + (n0 * c00 + n1 * c10 + n2 * c20) - tl.load(pos_ptr + a3)) / autoang
    y = (tl.load(pos_ptr + b3 + 1) + (n0 * c01 + n1 * c11 + n2 * c21) - tl.load(pos_ptr + a3 + 1)) / autoang
    z = (tl.load(pos_ptr + b3 + 2) + (n0 * c02 + n1 * c12 + n2 * c22) - tl.load(pos_ptr + a3 + 2)) / autoang
    return x.to(dt), y.to(dt), z.to(dt)

@triton.jit
def segment_pair(atom, k, row_ptr, col_ptr, by_j_ptr, pair_i_ptr, pair_j_ptr, images_ptr,
                 J_SIDE: tl.constexpr, BLOCK: tl.constexpr):
    """Pairs k..k+BLOCK of the atom's i (J_SIDE False) or j segment: (i, j, n0, n1, n2, mask)."""
    if J_SIDE:
        end = tl.load(col_ptr + atom + 1)
    else:
        end = tl.load(row_ptr + atom + 1)
    idx = k + tl.arange(0, BLOCK)
    mask = idx < end
    if J_SIDE:
        p = tl.load(by_j_ptr + idx, mask=mask, other=0)
    else:
        p = idx
    i = tl.load(pair_i_ptr + p, mask=mask, other=0)
    j = tl.load(pair_j_ptr + p, mask=mask, other=0)
    p3 = p.to(tl.int64) * 3
    n0 = tl.load(images_ptr + p3, mask=mask, other=0)
    n1 = tl.load(images_ptr + p3 + 1, mask=mask, other=0)
    n2 = tl.load(images_ptr + p3 + 2, mask=mask, other=0)
    return i, j, n0, n1, n2, mask

@triton.jit
def damping(rcov_ptr, species_ptr, i, j, r):
    rco = tl.load(rcov_ptr + tl.load(species_ptr + i)) + tl.load(rcov_ptr + tl.load(species_ptr + j))
    damp = 1.0 / (1.0 + tl.exp(-16.0 * (rco / r - 1.0)))
    return damp, rco

@triton.jit
def cn_kernel(pos_ptr, cell_ptr, scalars_ptr, species_ptr, pair_i_ptr, pair_j_ptr, images_ptr, row_ptr,
              col_ptr, by_j_ptr, rcov_ptr, cn_ptr, F64: tl.constexpr, BLOCK: tl.constexpr):
    dt = tl.float64 if F64 else tl.float32
    atom = tl.program_id(0)
    autoang = tl.load(scalars_ptr).to(tl.load(pos_ptr).dtype)
    cutoff2 = tl.load(scalars_ptr + 2).to(dt)
    acc = tl.zeros((BLOCK,), dtype=tl.float64)
    for side in tl.static_range(2):
        if side == 0:
            start = tl.load(row_ptr + atom)
            end = tl.load(row_ptr + atom + 1)
        else:
            start = tl.load(col_ptr + atom)
            end = tl.load(col_ptr + atom + 1)
        for k in range(start, end, BLOCK):
            i, j, n0, n1, n2, mask = segment_pair(atom, k, row_ptr, col_ptr, by_j_ptr, pair_i_ptr,
                                                  pair_j_ptr, images_ptr, side == 1, BLOCK)
            x, y, z = pair_vector(pos_ptr, cell_ptr, i, j, n0, n1, n2, autoang, F64)
            r2 = x * x + y * y + z * z
            keep = mask & (r2 <= cutoff2)
            r = tl.sqrt(tl.where(keep, r2, 1.0))
            damp, _ = damping(rcov_ptr, species_ptr, i, j, r)
            acc += tl.where(keep, damp, 0.0).to(tl.float64)
    tl.store(cn_ptr + atom, tl.sum(acc, axis=0))

@triton.jit
def energy_kernel(pos_ptr, cell_ptr, scalars_ptr, species_ptr, pair_i_ptr, pair_j_ptr, images_ptr, row_ptr,
                  col_ptr, by_j_ptr, cn_ptr, references_ptr, c8_ptr, r06_ptr, r08_ptr, energy_ptr, de_dcn_ptr,
                  forces_ptr, virial_ptr, count, F64: tl.constexpr, BLOCK: tl.constexpr):
    dt = tl.float64 if F64 else tl.float32
    atom = tl.program_id(0)
    autoang = tl.load(scalars_ptr).to(tl.load(pos_ptr).dtype)
    cutoff2 = tl.load(scalars_ptr + 1).to(dt)
    s6 = tl.load(scalars_ptr + 3).to(dt)
    s8 = tl.load(scalars_ptr + 4).to(dt)
    e_acc = tl.zeros((BLOCK,), dtype=tl.float64)
    dcn_acc = tl.zeros((BLOCK,), dtype=tl.float64)
    fx = tl.zeros((BLOCK,), dtype=tl.float64)
    fy = tl.zeros((BLOCK,), dtype=tl.float64)
    fz = tl.zeros((BLOCK,), dtype=tl.float64)
    vxx = tl.zeros((BLOCK,), dtype=tl.float64)
    vxy = tl.zeros((BLOCK,), dtype=tl.float64)
    vxz = tl.zeros((BLOCK,), dtype=tl.float64)
    vyy = tl.zeros((BLOCK,), dtype=tl.float64)
    vyz = tl.zeros((BLOCK,), dtype=tl.float64)
    vzz = tl.zeros((BLOCK,), dtype=tl.float64)
    ref = tl.arange(0, 32)
    for side in tl.static_range(2):
        if side == 0:
            start = tl.load(row_ptr + atom)
            end = tl.load(row_ptr + atom + 1)
        else:
            start = tl.load(col_ptr + atom)
            end = tl.load(col_ptr + atom + 1)
        for k in range(start, end, BLOCK):
            i, j, n0, n1, n2, mask = segment_pair(atom, k, row_ptr, col_ptr, by_j_ptr, pair_i_ptr,
                                                  pair_j_ptr, images_ptr, side == 1, BLOCK)
            x, y, z = pair_vector(pos_ptr, cell_ptr, i, j, n0, n1, n2, autoang, F64)
            r2 = x * x + y * y + z * z
            keep = mask & (r2 <= cutoff2)
            cn_i = tl.load(cn_ptr + i).to(dt)
            cn_j = tl.load(cn_ptr + j).to(dt)
            pair = tl.load(species_ptr + i) * count + tl.load(species_ptr + j)
            # C6(CN_i, CN_j): torch-dftd's softmax over the reference pairs with c6 > 0
            base = (pair[:, None].to(tl.int64) * 32 + ref[None, :]) * 3
            c6_ref = tl.load(references_ptr + base, mask=keep[:, None], other=0.0)
            da = cn_i[:, None] - tl.load(references_ptr + base + 1, mask=keep[:, None], other=0.0)
            db = cn_j[:, None] - tl.load(references_ptr + base + 2, mask=keep[:, None], other=0.0)
            log_weight = tl.where(c6_ref > 0.0, -4.0 * (da * da + db * db), -1.0e20)
            weight = tl.exp(log_weight - tl.max(log_weight, axis=1)[:, None])
            total = tl.sum(weight, axis=1)
            c6 = tl.sum(weight * c6_ref, axis=1) / total
            excess = weight * (c6_ref - c6[:, None])
            r6 = r2 * r2 * r2
            d6 = 1.0 / (r6 + tl.load(r06_ptr + pair))
            d8 = 1.0 / (r6 * r2 + tl.load(r08_ptr + pair))
            c8_factor = tl.load(c8_ptr + pair)
            de_dc6 = -(s6 * d6 + s8 * c8_factor * d8)
            g = tl.where(keep, c6 * (6.0 * s6 * r2 * r2 * d6 * d6 + 8.0 * s8 * c8_factor * r6 * d8 * d8), 0.0)
            fx_pair = (g * x).to(tl.float64)
            fy_pair = (g * y).to(tl.float64)
            fz_pair = (g * z).to(tl.float64)
            if side == 0:  # the atom is i: energy, virial, dE/dCN_i, +pair force
                dc6 = tl.sum(excess * (-8.0 * da), axis=1) / total
                e_acc += tl.where(keep, c6 * de_dc6, 0.0).to(tl.float64)
                dcn_acc += tl.where(keep, de_dc6 * dc6, 0.0).to(tl.float64)
                fx += fx_pair
                fy += fy_pair
                fz += fz_pair
                xd = x.to(tl.float64)
                yd = y.to(tl.float64)
                zd = z.to(tl.float64)
                vxx -= xd * fx_pair
                vxy -= xd * fy_pair
                vxz -= xd * fz_pair
                vyy -= yd * fy_pair
                vyz -= yd * fz_pair
                vzz -= zd * fz_pair
            else:  # the atom is j: dE/dCN_j, -pair force
                dc6 = tl.sum(excess * (-8.0 * db), axis=1) / total
                dcn_acc += tl.where(keep, de_dc6 * dc6, 0.0).to(tl.float64)
                fx -= fx_pair
                fy -= fy_pair
                fz -= fz_pair
    tl.store(energy_ptr + atom, tl.sum(e_acc, axis=0))
    tl.store(de_dcn_ptr + atom, tl.sum(dcn_acc, axis=0))
    a3 = atom.to(tl.int64) * 3
    tl.store(forces_ptr + a3, tl.sum(fx, axis=0))
    tl.store(forces_ptr + a3 + 1, tl.sum(fy, axis=0))
    tl.store(forces_ptr + a3 + 2, tl.sum(fz, axis=0))
    a9 = atom.to(tl.int64) * 9
    sxy = tl.sum(vxy, axis=0)
    sxz = tl.sum(vxz, axis=0)
    syz = tl.sum(vyz, axis=0)
    tl.store(virial_ptr + a9 + 0, tl.sum(vxx, axis=0))
    tl.store(virial_ptr + a9 + 1, sxy)
    tl.store(virial_ptr + a9 + 2, sxz)
    tl.store(virial_ptr + a9 + 3, sxy)
    tl.store(virial_ptr + a9 + 4, tl.sum(vyy, axis=0))
    tl.store(virial_ptr + a9 + 5, syz)
    tl.store(virial_ptr + a9 + 6, sxz)
    tl.store(virial_ptr + a9 + 7, syz)
    tl.store(virial_ptr + a9 + 8, tl.sum(vzz, axis=0))

@triton.jit
def chain_kernel(pos_ptr, cell_ptr, scalars_ptr, species_ptr, pair_i_ptr, pair_j_ptr, images_ptr, row_ptr,
                 col_ptr, by_j_ptr, rcov_ptr, de_dcn_ptr, forces_ptr, virial_ptr, F64: tl.constexpr,
                 BLOCK: tl.constexpr):
    dt = tl.float64 if F64 else tl.float32
    atom = tl.program_id(0)
    autoang = tl.load(scalars_ptr).to(tl.load(pos_ptr).dtype)
    cutoff2 = tl.load(scalars_ptr + 2).to(dt)
    fx = tl.zeros((BLOCK,), dtype=tl.float64)
    fy = tl.zeros((BLOCK,), dtype=tl.float64)
    fz = tl.zeros((BLOCK,), dtype=tl.float64)
    vxx = tl.zeros((BLOCK,), dtype=tl.float64)
    vxy = tl.zeros((BLOCK,), dtype=tl.float64)
    vxz = tl.zeros((BLOCK,), dtype=tl.float64)
    vyy = tl.zeros((BLOCK,), dtype=tl.float64)
    vyz = tl.zeros((BLOCK,), dtype=tl.float64)
    vzz = tl.zeros((BLOCK,), dtype=tl.float64)
    for side in tl.static_range(2):
        if side == 0:
            start = tl.load(row_ptr + atom)
            end = tl.load(row_ptr + atom + 1)
        else:
            start = tl.load(col_ptr + atom)
            end = tl.load(col_ptr + atom + 1)
        for k in range(start, end, BLOCK):
            i, j, n0, n1, n2, mask = segment_pair(atom, k, row_ptr, col_ptr, by_j_ptr, pair_i_ptr,
                                                  pair_j_ptr, images_ptr, side == 1, BLOCK)
            x, y, z = pair_vector(pos_ptr, cell_ptr, i, j, n0, n1, n2, autoang, F64)
            r2 = x * x + y * y + z * z
            keep = mask & (r2 <= cutoff2)
            r = tl.sqrt(tl.where(keep, r2, 1.0))
            damp, rco = damping(rcov_ptr, species_ptr, i, j, r)
            ddamp = -damp * (1.0 - damp) * 16.0 * rco / (r * r)
            de_dcn = (tl.load(de_dcn_ptr + i) + tl.load(de_dcn_ptr + j)).to(dt)
            g = tl.where(keep, de_dcn * ddamp / r, 0.0)
            fx_pair = (g * x).to(tl.float64)
            fy_pair = (g * y).to(tl.float64)
            fz_pair = (g * z).to(tl.float64)
            if side == 0:
                fx += fx_pair
                fy += fy_pair
                fz += fz_pair
                xd = x.to(tl.float64)
                yd = y.to(tl.float64)
                zd = z.to(tl.float64)
                vxx -= xd * fx_pair
                vxy -= xd * fy_pair
                vxz -= xd * fz_pair
                vyy -= yd * fy_pair
                vyz -= yd * fz_pair
                vzz -= zd * fz_pair
            else:
                fx -= fx_pair
                fy -= fy_pair
                fz -= fz_pair
    a3 = atom.to(tl.int64) * 3
    tl.store(forces_ptr + a3, tl.load(forces_ptr + a3) + tl.sum(fx, axis=0))
    tl.store(forces_ptr + a3 + 1, tl.load(forces_ptr + a3 + 1) + tl.sum(fy, axis=0))
    tl.store(forces_ptr + a3 + 2, tl.load(forces_ptr + a3 + 2) + tl.sum(fz, axis=0))
    a9 = atom.to(tl.int64) * 9
    sxy = tl.sum(vxy, axis=0)
    sxz = tl.sum(vxz, axis=0)
    syz = tl.sum(vyz, axis=0)
    tl.store(virial_ptr + a9 + 0, tl.load(virial_ptr + a9 + 0) + tl.sum(vxx, axis=0))
    tl.store(virial_ptr + a9 + 1, tl.load(virial_ptr + a9 + 1) + sxy)
    tl.store(virial_ptr + a9 + 2, tl.load(virial_ptr + a9 + 2) + sxz)
    tl.store(virial_ptr + a9 + 3, tl.load(virial_ptr + a9 + 3) + sxy)
    tl.store(virial_ptr + a9 + 4, tl.load(virial_ptr + a9 + 4) + tl.sum(vyy, axis=0))
    tl.store(virial_ptr + a9 + 5, tl.load(virial_ptr + a9 + 5) + syz)
    tl.store(virial_ptr + a9 + 6, tl.load(virial_ptr + a9 + 6) + sxz)
    tl.store(virial_ptr + a9 + 7, tl.load(virial_ptr + a9 + 7) + syz)
    tl.store(virial_ptr + a9 + 8, tl.load(virial_ptr + a9 + 8) + tl.sum(vzz, axis=0))
