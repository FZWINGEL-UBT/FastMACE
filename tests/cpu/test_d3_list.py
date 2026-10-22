"""The D3 half list (SPEC §11.2; plan task 3b-T02): pairs against a brute-force enumeration, the rebuild test,
the layout the kernel reads, counters and the canonical rebuild; rule 11's D3 skin and rule 12's estimate."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase.data import atomic_masses

from _support.structures import STRUCTURES, nbcl5
from fastmace.integrators.baoab import barostat_mass
from fastmace.neighbors import adaptive, d3_list
from fastmace.neighbors.d3_list import D3PairList

D3_CUTOFF = 40 * 0.52917726  # 40 bohr in torch-dftd's constant, 21.167 Å
SKIN = 1.0
SHEAR = np.array([[1.0, 0.07, -0.03], [0.05, 0.93, 0.04], [-0.02, 0.06, 1.08]])
CL = atomic_masses[17]


def brute_force(positions, cell, radius):
    """{(i, j, n1, n2, n3)}: i < j with every image, and i == j with n lexicographically positive, |r| < radius.

    Independent of the list: unwrapped positions, a generous image box from the cell heights.
    """
    positions, cell = np.asarray(positions, float), np.asarray(cell, float)
    heights = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    span = positions @ np.linalg.inv(cell)
    extent = np.ceil(radius / heights + (span.max(axis=0) - span.min(axis=0)) + 1).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(-e, e + 1) for e in extent], indexing="ij"), -1).reshape(-1, 3)
    translations = grid @ cell
    natoms = len(positions)
    found = set()
    for i in range(natoms):
        d = positions[i:, None, :] - positions[i] + translations[None]  # (j - i, images, 3)
        close = np.argwhere((d * d).sum(-1) < radius * radius)
        for offset, k in close:
            j, n = i + offset, grid[k]
            if j == i and tuple(n) <= (0, 0, 0):
                continue
            found.add((i, int(j), *map(int, n)))
    return found


def listed(nl):
    i, j, shifts = nl.pairs()
    return set(zip(i.tolist(), j.tolist(), *shifts.T.tolist()))


def inside(nl, positions, cell=None):
    """The pairs the kernel's mask keeps at ``positions`` (and ``cell``): {(i, j, n)}."""
    if cell is not None:
        nl.set_cell(torch.as_tensor(cell, dtype=torch.float64))
    nl.prepare(torch.as_tensor(positions, dtype=torch.float64))
    mask = nl.inside()
    i, j, shifts = nl.pairs()
    keep = mask[:nl.num_pairs].numpy()
    assert not mask[nl.num_pairs:].any(), "a padding pair is within the cutoff"
    return set(zip(i[keep].tolist(), j[keep].tolist(), *shifts[keep].T.tolist()))


def cases():
    out = []
    for name in ("NaCl", "NaNbO3", "NbCl5"):
        build = STRUCTURES[name]
        out.append((f"{name}-rattled", build(repeat=1, rattle=0.05, seed=3)))
        strained = build(repeat=1, rattle=0.05, seed=4)
        strained.set_cell(strained.cell.array @ SHEAR, scale_atoms=True)
        out.append((f"{name}-strained", strained))
    unwrapped = nbcl5(rattle=0.05, seed=5)  # atoms outside the cell, several cells away
    rng = np.random.default_rng(1)
    unwrapped.positions += rng.integers(-3, 4, size=(len(unwrapped), 3)) @ unwrapped.cell.array
    out.append(("NbCl5-unwrapped", unwrapped))
    return out


CASES = cases()
IDS = [name for name, _ in CASES]


@pytest.mark.parametrize("atoms", [a for _, a in CASES], ids=IDS)
def test_build_equals_brute_force(atoms):
    """Every pair within cutoff + skin once, every image, no minimum image (cells far narrower than 2 r_c)."""
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    nl.build(atoms.positions)
    expected = brute_force(atoms.positions, atoms.cell.array, D3_CUTOFF + SKIN)
    got = listed(nl)
    assert nl.num_pairs == len(got)  # no duplicates
    assert got == expected
    assert all(i <= j for i, j, *_ in got)
    assert inside(nl, atoms.positions) == brute_force(atoms.positions, atoms.cell.array, D3_CUTOFF)


def test_pair_vectors_from_the_layout():
    """shifted_positions[j] - shifted_positions[i] + image_vectors[image] = x_j - x_i + n h, padding far away."""
    atoms = CASES[-1][1]
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    nl.build(atoms.positions)
    x = torch.tensor(atoms.positions)
    nl.prepare(x)
    i, j = nl.index[0].long(), nl.index[1].long()
    vectors = nl.shifted_positions[j] - nl.shifted_positions[i] + nl.image_vectors[nl.image.long()]
    ii, jj, shifts = nl.pairs()
    expected = x[jj] - x[ii] + torch.tensor(shifts, dtype=torch.float64) @ torch.tensor(atoms.cell.array)
    n = nl.num_pairs
    torch.testing.assert_close(vectors[:n], expected, rtol=0, atol=1e-10)
    assert nl.capacity > n and (vectors[n:].norm(dim=1) > 1000).all()
    assert nl.index.dtype == nl.image.dtype == torch.int32
    assert int(nl.count) == n


def test_self_image_pairs_once():
    """A one-atom cell: only self-image pairs, n and -n stored once."""
    from ase import Atoms

    atoms = Atoms("Ar", positions=[[0.3, 0.2, 0.1]], cell=np.diag([3.0, 3.5, 4.0]), pbc=True)
    nl = D3PairList(atoms.cell.array, 7.0, 0.5, "cpu")
    nl.build(atoms.positions)
    got = listed(nl)
    assert got == brute_force(atoms.positions, atoms.cell.array, 7.5)
    shifts = {(a, b, c) for _, _, a, b, c in got}
    assert len(shifts) == len(got) > 0
    assert not any((-a, -b, -c) in shifts for a, b, c in shifts)


@pytest.mark.parametrize("variable_cell", [False, True], ids=["fixed", "variable"])
def test_masked_pairs_stay_exact_until_the_rebuild_test_fires(variable_cell):
    """While needs_rebuild is False, the stored pairs within the cutoff are exactly the brute-force pairs."""
    atoms = nbcl5(rattle=0.05, seed=7)
    cell0 = atoms.cell.array
    nl = D3PairList(cell0, D3_CUTOFF, SKIN, "cpu", variable_cell=variable_cell)
    nl.build(atoms.positions)
    rng = np.random.default_rng(11)
    checked = 0
    for trial in range(12):
        strain = np.eye(3) + (rng.normal(scale=0.004, size=(3, 3)) if variable_cell else 0)
        cell = cell0 @ strain
        x = atoms.positions @ strain + rng.uniform(-0.3, 0.3, (len(atoms), 3)) * (trial / 12)
        if variable_cell:
            nl.set_cell(torch.tensor(cell))
        if bool(nl.needs_rebuild(torch.tensor(x))):
            continue
        checked += 1
        assert inside(nl, x, cell if variable_cell else None) == brute_force(x, cell, D3_CUTOFF)
    assert checked >= 3


def test_rebuild_test_fires():
    atoms = nbcl5(rattle=0.05, seed=8)
    fixed = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    fixed.build(atoms.positions)
    x = torch.tensor(atoms.positions)
    assert not bool(fixed.needs_rebuild(x))
    moved = x.clone()
    moved[5, 0] += 0.51 * SKIN
    assert bool(fixed.needs_rebuild(moved))
    assert not bool(fixed.needs_rebuild(moved, fraction=1.1))

    variable = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu", variable_cell=True)
    variable.build(atoms.positions)
    # Uniform strain with every atom carried along: no displacement in build coordinates, only the strain
    # term cutoff f / (1 - f) against the skin, f = sqrt(3) eps.
    eps_ok = 0.9 * SKIN / (D3_CUTOFF + SKIN) / math.sqrt(3)
    eps_bad = 1.1 * SKIN / (D3_CUTOFF + SKIN) / math.sqrt(3)
    for eps, fires in ((eps_ok, False), (eps_bad, True)):
        scale = 1 + eps
        variable.set_cell(torch.tensor(atoms.cell.array * scale))
        assert bool(variable.needs_rebuild(x * scale)) is fires


def test_variable_cell_rebuild_follows_the_cell():
    atoms = STRUCTURES["NaCl"](repeat=1, rattle=0.05, seed=9)
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu", variable_cell=True)
    nl.build(atoms.positions)
    cell = atoms.cell.array @ SHEAR
    x = atoms.positions @ SHEAR
    nl.set_cell(torch.tensor(cell))
    assert bool(nl.needs_rebuild(torch.tensor(x)))
    moved = nl.build(x)
    assert moved  # a sheared cell needs more images
    assert listed(nl) == brute_force(x, cell, D3_CUTOFF + SKIN)
    torch.testing.assert_close(nl.build_cell, torch.tensor(cell))


def test_counters_capacity_and_canonical_rebuild():
    atoms = nbcl5(rattle=0.05, seed=10)
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    assert nl.build(atoms.positions)  # the first build allocates
    first = {name: tensor.clone() for name, tensor in nl.buffers().items()}
    assert not nl.build(atoms.positions)  # the same capacity, same addresses
    counters = nl.counters()
    assert counters["d3_builds"] == 2 and counters["d3_pairs"] == nl.num_pairs
    assert counters["d3_build_time_s"] >= counters["d3_last_build_time_s"] > 0
    assert counters["d3_capacity"] >= math.ceil(nl.num_pairs * d3_list.DEFAULT_HEADROOM) - 1
    # A canonical point at the same positions reproduces every buffer bit for bit.
    fresh = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    fresh.set_capacity(nl.capacity)
    fresh.canonicalise(atoms.positions)
    nl.canonicalise(atoms.positions)
    for name, tensor in nl.buffers().items():
        assert torch.equal(tensor, fresh.buffers()[name]), name
        assert torch.equal(tensor, first[name]) or name == "shifted_positions", name
    with pytest.raises(ValueError, match="below"):
        nl.set_capacity(nl.num_pairs - 1)
    assert nl.set_capacity(nl.capacity + 1024)
    assert (nl.image == nl.images.shape[0] - 1).all()  # padding until the next build


def test_set_cell_needs_variable_cell():
    atoms = STRUCTURES["NaCl"](repeat=1)
    with pytest.raises(RuntimeError, match="variable_cell"):
        D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu").set_cell(torch.eye(3, dtype=torch.float64))


def test_image_grid_is_symmetric():
    grid = d3_list.image_grid(nbcl5().cell.array @ SHEAR, 22.0)
    assert np.array_equal(grid[::-1], -grid)
    assert np.array_equal(grid[(len(grid) - 1) // 2], [0, 0, 0])


# ---------------------------------------------------------------------------------------------------- rules 11, 12


def test_barostat_strain_rate_uses_the_integrators_mass():
    from ase import units

    for p_tau, natoms, factor in ((2000.0, 1944, 1.0), (500.0, 216, 2.0)):
        assert adaptive.barostat_mass(p_tau, natoms, factor) == pytest.approx(
            barostat_mass(p_tau * units.fs, natoms, None, factor), rel=1e-12)
    assert adaptive.ASE_TIME_PER_FS == pytest.approx(units.fs, rel=1e-12)
    rate = adaptive.barostat_strain_rate(temperature_K=1000.0, natoms=1944, P_tau_fs=2000.0)
    deviation = math.sqrt(units.kB * 1000.0 / barostat_mass(2000.0 * units.fs, 1944)) * units.fs
    assert rate == pytest.approx(3.0 * math.sqrt(3.0) * deviation, rel=1e-6)  # CODATA versions of k_B
    full = adaptive.barostat_strain_rate(temperature_K=1000.0, natoms=1944, P_tau_fs=2000.0, cell_mode="full")
    assert full == pytest.approx(rate * math.sqrt(3.0), rel=1e-12)


def test_rule_11_d3_skin():
    base = dict(max_temperature_K=1000.0, lightest_mass_u=CL, timestep_fs=1.0, natoms=1944, cutoff_A=D3_CUTOFF)
    nvt = adaptive.rule_11_d3(**base)
    d = adaptive.fastest_speed_quantile(1944) * adaptive.thermal_speed(1000.0, CL)
    assert nvt.step_displacement_A == pytest.approx(d)
    assert nvt.displacement_term_A == pytest.approx(2 * adaptive.D3_TARGET_REBUILD_STEPS * d)
    assert nvt.strain_term_A == 0 and nvt.skin_A >= nvt.displacement_term_A
    rate = adaptive.barostat_strain_rate(temperature_K=1000.0, natoms=1944, P_tau_fs=2000.0)
    npt = adaptive.rule_11_d3(**base, strain_rate_per_fs=rate)
    f = rate * adaptive.D3_TARGET_REBUILD_STEPS
    assert npt.strain_term_A == pytest.approx(D3_CUTOFF * f / (1 - f))
    assert npt.skin_A > nvt.skin_A and 0.05 < npt.strain_term_A < 0.5  # NbCl5 3x3x3 at 1000 K, P_tau 2 ps
    cold = adaptive.rule_11_d3(**{**base, "max_temperature_K": 10.0})
    assert cold.skin_A == adaptive.D3_FLOOR_SKIN_A
    assert "barostat strain" in npt.text() and "barostat" not in nvt.text()
    with pytest.raises(ValueError, match="strain"):
        adaptive.rule_11_d3(**base, strain_rate_per_fs=0.02)


def test_rule_11_d3_skin_keeps_the_rebuild_interval():
    """At the planned skin, the fastest atom and the barostat strain need the target steps to fire the test."""
    atoms = nbcl5(repeat=1)
    rate = adaptive.barostat_strain_rate(temperature_K=1000.0, natoms=1944, P_tau_fs=2000.0)
    sized = adaptive.rule_11_d3(max_temperature_K=1000.0, lightest_mass_u=CL, timestep_fs=1.0, natoms=1944,
                                cutoff_A=D3_CUTOFF, strain_rate_per_fs=rate)
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, sized.skin_A, "cpu", variable_cell=True)
    nl.build(atoms.positions)
    n = sized.target_rebuild_steps
    eps = sized.strain_per_step * n / math.sqrt(3) * 0.999  # hydrostatic: |F - I|_F = sqrt(3) eps
    x = torch.tensor(atoms.positions) * (1 + eps)
    x[0, 0] += n * sized.step_displacement_A * 0.999
    nl.set_cell(torch.tensor(atoms.cell.array * (1 + eps)))
    assert not bool(nl.needs_rebuild(x))


def test_rule_12_d3_matches_the_spec_and_a_build():
    """SPEC §11.2: NbCl5 3x3x3 at 28.8 Å^3/atom: about 1.4 M stored pairs at 21.5 Å, 1.75 M at 23.2 Å."""
    volume = 1944 * 28.8
    assert adaptive.rule_12_d3(natoms=1944, volume_A3=volume, radius_A=21.5).pairs == pytest.approx(1.40e6, rel=0.02)
    big = adaptive.rule_12_d3(natoms=1944, volume_A3=volume, radius_A=D3_CUTOFF + 2.0, gpu_memory_bytes=16 * 2**30)
    assert big.pairs == pytest.approx(1.75e6, rel=0.02)
    assert big.list_bytes == pytest.approx(big.capacity * d3_list.BYTES_PER_PAIR, rel=0.01)
    assert big.capacity >= big.pairs * d3_list.DEFAULT_HEADROOM and not big.warn
    assert "M pairs" in big.text() and "GiB" in big.text()
    assert adaptive.rule_12_d3(natoms=1944, volume_A3=volume, radius_A=23.2, gpu_memory_bytes=2**28).warn
    # The estimate against a real build: NbCl5 2x2x2 (288 atoms).
    atoms = nbcl5(repeat=2, rattle=0.05, seed=12)
    nl = D3PairList(atoms.cell.array, D3_CUTOFF, SKIN, "cpu")
    nl.build(atoms.positions)
    estimate = adaptive.rule_12_d3(natoms=len(atoms), volume_A3=atoms.get_volume(), radius_A=D3_CUTOFF + SKIN)
    assert nl.num_pairs == pytest.approx(estimate.pairs, rel=0.03)
