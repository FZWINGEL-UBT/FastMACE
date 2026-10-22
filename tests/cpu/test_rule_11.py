"""Rule 11 (SPEC §9.3; plan task 3-T05): skin, headroom and re-capture budget from the workload."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase import units
from ase.data import atomic_masses

from fastmace.config import DEFAULT_SKIN_A, NeighbourConfig
from fastmace.neighbors import adaptive
from fastmace.neighbors.adaptive import rule_11
from fastmace.neighbors.skin import SkinNeighbourList

NA, OXYGEN, CL = atomic_masses[11], atomic_masses[8], atomic_masses[17]


def rule(temperature, mass, dt, natoms, displacement=None, cutoff=6.0, fraction=0.05):
    return rule_11(max_temperature_K=temperature, lightest_mass_u=mass, timestep_fs=dt, natoms=natoms,
                   cutoff_A=cutoff, edge_bucket_fraction=fraction, max_displacement_A=displacement)


def test_thermal_speed_in_angstrom_per_fs_matches_ase_units():
    sigma = math.sqrt(units.kB * 1500.0 / NA) * units.fs  # ASE: eV, u, Å; time unit sqrt(u Å^2 / eV)
    assert adaptive.thermal_speed(1500.0, NA) == pytest.approx(sigma, rel=1e-6)


def test_fastest_speed_quantile():
    z = adaptive.fastest_speed_quantile(512)
    assert 512 * adaptive.maxwell_tail(z) == pytest.approx(1.0, rel=1e-9)
    assert z == pytest.approx(3.853, abs=1e-3)
    assert adaptive.fastest_speed_quantile(1296) > z  # more atoms, a faster fastest one
    assert adaptive.fastest_speed_quantile(1) == pytest.approx(math.sqrt(3.0))  # the rms speed, at least
    assert adaptive.maxwell_tail(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("mass, natoms", [(NA, 64), (NA, 512), (CL, 1296), (OXYGEN, 1080)])
def test_the_floor_holds_for_the_ionic_crystals_at_300_K_and_half_a_femtosecond(mass, natoms):
    """NaCl (Na), NbCl5 (Cl) and NaNbO3 (O) keep phase 1a's 0.3 Å and 1.15 at the library defaults."""
    sized = rule(300.0, mass, 0.5, natoms)
    assert (sized.skin_A, sized.capacity_headroom) == (DEFAULT_SKIN_A, adaptive.DEFAULT_HEADROOM)
    assert 2 * adaptive.TARGET_REBUILD_STEPS * sized.step_displacement_A < DEFAULT_SKIN_A


@pytest.mark.parametrize("temperature, mass, dt, natoms, skin, headroom, budget", [
    (1500.0, NA, 1.0, 512, 0.91, 1.20, 9),   # NaCl 4x4x4, the 1500 K design point
    (1500.0, CL, 1.0, 1296, 0.78, 1.18, 9),  # NbCl5, Cl the lightest
    (1000.0, NA, 1.0, 512, 0.75, 1.17, 9),   # the production case: 1 fs, 1000 K
    (1500.0, NA, 0.5, 512, 0.46, 1.15, 7),
])
def test_the_rule_values(temperature, mass, dt, natoms, skin, headroom, budget):
    sized = rule(temperature, mass, dt, natoms)
    assert (sized.skin_A, sized.capacity_headroom, sized.recapture_budget) == (skin, headroom, budget)
    # skin = 2 n* z_N sigma dt, rounded up to 0.01 Å
    raw = 2 * adaptive.TARGET_REBUILD_STEPS * adaptive.fastest_speed_quantile(natoms) * dt * math.sqrt(
        units.kB * temperature / mass) * units.fs
    assert raw <= skin < raw + 0.01
    assert headroom == max(1.15, math.ceil(100 * (1 + 1.5 * skin / (6.0 + skin))) / 100)
    assert "skin" in sized.text() and f"{temperature:g} K" in sized.text()


def test_the_skin_scales_with_speed_and_timestep_above_the_floor():
    base = rule(1500.0, NA, 1.0, 512).md_skin_A
    assert rule(1500.0, NA, 2.0, 512).md_skin_A == pytest.approx(2 * base, abs=0.011)
    assert rule(6000.0, NA, 1.0, 512).md_skin_A == pytest.approx(2 * base, abs=0.011)
    assert rule(1500.0, NA / 4, 1.0, 512).md_skin_A == pytest.approx(2 * base, abs=0.011)


def test_displacement_moves_add_twice_the_largest_displacement():
    sized = rule(300.0, NA, 0.5, 512, displacement=0.2)
    assert sized.md_skin_A == DEFAULT_SKIN_A
    assert sized.skin_A == pytest.approx(DEFAULT_SKIN_A + 0.4)
    assert sized.capacity_headroom == 1.16 and sized.recapture_budget == 9
    assert "displacement" in sized.text()


def test_no_budget_without_edge_buckets():
    assert rule(1500.0, NA, 1.0, 512, fraction=0.0).recapture_budget is None


@pytest.mark.parametrize("kwargs", [{"dt": 0.0}, {"cutoff": 0.0}, {"natoms": 0}, {"mass": 0.0}])
def test_bad_inputs_raise(kwargs):
    args = dict(temperature=300.0, mass=NA, dt=0.5, natoms=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        rule(**args)


def _edges(positions, cell, skin):
    neighbours = SkinNeighbourList(cell, 5.0, "cpu", torch.float64, NeighbourConfig(skin=skin),
                                   sort_receivers=True, gpu_build=False)
    neighbours.build_gpu(torch.as_tensor(positions, dtype=torch.float64))
    positions = torch.as_tensor(positions, dtype=torch.float64)
    neighbours.compact(positions)
    count = int(neighbours.count)
    index, shifts, unit_shifts = neighbours.edges(count)
    return index.clone(), unit_shifts.clone()


def test_a_larger_skin_keeps_the_sorted_edges_within_the_cutoff():
    """With receiver-sorted lists, the skin changes no edge within the cutoff and not their order.

    So a larger skin changes only the rebuild rate, not the forces: the evidence for rule 11 as an
    MD-only default later (conductor answer 3-T05 1-2).
    """
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.15, seed=3)
    for step in range(3):  # the same positions moved, as between rebuilds
        positions = atoms.positions + 0.1 * step * np.sin(np.arange(atoms.positions.size)).reshape(-1, 3)
        small = _edges(positions, atoms.cell.array, DEFAULT_SKIN_A)
        large = _edges(positions, atoms.cell.array, 0.9)
        assert small[0].shape[1] > 0
        assert torch.equal(small[0], large[0]) and torch.equal(small[1], large[1])


def test_the_fused_integrator_thresholds_follow_the_planned_skin():
    """The fused stage reads the list's skin at construction, as ``needs_rebuild`` forms its limits."""
    from types import SimpleNamespace

    from _support.structures import nacl
    from fastmace.config import EngineConfig
    from fastmace.kernels.triton.fused_integrator import FusedIntegrator

    atoms = nacl()
    sized = rule(1500.0, NA, 1.0, len(atoms))
    neighbours = SkinNeighbourList(atoms.cell.array, 5.0, "cpu", torch.float64, NeighbourConfig(skin=sized.skin_A),
                                   sort_receivers=True, gpu_build=False)
    config = EngineConfig()
    integrator = SimpleNamespace(natoms=len(atoms), fix_com=True, c1=0.9, c2=0.1, dt=1.0)
    state = SimpleNamespace(positions=torch.zeros(len(atoms), 3, dtype=torch.float64),
                            masses=torch.tensor(atoms.get_masses(), dtype=torch.float64))
    engine = SimpleNamespace(integrator=integrator, state=state, config=config,
                             force_field=SimpleNamespace(neighbours=neighbours))
    fused = FusedIntegrator(engine)
    half = sized.skin_A / 2
    assert fused.thresholds.tolist() == [half ** 2, (config.pipeline_rebuild_fraction * half) ** 2]
