"""Per-species kinetic temperatures under the Langevin ``fixcm`` projections (plan task 3-T22).

    python tools/fixcm_bias.py --out results/3-T22/bias.json [--cases NaCl-216,LiCs-16,LiCs-16-mace]
        [--temperatures 600,1500] [--projections ase,off,mass_weighted] [--scale 1.0] [--mace-steps N]
        [--nanbocl4 PATH]

A case with the suffix ``-mace`` runs through the planner and the graph engine with a tiny random MACE
model instead of the springs (``run_tiny_mace``).

FastMACE's ``Langevin`` integrator in float64 on the CPU, driven step by step (``prepare_step``,
``first_half``, forces, ``second_half``, as the engine's stages do). The force field is a translation-
invariant network of harmonic springs, k = 1 eV/Å², between the pairs within a cutoff of the starting
structure. The momentum marginal of the canonical distribution is Maxwell–Boltzmann for any potential,
so the per-species kinetic temperature tests the thermostat itself; the springs only couple the species.

Each case starts from Maxwell–Boltzmann velocities at the target with no centre-of-mass motion. The
timestep is 1 fs; γ = 0.05/fs. After ``equilibration`` steps, every step is sampled, with batch means
over 20 batches. Reported per species s and in total, from Σ m v²:
- the 3N convention: Σ m v² / (3 N_s k_B);
- the fixed-centre-of-mass convention: Σ m v² / ((3 N_s − 3 M_s / M) k_B). Summed over the species,
  that is 3N − 3 degrees of freedom.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import fastmace  # noqa: F401  (before torch: file locks, SPEC §14)

# isort: split
import torch
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

DEFAULT_NANBOCL4 = "/mnt/g/Margraf/ResearchModule/Structures/NaMOCl4/NaNbOCl4_I4m_Ma2026.cif"
NBCL5_CELLPAR = (13.678, 13.678, 6.413, 89.411, 89.411, 85.947)  # tests/_support/structures.py
BATCHES = 20


def nbcl5(repeat):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from _support.structures import nbcl5 as build

    return build(repeat=1).repeat(repeat)


def licscl(repeat):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(repeat)
    numbers = atoms.numbers.copy()
    cations = np.flatnonzero(numbers == 11)
    numbers[cations[0::2]], numbers[cations[1::2]] = 3, 55
    atoms.set_atomic_numbers(numbers)
    return atoms


def structure(case, nanbocl4):
    from ase.io import read

    if case == "NaCl-216":
        return bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3)), 3.2
    if case == "NbCl5-1944":
        return nbcl5((3, 3, 6)), 3.8
    if case == "NaNbOCl4-756":
        return read(nanbocl4).repeat((3, 3, 3)), 3.6
    if case == "LiCs-16":
        return licscl((2, 1, 1)), 3.2
    if case == "LiCs-216":
        return licscl((3, 3, 3)), 3.2
    raise ValueError(f"unknown case {case!r}")


def springs(atoms, cutoff):
    from ase.neighborlist import neighbor_list

    i, j, shift = neighbor_list("ijS", atoms, cutoff)
    keep = i < j
    i, j = torch.tensor(i[keep]), torch.tensor(j[keep])
    shifts = torch.tensor(shift[keep] @ atoms.cell.array)
    positions = torch.tensor(atoms.get_positions())
    lengths = torch.linalg.norm(positions[j] + shifts - positions[i], dim=1, keepdim=True)
    return i, j, shifts, lengths


def forces(positions, network, out, k=1.0):
    i, j, shifts, lengths = network
    d = positions[j] + shifts - positions[i]
    r = torch.linalg.norm(d, dim=1, keepdim=True)
    f = k * (r - lengths) * d / r
    out.zero_()
    out.index_add_(0, i, f)
    out.index_add_(0, j, -f)


def run_case(atoms, cutoff, temperature_K, projection, steps, equilibration, seed=11):
    from fastmace.config import IntegratorConfig
    from fastmace.integrators.langevin import Langevin
    from fastmace.state import SimulationState

    atoms = atoms.copy()
    atoms.rattle(0.01, seed=seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    Stationary(atoms)
    network = springs(atoms, cutoff)
    state = SimulationState.from_atoms(atoms, "cpu", seed)
    options = {"fixcm": False} if projection == "off" else {"fixcm": True, "fixcm_projection": projection}
    config = IntegratorConfig(timestep_fs=1.0, friction=0.05 / units.fs, noise_batch=256, **options)
    dyn = Langevin(state, config, temperature_K=temperature_K)
    numbers = atoms.numbers
    species = sorted(set(numbers.tolist()))
    masks = torch.stack([torch.tensor(numbers == z) for z in species]).to(torch.float64)  # (S, N)
    masses = state.masses[:, 0]
    total_mass = float(masses.sum())
    group_mass = (masks * masses).sum(dim=1)
    counts = masks.sum(dim=1)
    sums = torch.zeros((BATCHES, len(species)), dtype=torch.float64)
    per_batch = steps // BATCHES
    forces(state.positions, network, state.forces)
    started = time.perf_counter()
    for step in range(equilibration + per_batch * BATCHES):
        dyn.prepare_step(state)
        dyn.first_half(state)
        forces(state.positions, network, state.forces)
        dyn.second_half(state)
        if step >= equilibration:
            twice_kinetic = masks @ (masses * (state.velocities ** 2).sum(dim=1))  # per species: Σ m v²
            sums[(step - equilibration) // per_batch] += twice_kinetic
    return _result(atoms, sums / per_batch / units.kB, species, counts, group_mass, total_mass, steps, equilibration,
                   time.perf_counter() - started, state)


def run_tiny_mace(atoms, temperature_K, projection, steps, equilibration, directory, seed=11):
    """The same measurement through the planner and the graph engine (CPU, eager stages), with a tiny random
    MH-1-like MACE model in float64 instead of the springs: 3-T12's setting."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.engine import constraints
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    atoms = atoms.copy()
    atoms.rattle(0.01, seed=seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    Stationary(atoms)
    planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none", pipelined="off"),
                           environ={})
    path = Path(directory) / "tiny-mh1-3-17-55.model"
    if not path.exists():
        torch.save(tiny_mace_model(sorted(set(atoms.numbers.tolist())), **FAMILIES["mh1"]), path)
    loaded = planner.load(str(path), trusted=True)
    options = {"fixcm": False} if projection == "off" else {"fixcm": True, "fixcm_projection": projection}
    config = IntegratorConfig(timestep_fs=1.0, friction=0.05 / units.fs, noise_batch=256, seed=seed, **options)
    plan = planner.plan(loaded, atoms, config, Hardware("cpu", (0, 0), 0))
    engine = planner.build(plan, loaded, atoms, temperature_K=temperature_K, constraints=constraints.ProcessRecord())
    assert engine.integrator.config.fixcm_projection == config.fixcm_projection
    state = engine.state
    numbers = atoms.numbers
    species = sorted(set(numbers.tolist()))
    masks = torch.stack([torch.tensor(numbers == z) for z in species]).to(torch.float64)
    masses = state.masses[:, 0]
    total_mass, group_mass, counts = float(masses.sum()), (masks * masses).sum(dim=1), masks.sum(dim=1)
    sums = torch.zeros((BATCHES, len(species)), dtype=torch.float64)
    per_batch = steps // BATCHES
    started = time.perf_counter()
    for step in range(equilibration + per_batch * BATCHES):
        engine.step()
        if step >= equilibration:
            sums[(step - equilibration) // per_batch] += masks @ (masses * (state.velocities ** 2).sum(dim=1))
    return _result(atoms, sums / per_batch / units.kB, species, counts, group_mass, total_mass, steps, equilibration,
                   time.perf_counter() - started, state)


def run_mc_check(temperature_K, projection, mc, steps, equilibration, directory, friction_per_fs=0.02, seed=11):
    """``redraw_all`` against the thermostat (3-T12's addendum): (Na,K)Cl 2×1×1, a tiny MACE model with the
    superset (Na, Cl, K), Na↔K swaps every 10 fs with ``redraw_all`` and real Metropolis decisions (``mc``), or
    none. Through a ``Simulation``'s engine and ``fastmace.mc.MCDriver``, as ``tests/cpu/test_mc_driver.py``.

    The public ``fm.integrators.Langevin`` record has no ``fixcm_projection`` (the default is the owner's
    decision), so ``"mass_weighted"`` is set on the built engine's integrator here, before the first step."""
    import dataclasses
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from _support.mc_fixtures import mc_simulation, tiny_loaded
    from _support.structures import nacl
    from fastmace import integrators
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.mc import MCDriver, Swap
    from fastmace.planner import minimal

    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    minimal.default_planner = lambda force_field: planner
    atoms = nacl(repeat=1).repeat((2, 1, 1))
    numbers = atoms.numbers.copy()
    numbers[np.flatnonzero(numbers == 11)[1::2]] = 19
    atoms.set_atomic_numbers(numbers)
    atoms.rattle(0.02, seed=seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    Stationary(atoms)
    loaded = tiny_loaded(Path(directory), (11, 17, 19))
    record = integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K, friction_per_fs=friction_per_fs)
    sim = mc_simulation(loaded, atoms, (11, 17, 19), integrator=record, seed=seed)
    engine = sim._engine
    dyn = engine.integrator
    if projection == "mass_weighted":
        dyn.config = dataclasses.replace(dyn.config, fixcm_projection="mass_weighted")
        dyn.mass_weighted_com = True
    driver = None
    if mc:
        driver = MCDriver([Swap([("Na", "K")], every=10, attempts=4, velocities="redraw_all")], seed=seed)
        driver.attach(engine)
    state = engine.state
    species = sorted(set(numbers.tolist()))
    masks = torch.stack([torch.tensor(numbers == z) for z in species]).to(torch.float64)
    masses = state.masses[:, 0]
    sums = torch.zeros((BATCHES, len(species)), dtype=torch.float64)
    per_batch = steps // BATCHES
    accepted = attempted = 0
    started = time.perf_counter()
    for index in range(equilibration + per_batch * BATCHES):
        engine.step()
        step = int(state.step)
        if driver is not None:
            block = driver.run_block(engine, step)
            for move in (block or {}).get("moves", ()):
                accepted += move["accepted"]
                attempted += move["attempted"]
        if index >= equilibration:
            # The species move with the swaps: the masks follow the committed species.
            masks = torch.stack([(state.species == z) for z in species]).to(torch.float64)
            masses = state.masses[:, 0]
            sums[(index - equilibration) // per_batch] += masks @ (masses * (state.velocities ** 2).sum(dim=1))
    group_mass = (masks * masses).sum(dim=1)
    result = _result(atoms, sums / per_batch / units.kB, species, masks.sum(dim=1), group_mass, float(masses.sum()),
                     steps, equilibration, time.perf_counter() - started, state)
    result["mc"] = {"accepted": accepted, "attempted": attempted} if mc else None
    return result


def _result(atoms, batches, species, counts, group_mass, total_mass, steps, equilibration, seconds, state):
    numbers = atoms.numbers
    result = {"natoms": len(atoms), "steps": steps // BATCHES * BATCHES, "equilibration": equilibration,
              "seconds": seconds, "momentum": (state.velocities * state.masses).sum(dim=0).tolist(), "species": {}}

    def estimate(values):
        return float(values.mean()), float(values.std(correction=1) / math.sqrt(len(values)))

    for index, z in enumerate(species):
        dof_3n = 3.0 * float(counts[index])
        dof_com = dof_3n - 3.0 * float(group_mass[index]) / total_mass
        values = batches[:, index]
        result["species"][atoms[int(np.flatnonzero(numbers == z)[0])].symbol] = {
            "count": int(counts[index]), "T_3N": estimate(values / dof_3n), "T_fixed_com": estimate(values / dof_com)}
    total = batches.sum(dim=1)
    n = len(atoms)
    result["total"] = {"T_3N": estimate(total / (3.0 * n)), "T_3N_minus_3": estimate(total / (3.0 * n - 3.0))}
    return result


def default_steps(natoms, scale):
    # 0.1 % on the total needs about 2 / (3N) / 1e-6 independent samples, one per ~20 steps (2 γ = 0.1/fs).
    return int(scale * max(60_000, 1.5e7 / natoms))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="LiCs-16,LiCs-216,NaCl-216,NaNbOCl4-756,NbCl5-1944")
    parser.add_argument("--temperatures", default="600,1500")
    parser.add_argument("--projections", default="ase,off,mass_weighted")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--nanbocl4", default=DEFAULT_NANBOCL4)
    parser.add_argument("--mace-steps", type=int, default=60_000, help="steps of a '<case>-mace' run")
    parser.add_argument("--mc-check", action="store_true",
                        help="run_mc_check instead: (Na,K)Cl 16, redraw_all swaps on and off, both projections")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(out.read_text()) if out.exists() else {"runs": {}}
    record.update({"integrator": "fastmace.integrators.langevin.Langevin", "timestep_fs": 1.0,
                   "friction_per_fs": 0.05, "force_field": "harmonic springs, k = 1 eV/A^2", "batches": BATCHES})
    if args.mc_check:
        for temperature in (float(t) for t in args.temperatures.split(",")):
            for projection in ("ase", "mass_weighted"):
                for mc in (False, True):
                    key = f"NaK-16-mace-mc/{temperature:g}K/{projection}/mc-{'on' if mc else 'off'}"
                    if key in record["runs"]:
                        continue
                    result = run_mc_check(temperature, projection, mc, args.mace_steps, 2000, out.parent)
                    result.update({"case": "NaK-16-mace-mc", "temperature_K": temperature, "projection": projection})
                    record["runs"][key] = result
                    out.write_text(json.dumps(record, indent=1))
                    print(f"{key}: total(3N) {result['total']['T_3N'][0]:.1f} ± {result['total']['T_3N'][1]:.1f}, "
                          f"(3N-3) {result['total']['T_3N_minus_3'][0]:.1f} K; mc {result['mc']}; "
                          f"{result['seconds']:.0f} s", flush=True)
        return
    for case in args.cases.split(","):
        mace = case.endswith("-mace")
        atoms, cutoff = structure(case.removesuffix("-mace"), args.nanbocl4)
        steps = args.mace_steps if mace else default_steps(len(atoms), args.scale)
        for temperature in (float(t) for t in args.temperatures.split(",")):
            for projection in args.projections.split(","):
                key = f"{case}/{temperature:g}K/{projection}"
                if key in record["runs"]:
                    continue
                if mace:
                    result = run_tiny_mace(atoms, temperature, projection, steps, 2000, out.parent)
                else:
                    result = run_case(atoms, cutoff, temperature, projection, steps, equilibration=2000)
                result.update({"case": case, "temperature_K": temperature, "projection": projection})
                record["runs"][key] = result
                out.write_text(json.dumps(record, indent=1))
                parts = ", ".join(f"{name} {values['T_3N'][0]:.1f}" for name, values in result["species"].items())
                print(f"{key}: total(3N) {result['total']['T_3N'][0]:.1f} ± {result['total']['T_3N'][1]:.1f} K; "
                      f"{parts}; {result['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
