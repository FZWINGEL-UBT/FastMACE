"""Per-species temperatures under the default fixcm projection on the GPU (plan task 3-T22c, decision 37).

    FASTMACE_MODELS=~/models python tools/fixcm_default_gpu.py --out results/3-T22c/default.json [--steps 20000]

``fm.Simulation`` with the default plan and no ``fixcm_projection`` given, so the plan's ``"auto"``
resolves to ``"mass_weighted"`` with the fused stage on (both asserted). MH-1 (``mh1-r2scan``), Langevin
1 fs, γ = 0.05/fs, 600 K, after 2,000 steps of equilibration. Three structures: NaCl 3×3×3 (216 atoms),
and Li/Cs NaCl (cations alternately Li and Cs) at 3×3×3 (216) and 2×1×1 (16 atoms). A sink reads every step's
momenta; per species Σ p²/m is averaged in 20 batches. The fixed-centre-of-mass convention gives
Σ p²/m / ((3 Nₛ − 3 Mₛ/M) k_B) per species and 3N − 3 in total, which is what the thermo rows report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import fastmace as fm  # before torch (SPEC §14)

# isort: split
import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

BATCHES = 20


def structure(name):
    if name == "NaCl-216":
        return bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    repeat = (3, 3, 3) if name == "LiCs-216" else (2, 1, 1)
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(repeat)
    numbers = atoms.numbers.copy()
    cations = np.flatnonzero(numbers == 11)
    numbers[cations[0::2]], numbers[cations[1::2]] = 3, 55
    atoms.set_atomic_numbers(numbers)
    return atoms


def run(model, name, steps, equilibration, temperature_K=600.0):
    atoms = structure(name)
    atoms.rattle(0.01, seed=3)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(5))
    Stationary(atoms)
    integrator = fm.integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K, friction_per_fs=0.05)
    sim = fm.Simulation(atoms, fm.ForceField(model, atoms), integrator=integrator, seed=7,
                        workload=fm.Workload(max_temperature_K=temperature_K))
    engine = sim._engine
    decision = next(entry for entry in sim.plan.decisions if entry[0] == "IntegratorConfig.fixcm_projection")
    assert sim.plan.integrator.fixcm_projection == "mass_weighted", decision
    assert engine._fused is not None, engine.fused_integrator_reason
    numbers, masses = atoms.numbers, atoms.get_masses()
    species = sorted(set(numbers.tolist()))
    masks = np.stack([numbers == z for z in species]).astype(np.float64)
    sums = np.zeros((BATCHES, len(species)))
    per_batch = steps // BATCHES
    seen = {"count": 0}

    def sink(step, values):
        index = seen["count"]
        seen["count"] += 1
        if index < equilibration:
            return
        batch = (index - equilibration) // per_batch
        if batch < BATCHES:
            momenta = np.asarray(values["momenta"])
            sums[batch] += masks @ ((momenta ** 2).sum(axis=1) / masses)

    sim.add_sink(sink, 1)
    started = time.perf_counter()
    sim.run(steps=equilibration + per_batch * BATCHES)
    sim.flush_output()
    seconds = time.perf_counter() - started
    total_mass = masses.sum()
    batches = sums / per_batch / units.kB
    out = {"natoms": len(atoms), "steps": per_batch * BATCHES, "seconds": seconds,
           "decision": [str(x) for x in decision],
           "degrees_of_freedom": sim.degrees_of_freedom, "species": {}}

    def estimate(values):
        return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))

    for index, z in enumerate(species):
        count = masks[index].sum()
        dof = 3 * count - 3 * float((masks[index] * masses).sum()) / total_mass
        symbol = atoms[int(np.flatnonzero(numbers == z)[0])].symbol
        out["species"][symbol] = {"count": int(count), "T_fixed_com": estimate(batches[:, index] / dof),
                                  "T_3N": estimate(batches[:, index] / (3 * count))}
    total = batches.sum(axis=1)
    out["total"] = {"T_3N_minus_3": estimate(total / (3 * len(atoms) - 3)), "T_3N": estimate(total / (3 * len(atoms)))}
    sim.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--cases", default="NaCl-216,LiCs-216,LiCs-16")
    args = parser.parse_args(argv)
    model = fm.load("mh1-r2scan")
    record = {"torch": torch.__version__, "model": "mh1-r2scan",  # the GPU's name after the runs: the plan sets
              # the allocator before CUDA is touched (SPEC §8.7)
              "temperature_K": 600.0, "friction_per_fs": 0.05, "timestep_fs": 1.0, "runs": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases.split(","):
        result = run(model, name, args.steps, 2000)
        record["runs"][name] = result
        record["gpu"] = torch.cuda.get_device_name(0)
        out.write_text(json.dumps(record, indent=1))
        parts = ", ".join(f"{s} {v['T_fixed_com'][0]:.1f} ± {v['T_fixed_com'][1]:.1f}"
                          for s, v in result["species"].items())
        print(f"{name}: total(3N-3) {result['total']['T_3N_minus_3'][0]:.1f} ± "
              f"{result['total']['T_3N_minus_3'][1]:.1f} K; {parts}; {result['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
