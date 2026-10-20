"""2-T14e: does a shorter Langevin ramp keep the ASE-equivalence test's power? (CPU stand-in)

The acceptance test runs ASE's Langevin and FastMACE's from one seed and compares the mean T and PE
per 50 K target bin (|dT| < 0.1 K, |dPE| < 0.01 meV/atom). Here ASE's Langevin plays both sides on a
Lennard-Jones solid (fcc, 256 atoms, epsilon 0.1 eV so that it stays solid to 400 K), with the test's
timestep, friction and ramp shape. Each run pairs a baseline with a variant carrying one injected
difference, from the same seed, and reports the test's two metrics and the final temperature.

usage: python langevin_power.py <heat_steps> <cool_steps> <variant>   (prints one JSON line)
"""

import json
import sys
import warnings

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md.langevin import Langevin

from fastmace.config import IntegratorConfig

HEAT, COOL, VARIANT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PEAK, BIN, SEED = 400.0, 50.0, 20261019
CONFIG = IntegratorConfig(ensemble="nvt", seed=SEED)


class Scaled(LennardJones):
    """Forces (not the energy) scaled by a factor: a force error that the energy does not show."""

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.results["forces"] = self.results["forces"] * self.factor


def target(step, late=0):
    step = max(step - late, 0)
    return PEAK * step / HEAT if step <= HEAT else PEAK * max(1.0 - (step - HEAT) / COOL, 0.0)


def run(variant):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((4, 4, 4))
    factor = {"force_1e-6": 1 + 1e-6, "force_1e-9": 1 + 1e-9}.get(variant, 1.0)
    atoms.calc = Scaled(factor, sigma=3.4, epsilon=0.1, rc=8.5, smooth=True)
    friction = CONFIG.friction * (1.01 if variant == "friction_1pct" else 1.0)
    late = 1 if variant == "target_one_step_late" else 0
    scale = 1.002 if variant == "temperature_0.2pct" else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dyn = Langevin(atoms, timestep=CONFIG.timestep_fs * units.fs, friction=friction, fixcm=CONFIG.fixcm,
                       temperature_K=target(0, late) * scale, rng=np.random.default_rng(SEED))
    n, rows = len(atoms), []
    for step in range(1, HEAT + COOL + 1):
        dyn.set_temperature(temperature_K=target(step, late) * scale)
        dyn.run(1)
        rows.append((step, target(step), 2 * atoms.get_kinetic_energy() / (3 * n * units.kB),
                     atoms.get_potential_energy() / n))
    return np.array(rows)


def bins(rows):
    segment = np.where(rows[:, 0] <= HEAT, 0, 1)
    index = np.minimum(np.floor(rows[:, 1] / BIN), PEAK / BIN - 1).astype(int)
    return {key: (rows[(segment == key[0]) & (index == key[1]), 2].mean(),
                  rows[(segment == key[0]) & (index == key[1]), 3].mean())
            for key in sorted(set(zip(segment.tolist(), index.tolist())))}


base, other = run("baseline"), run(VARIANT)
b, o = bins(base), bins(other)
dT = max(abs(o[k][0] - b[k][0]) for k in b)
dPE = max(abs(o[k][1] - b[k][1]) for k in b) * 1e3
print(json.dumps({"heat": HEAT, "cool": COOL, "variant": VARIANT, "bins": len(b), "max_dT_K": dT,
                  "max_dPE_meV_per_atom": dPE, "detected": bool(dT >= 0.1 or dPE >= 0.01),
                  "final_T_K": float(base[-1, 2]), "steps": HEAT + COOL}))
