"""Shared setup for the FastMACE vs TorchSim step-time comparison (conductor, 24 October 2026)."""
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("FASTMACE_STRUCTURES", "/mnt/d/Margraf/ResearchModule"))
MODELS = Path(os.environ.get("FASTMACE_MODELS", str(Path.home() / "models")))
CASES = {  # label -> (checkpoint file, head)
    "mh1-r2scan": ("mace-mh-1.model", "matpes_r2scan"),
    "omat0-medium": ("mace-omat-0-medium.model", None),
}
T_K, DT_FS, FRICTION_PER_FS, SEED = 300.0, 1.0, 0.01, 0


def structure(name, repeat=3):
    import numpy as np
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = read(ROOT / "Structures/Benchmark/R2SCAN" / f"{name}.cif").repeat((repeat,) * 3)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    return atoms


def timed(step_block, warmup, steps, blocks, sync):
    """Warm up, then ``blocks`` timed blocks of ``steps`` steps; returns ms/step per block."""
    step_block(warmup)
    sync()
    out = []
    for _ in range(blocks):
        t = time.perf_counter()
        step_block(steps)
        sync()
        out.append(1e3 * (time.perf_counter() - t) / steps)
    return out


def report(record):
    print("RESULT " + json.dumps(record), flush=True)
