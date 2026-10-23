"""Step time of the fused Nose-Hoover chain stage against the PyTorch stage (plan task 4-T03b).

    python tools/nhc_fused_timing.py --part stages --out results/4-T03b/stages.json
    FASTMACE_MODELS=~/models python tools/nhc_fused_timing.py --part mh1 --out results/4-T03b/mh1.json

Two parts, each in its own process (a Simulation needs the allocator configured before CUDA is touched),
with alternating blocks (fused, PyTorch, then PyTorch, fused, ...) after a warm-up; the median per-step
time per stage is reported:

1. **Stages A + B** of ``NoseHooverChainNVT`` and ``MTK`` (hydrostatic, full), captured as CUDA graphs with a
   harmonic stand-in force field (``tests/gpu/test_nose_hoover_mtk_gpu.py``'s), NaCl 3 x 3 x 3 (216 atoms),
   ``fixcm`` on, with Langevin's stages as the yardstick.
2. **MH-1 end to end**: two ``fm.Simulation`` engines, NaCl 3 x 3 x 3, MH-1 (``mh1-r2scan``), the planner's
   default plan, NHC and full-cell MTK at 600 K, 1 fs. They differ only in the chain stage: the PyTorch one
   is forced by building with ``fused=False`` (the planner builds with the default, fused on CUDA).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import fastmace as fm  # before torch (SPEC §14)

# isort: split
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmace.integrators import mtk, nose_hoover
from fastmace.integrators.registry import registry
from fastmace.neighbors.skin import det_3x3
from fastmace.state import SimulationState

STAGES = ("fused", "pytorch")


def nacl(temperature_K=600.0):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(5))
    return atoms


def record(name, temperature_K=600.0):
    common = dict(timestep_fs=1.0, temperature_K=temperature_K, tdamp_fs=100.0)
    if name == "nhc":
        return fm.integrators.NoseHooverChain(**common)
    if name == "langevin":
        return fm.integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K, friction_per_fs=0.02)
    return fm.integrators.MTK(pressure_GPa=0.0, pdamp_fs=1000.0, cell_mode=name, **common)


class Harmonic:
    def __init__(self, state, k=0.5, bulk_modulus=0.5):
        self.x0 = state.positions.clone()
        self.v0 = det_3x3(state.cell).clone()
        self.k, self.bulk_modulus = k, bulk_modulus
        self.eye = torch.eye(3, dtype=torch.float64, device=state.positions.device)

    def __call__(self, state):
        d = state.positions - self.x0
        state.forces.copy_(-self.k * d)
        state.energy.copy_((0.5 * self.k * (d * d).sum()).reshape(1))
        if state.stress is not None:
            volume = det_3x3(state.cell)
            state.stress.copy_(self.k * (d.T @ d) / volume + self.bulk_modulus * (volume / self.v0 - 1.0) * self.eye)


def stage_graphs(name, fused):
    """Captured stages A and B of the scheme on NaCl with the harmonic field."""
    config = record(name).to_config(seed=1)
    state = SimulationState.from_atoms(nacl(), "cuda", 1, compute_stress=config.ensemble == "npt")
    cls = registry().integrator(config.scheme).load()
    dyn = cls(state, config, fused=fused) if name != "langevin" else cls(state, config, temperature_K=600.0)
    field = Harmonic(state)
    field(state)
    dyn.start(state)

    def stage_a():
        dyn.first_half(state)

    def stage_b():
        field(state)
        dyn.second_half(state)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            stage_a()
            stage_b()
    torch.cuda.current_stream().wait_stream(side)
    graphs = []
    for stage in (stage_a, stage_b):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()
        graphs.append(graph)
    return dyn, graphs


def replay(graphs, dyn, steps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        graphs[0].replay()
        graphs[1].replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def alternate(runners, blocks, block):
    """Median seconds per step of each runner over alternating blocks."""
    names = list(runners)
    times = {name: [] for name in names}
    for index in range(blocks):
        for name in (names if index % 2 == 0 else names[::-1]):
            times[name].append(runners[name](block))
    return {name: statistics.median(values) for name, values in times.items()}, times


def simulation(model, name, fused):
    atoms = nacl()
    original = nose_hoover.use_fused, mtk.use_fused
    if not fused:  # build the PyTorch stage; the planner otherwise builds with the default
        forced = lambda fused, device: False  # noqa: E731
        nose_hoover.use_fused = mtk.use_fused = forced
    try:
        sim = fm.Simulation(atoms, fm.ForceField(model, atoms), integrator=record(name), seed=7,
                            workload=fm.Workload(max_temperature_K=600.0))
    finally:
        nose_hoover.use_fused, mtk.use_fused = original
    assert sim._engine.integrator.fused == fused
    return sim


def timed_run(sim):
    def run(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sim.run(steps=steps)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / steps
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--stage-block", type=int, default=1000)
    parser.add_argument("--block", type=int, default=300)
    parser.add_argument("--part", choices=("stages", "mh1"), required=True)
    args = parser.parse_args(argv)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fm.__version__,
                  atoms=216, stages={}, mh1={})

    if args.part == "stages":
        stages(args, result)
    else:
        mh1(args, result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


def stages(args, result):
    dyn, graphs = stage_graphs("langevin", None)
    result["stages"]["langevin"] = alternate({"langevin": lambda n: replay(graphs, dyn, n)}, 2,
                                             args.stage_block)[0]["langevin"] * 1e6
    for name in ("nhc", "hydrostatic", "full"):
        built = {stage: stage_graphs(name, stage == "fused") for stage in STAGES}
        for _, (dyn, graphs) in built.items():
            replay(graphs, dyn, 200)  # warm-up
        medians, blocks = alternate({stage: (lambda n, b=built[stage]: replay(b[1], b[0], n)) for stage in STAGES},
                                    args.blocks, args.stage_block)
        result["stages"][name] = {stage: medians[stage] * 1e6 for stage in STAGES}
        print(f"stages {name:12s} " + "  ".join(f"{stage} {medians[stage] * 1e6:8.1f} us" for stage in STAGES),
              flush=True)
    print(f"stages langevin     {result['stages']['langevin']:8.1f} us", flush=True)



def mh1(args, result):
    model = fm.load("mh1-r2scan")
    for name in ("nhc", "full"):
        sims = {stage: simulation(model, name, stage == "fused") for stage in STAGES}
        for sim in sims.values():
            sim.run(steps=args.block)  # warm-up: captures, buckets
        medians, blocks = alternate({stage: timed_run(sims[stage]) for stage in STAGES}, args.blocks, args.block)
        result["mh1"][name] = {stage: medians[stage] * 1e3 for stage in STAGES}
        result["mh1"][name]["blocks_ms"] = {stage: [t * 1e3 for t in blocks[stage]] for stage in STAGES}
        print(f"MH-1 {name:5s} " + "  ".join(f"{stage} {medians[stage] * 1e3:7.3f} ms" for stage in STAGES),
              flush=True)
        del sims
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
