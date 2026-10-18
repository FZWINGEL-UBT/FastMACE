"""Reproducer: does a new edge bucket after warm-up compile? FastMACE (planner) or the research engine.

usage: python repro_bucket.py fastmace|baseline <baseline root> [steps] [temperature] [fraction]
Prints one JSON line: per new padded edge size, the step, dynamo counters before/after, step time.
"""
import json
import sys
import time

side = sys.argv[1]
root = sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0
fraction = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

import numpy as np  # noqa: E402

CALLS = []  # (edges, capturing) of every force call, runs collapsed


def trace(cls, edges_of):
    import torch
    original = cls.__call__

    def wrapped(self, positions):
        entry = [edges_of(self), bool(torch.cuda.is_current_stream_capturing())]
        if CALLS and CALLS[-1][:2] == entry:
            CALLS[-1][2] += 1
        else:
            CALLS.append(entry + [1])
        return original(self, positions)
    cls.__call__ = wrapped


def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(11))
    return atoms


if side == "fastmace":
    import fastmace
    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.planner import fixed
    import logging
    import torch
    torch._logging.set_logs(recompiles=True, dynamic=logging.INFO)
    from fastmace.model.forces import MACEForceField
    trace(MACEForceField, lambda field: int(field.data["edge_index"].shape[1]))
    planner = fixed.FixedPlanner(engine=EngineConfig(backend="hybrid"),
                                 neighbours=NeighbourConfig(edge_bucket_fraction=fraction))
    hardware = planner.hardware()
    loaded = planner.load("mh1-r2scan")
    import torch
    atoms = structure()
    plan = planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=20261018), hardware)
    t0 = time.perf_counter()
    engine = planner.build(plan, loaded, atoms, temperature_K=temperature)
    build_s = time.perf_counter() - t0

    def step():
        engine.step()

    def flush():
        engine.flush()

    def sizes():
        return set(engine.edge_sizes_used)
    bucket = engine.edge_bucket
else:
    sys.path[:0] = [root + "/Experiments/Langevin_Thermostat", root + "/Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts"]
    from langevin import gpu_langevin
    gpu_langevin.configure_allocator()
    import common
    import logging
    import torch
    torch._logging.set_logs(recompiles=True, dynamic=logging.INFO)
    trace(gpu_langevin.MACEForces, lambda field: int(field.data["edge_index"].shape[1]))
    from ase import units
    from langevin import upstream_mace, workflow
    upstream_mace.use_single_graph_strain(True)
    upstream_mace.use_cueq_conv_fusion(True)
    calc = common.load_calculator("mh1-r2scan", dtype="float32", enable_cueq=True, enable_oeq=True)
    upstream_mace.require_fused_cueq(calc.models[0])
    workflow.configure_md_calculator(calc, compute_stress=False)
    options = dict(factor_species=True, fold_cutoff=True, density_first=True, joint_radial_input=True,
                   fused_folded_backward=True, reassociate_first=True, segmented_first_sum=True, block_linear=True,
                   grouped_linear=True, merged_residual=True, streaming_conv=True, species_skip=True)
    t0 = time.perf_counter()
    atoms = structure()
    eng = workflow.GPUEngine(atoms, calc, timestep_fs=0.5, friction=0.02, seed=20261018, skin=0.3,
                             compile_mode="default", gpu_neighbour_list=True, sort_receivers=True, pipelined=True,
                             ensemble="nvt", **options)
    dyn = eng.dyn
    if fraction != gpu_langevin.EDGE_BUCKET_FRACTION:
        atoms = structure()
        dyn = gpu_langevin.GPULangevin(atoms, eng.field, timestep=0.5 * units.fs, friction=0.02,
                                       rng=np.random.default_rng(20261018), temperature_K=temperature, fixcm=True,
                                       cuda_graph=True, pipelined=True, edge_bucket_fraction=fraction)
    dyn.set_temperature(temperature)
    build_s = time.perf_counter() - t0

    def step():
        dyn.step()

    def flush():
        dyn.flush_speculative()

    def sizes():
        return set(dyn.edge_sizes_used)
    bucket = dyn.edge_bucket

from torch._dynamo.utils import counters  # noqa: E402


def snap():
    return {"unique_graphs": counters["stats"]["unique_graphs"], "frames_ok": counters["frames"]["ok"],
            "frames_total": counters["frames"]["total"]}


events, seen = [], set()
warm = 20
times = []
for n in range(1, steps + 1):
    if n == warm + 1:
        flush(); torch.cuda.synchronize()
        after_warmup = snap()
        seen = sizes()
    before = snap()
    t = time.perf_counter()
    step()
    flush(); torch.cuda.synchronize()
    dt = time.perf_counter() - t
    times.append(dt)
    if n > warm and sizes() != seen:
        new = sorted(sizes() - seen)
        seen = sizes()
        events.append({"step": n, "new_sizes": new, "before": before, "after": snap(), "step_s": dt})
print(json.dumps({"side": side, "bucket": bucket, "build_s": build_s, "after_warmup": after_warmup,
                  "end": snap(), "sizes": sorted(sizes()), "events": events,
                  "calls_to_first_replay_and_new_sizes": CALLS[:40], "median_ms": float(np.median(times[warm:]) * 1e3), "max_ms": float(max(times[warm:]) * 1e3),
                  "torch": torch.__version__}))
