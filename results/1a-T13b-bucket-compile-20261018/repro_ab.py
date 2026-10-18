"""Same-process reproducer: the research engine and FastMACE in one process, alternating blocks.

usage: python repro_ab.py <baseline root> <order: bf|fb> [blocks] [block_steps] [temperature]
Prints one JSON line: per side and new padded edge size, the step and dynamo counters before/after.
"""
import json
import sys
import time

root, order = sys.argv[1], sys.argv[2]
blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 12
block_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 150
temperature = float(sys.argv[5]) if len(sys.argv) > 5 else 900.0

import fastmace  # noqa: E402,F401

sys.path[:0] = [root + "/Experiments/Langevin_Thermostat", root + "/Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts"]
from langevin import gpu_langevin  # noqa: E402

gpu_langevin.configure_allocator()
import numpy as np  # noqa: E402


def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(11))
    return atoms


def build_baseline():
    import common
    from langevin import upstream_mace, workflow
    upstream_mace.use_single_graph_strain(True)
    upstream_mace.use_cueq_conv_fusion(True)
    calc = common.load_calculator("mh1-r2scan", dtype="float32", enable_cueq=True, enable_oeq=True)
    upstream_mace.require_fused_cueq(calc.models[0])
    workflow.configure_md_calculator(calc, compute_stress=False)
    options = dict(factor_species=True, fold_cutoff=True, density_first=True, joint_radial_input=True,
                   fused_folded_backward=True, reassociate_first=True, segmented_first_sum=True, block_linear=True,
                   grouped_linear=True, merged_residual=True, streaming_conv=True, species_skip=True)
    eng = workflow.GPUEngine(structure(), calc, timestep_fs=0.5, friction=0.02, seed=20261018, skin=0.3,
                             compile_mode="default", gpu_neighbour_list=True, sort_receivers=True, pipelined=True,
                             ensemble="nvt", **options)
    eng.dyn.set_temperature(temperature)
    return dict(step=eng.dyn.step, flush=eng.dyn.flush_speculative, sizes=lambda: set(eng.dyn.edge_sizes_used),
                before=eng.field.bind_streaming_count, keep=eng)


def build_fastmace():
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.planner import fixed
    planner = fixed.FixedPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = planner.load("mh1-r2scan")
    atoms = structure()
    plan = planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=20261018), hardware)
    engine = planner.build(plan, loaded, atoms, temperature_K=temperature)
    return dict(step=engine.step, flush=engine.flush, sizes=lambda: set(engine.edge_sizes_used),
                before=lambda: None, keep=engine)


import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

torch._logging.set_logs(recompiles=True)


def snap():
    return {"unique_graphs": counters["stats"]["unique_graphs"], "frames_ok": counters["frames"]["ok"]}


sides = {}
builds = {}
for name in ("baseline", "fastmace") if order == "bf" else ("fastmace", "baseline"):
    t0 = time.perf_counter()
    print(f"=== build {name}", file=sys.stderr, flush=True)
    sides[name] = build_baseline() if name == "baseline" else build_fastmace()
    for _ in range(20):  # warm-up steps, as the A/B protocol
        sides[name]["before"]()
        sides[name]["step"]()
    sides[name]["flush"]()
    torch.cuda.synchronize()
    builds[name] = {"seconds": time.perf_counter() - t0, "after": snap(), "sizes": sorted(sides[name]["sizes"]())}
    if name == "fastmace" and order == "fb":
        # As T19's harness: the research engine refuses to install its addmm pass over FastMACE's.
        import torch._inductor.config as inductor_config
        builds["cleared_pass"] = repr(inductor_config.post_grad_custom_pre_pass)
        inductor_config.post_grad_custom_pre_pass = None

events = []
seen = {name: side["sizes"]() for name, side in sides.items()}
times = {name: [] for name in sides}
for block in range(blocks):
    for name in (("baseline", "fastmace") if block % 2 == 0 else ("fastmace", "baseline")):
        side = sides[name]
        side["before"]()
        before = snap()
        print(f"=== block {block} {name}", file=sys.stderr, flush=True)
        t = time.perf_counter()
        for _ in range(block_steps):
            side["step"]()
        side["flush"]()
        torch.cuda.synchronize()
        times[name].append((time.perf_counter() - t) / block_steps * 1e3)
        if side["sizes"]() != seen[name] or snap() != before:
            events.append({"block": block, "side": name, "new_sizes": sorted(side["sizes"]() - seen[name]),
                           "before": before, "after": snap(), "ms_per_step": times[name][-1]})
            seen[name] = side["sizes"]()
print(json.dumps({"order": order, "builds": builds, "events": events, "ms_per_step": times,
                  "sizes": {name: sorted(side["sizes"]()) for name, side in sides.items()}}))
