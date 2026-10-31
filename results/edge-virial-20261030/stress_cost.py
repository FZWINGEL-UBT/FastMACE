"""What the strain path costs in the force call: the bound on what an edge virial can save (edge-virial falsifier).

The edge virial needs the gradient by the edge vectors, which the force backward already computes on its way to the
positions; it removes the strain leaf (strained positions, cell and shifts, and their backward) and adds an
``[E, 3] x [E, 3] -> [3, 3]`` reduction. So the force call without stress is a lower bound on its time and memory,
and (strain - none) the most it can gain. Two engines of the default plan (MinimalPlanner, hybrid, float32), built
and prepared in one process: "strain" (NPT, the production stress path) and "none" (NVT, no stress). Their force
fields' calls are timed in alternating blocks on the same positions and edges, and each call's peak memory above
what was allocated before it is recorded.
    python stress_cost.py --checkpoint PATH --structures NaCl NbCl5@4x4x8 --output DIR
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    record = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, calls=args.calls,
                  blocks=args.blocks, structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        engines = {}
        for side, ensemble in (("strain", "npt"), ("none", "nvt")):
            field = ForceField(loaded, base, precision="float32", rewrites="auto")
            integrator = IntegratorConfig(ensemble=ensemble, temperature_K=300.0)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=300.0), hardware)
            engines[side] = planner.build(plan, field, base.copy(), temperature_K=300.0)
            engines[side].prepare()
        positions = engines["none"].state.positions.detach().to(torch.float64).clone()
        for engine in engines.values():
            engine.flush()
            if engine.force_field.edge_count != engines["none"].force_field.edge_count:
                raise SystemExit(f"{name}: the two engines' edge counts differ")

        def call(side):
            out = engines[side].force_field(positions)
            return out

        for side in engines:  # warm-up: compiled variants exist after prepare; one more call each
            call(side)
        torch.cuda.synchronize()
        peaks = {}
        for side in engines:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            call(side)
            torch.cuda.synchronize()
            peaks[side] = (torch.cuda.max_memory_allocated() - before) / 2**20
        times = {side: [] for side in engines}
        for block in range(args.blocks):
            for side in (("strain", "none") if block % 2 == 0 else ("none", "strain")):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.calls):
                    call(side)
                torch.cuda.synchronize()
                times[side].append((time.perf_counter() - started) / args.calls * 1e3)
        med = {side: statistics.median(v) for side, v in times.items()}
        stress = engines["strain"].force_field(positions).stress
        record["structures"][f"{name}-{len(base)}"] = dict(  # one key per cell: NbCl5 comes in two sizes
            atoms=len(base), edges=int(engines["none"].force_field.edge_count), median_ms=med, block_ms=times,
            ratio_none_over_strain=med["none"] / med["strain"], call_peak_MiB=peaks,
            stress_finite=bool(torch.isfinite(stress).all()))
        print(f"{name} {len(base)} atoms: force call strain {med['strain']:.3f} ms, none {med['none']:.3f} ms, "
              f"none/strain {med['none'] / med['strain']:.4f}; call peak MiB strain {peaks['strain']:.0f}, "
              f"none {peaks['none']:.0f}", flush=True)
        engines.clear()
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stress_cost.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
