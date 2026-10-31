"""Atom-chunked combined region (ninth list item 2): the compiled force call's allocator peak and time.

The default float32 plan (MinimalPlanner, hybrid) with ``recompute_conv_weights`` and ``recompute_node_segments`` on,
one region layout per process (``--region separate`` or ``combined``, ``--chunks`` for the combined region's atom
chunks). After the build, its first-step captures and two steps, the engine's force field is called eagerly (the
compiled model, forward and force backward): the peak is the caching allocator's peak above what was allocated
before the call, as in docs/validation/capacity-T08-chunks.md; the time is the median of ``--calls`` calls.
    python region_probe.py --checkpoint PATH --structure NbCl5@5x5x12 --region combined --chunks 4 --output DIR
"""

import argparse
import json
import os
import statistics
import sys
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
    p.add_argument("--structure", default="NbCl5@5x5x12")
    p.add_argument("--region", choices=("separate", "combined"), required=True)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload
    from fastmace.rewrites import segments

    segments.SEGMENT_REGION, segments.REGION_CHUNKS = args.region, args.chunks
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    name, base = ab_step_time.read_structure(args.structure, None)
    rewrites = RewriteConfig(recompute_conv_weights="on", recompute_node_segments="on")
    field = ForceField(loaded, base, precision="float32", rewrites=rewrites)
    integrator = IntegratorConfig(temperature_K=300.0)
    plan = planner.plan(field, base, integrator, Workload(max_temperature_K=300.0), hardware)
    engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
    engine.prepare()
    for _ in range(2):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    positions = engine.state.positions.detach().to(torch.float64).clone()
    engine.force_field(positions)  # warm (compiled variants exist after prepare)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    engine.force_field(positions)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    times = []
    for _ in range(args.calls):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        engine.force_field(positions)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    record = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), structure=args.structure, atoms=len(base),
                  region=args.region, chunks=args.chunks, peak_above_start_GiB=peak / 2**30,
                  allocated_before_GiB=before / 2**30, reserved_GiB=torch.cuda.memory_reserved() / 2**30,
                  call_ms=statistics.median(times), call_ms_all=times,
                  plan_rewrites=[line.strip() for line in plan.describe(full=True).splitlines()
                                 if "recompute_" in line])
    tag = f"{args.region}-{args.chunks}"
    print(f"{name} {len(base)} atoms {tag}: peak above start {record['peak_above_start_GiB']:.3f} GiB, "
          f"call {record['call_ms']:.1f} ms (before {record['allocated_before_GiB']:.2f} GiB)", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"probe-{tag}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
