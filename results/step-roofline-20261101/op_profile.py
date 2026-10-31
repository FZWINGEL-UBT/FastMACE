"""The force call's GPU kernels with the operator that launched each and its input shapes, for a roofline: one eager
call of the default plan's compiled force call (forward and gradient step) under the CUDA profiler with shapes.
Default float32 plan on this GPU (rule 12 picks the memory-rescue records); the captured graphs are released first.
    python op_profile.py --checkpoint PATH --structure NbCl5@5x5x12 --output DIR
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structure", default="NbCl5@5x5x12")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch.profiler import ProfilerActivity, profile

    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    loaded = loading.load_pristine("mh1-r2scan", checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    name, base = ab_step_time.read_structure(args.structure, None)
    field = ForceField(loaded, base, precision="float32", rewrites=RewriteConfig())
    plan = planner.plan(field, base, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        planner.hardware())
    engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
    engine.prepare()
    for _ in range(3):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    step_ms = []
    for _ in range(args.calls):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        engine.step()
        stop.record()
        engine.flush()
        torch.cuda.synchronize()
        step_ms.append(start.elapsed_time(stop))
    engine.invalidate_graphs("op_profile: eager calls only")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    ff = engine.force_field
    positions = engine.state.positions.detach().to(torch.float64).clone()
    ff(positions)
    torch.cuda.synchronize()
    call_ms = []
    for _ in range(args.calls):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        ff(positions)
        stop.record()
        torch.cuda.synchronize()
        call_ms.append(start.elapsed_time(stop))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        ff(positions)
        torch.cuda.synchronize()
    ops = []
    for event in prof.events():
        if event.device_type.name != "CPU" or not event.kernels:
            continue
        ops.append(dict(op=event.name, shapes=[list(s) if isinstance(s, (list, tuple)) else s
                                               for s in (event.input_shapes or [])],
                        kernels=[[k.name, round(k.duration / 1000.0, 4)] for k in event.kernels]))
    record = dict(host=__import__("os").uname().nodename, gpu=torch.cuda.get_device_name(), structure=args.structure,
                  atoms=len(base), fastmace_commit=ab_step_time.fastmace_commit(), edges=int(ff.edge_count),
                  step_ms=statistics.median(step_ms), call_ms=statistics.median(call_ms),
                  plan_rewrites=[line.strip() for line in plan.describe(full=True).splitlines()
                                 if "recompute_" in line or "gate_epilogue" in line],
                  ops=ops)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"ops-{name}-{len(base)}.json"
    out.write_text(json.dumps(record))
    kernel_ms = sum(k[1] for op in ops for k in op["kernels"])
    print(f"{name} {len(base)} atoms: step {record['step_ms']:.1f} ms (graph), eager call {record['call_ms']:.1f} ms, "
          f"profiled kernels {kernel_ms:.1f} ms, {len(ops)} ops -> {out}", flush=True)


if __name__ == "__main__":
    main()
