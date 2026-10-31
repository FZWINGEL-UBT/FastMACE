"""Where the force call's backward spends its time: the compiled model's forward and MACE's gradient step
(``utils.get_outputs``) timed apart with CUDA events, and the gradient step's kernels profiled by name. With the
memory-rescue records, the backward re-runs forward kernels (the recompute); comparing stages separates them.
Default float32 plan, hybrid, the rescue records set with ``--rewrites``.
    python backward_profile.py --checkpoint PATH --structure NbCl5@5x5x24 --tag s5 --rewrites F=V... --output DIR
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structure", default="NbCl5@5x5x24")
    p.add_argument("--tag", required=True)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from mace.modules import utils

    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    loaded = loading.load_pristine("mh1-r2scan", checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    name, base = ab_step_time.read_structure(args.structure, None)
    rewrites = RewriteConfig(**dict(item.split("=", 1) for item in args.rewrites))
    field = ForceField(loaded, base, precision="float32", rewrites=rewrites)
    plan = planner.plan(field, base, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        planner.hardware())
    engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
    engine.prepare()
    engine.step()
    engine.flush()
    torch.cuda.synchronize()
    # The captured graphs' pool would sit beside the eager calls below: release it (the compiled code stays).
    engine.invalidate_graphs("backward_profile: eager calls only")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    ff = engine.force_field
    if not getattr(ff, "split_forces", False):
        raise SystemExit("the force field does not split the gradient step (F2); this probe times that split")
    positions = engine.state.positions.detach().to(torch.float64).clone()

    def events():
        return {key: torch.cuda.Event(enable_timing=True) for key in ("start", "forward_end", "backward_end")}

    marks = events()
    original = utils.get_outputs

    def timed_get_outputs(*a, **kw):
        marks["forward_end"].record()
        out = original(*a, **kw)
        marks["backward_end"].record()
        return out

    utils.get_outputs = timed_get_outputs
    ff(positions)
    torch.cuda.synchronize()
    forward, backward = [], []
    for _ in range(args.calls):
        marks.update(events())
        marks["start"].record()
        ff(positions)
        torch.cuda.synchronize()
        forward.append(marks["start"].elapsed_time(marks["forward_end"]))
        backward.append(marks["forward_end"].elapsed_time(marks["backward_end"]))

    # The gradient step's kernels by name (one call, under the profiler).
    from torch.profiler import ProfilerActivity, profile

    kernels = defaultdict(lambda: [0.0, 0])

    def profiled_get_outputs(*a, **kw):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = original(*a, **kw)
            torch.cuda.synchronize()
        for event in prof.events():
            if event.device_type.name == "CUDA":
                kernels[event.name][0] += event.device_time_total / 1000.0
                kernels[event.name][1] += 1
        return out

    utils.get_outputs = profiled_get_outputs
    ff(positions)
    torch.cuda.synchronize()
    utils.get_outputs = original
    # The whole call's kernels, so that the forward's (whole minus gradient step) can be compared with the recompute.
    whole = defaultdict(lambda: [0.0, 0])
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        ff(positions)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type.name == "CUDA":
            whole[event.name][0] += event.device_time_total / 1000.0
            whole[event.name][1] += 1
    forward_kernels = sorted(([k, round(v[0] - kernels.get(k, [0.0, 0])[0], 3), v[1] - kernels.get(k, [0.0, 0])[1]]
                              for k, v in whole.items()), key=lambda r: -r[1])
    rows = sorted(([k, round(v[0], 3), v[1]] for k, v in kernels.items()), key=lambda r: -r[1])
    record = dict(host=__import__("os").uname().nodename, gpu=torch.cuda.get_device_name(), structure=args.structure,
                  atoms=len(base), tag=args.tag, rewrites=args.rewrites, fastmace_commit=ab_step_time.fastmace_commit(),
                  forward_ms=statistics.median(forward), backward_ms=statistics.median(backward),
                  forward_all=forward, backward_all=backward, backward_kernel_ms=round(sum(r[1] for r in rows), 3),
                  kernels=rows, forward_kernels=forward_kernels,
                  plan_rewrites=[line.strip() for line in plan.describe(full=True).splitlines()
                                 if "recompute_" in line or "gate_epilogue" in line])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"profile-{name}-{len(base)}-{args.tag}.json").write_text(json.dumps(record, indent=1))
    print(f"{name} {len(base)} atoms {args.tag}: forward {record['forward_ms']:.1f} ms, gradient step "
          f"{record['backward_ms']:.1f} ms (kernels {record['backward_kernel_ms']:.1f} ms)", flush=True)
    for k, ms, n in rows[:args.top]:
        print(f"  {ms:9.2f} ms  x{n:4d}  {k[:150]}", flush=True)


if __name__ == "__main__":
    main()
