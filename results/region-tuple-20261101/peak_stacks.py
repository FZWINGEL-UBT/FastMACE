"""The blocks alive at the compiled force call's peak, with their Python stacks: which module allocated each of the
largest (the combined region's ledger named only the innermost frame). Same setup as region_probe.py (default
float32 plan, hybrid, conv_weights and node_segments on, the combined region in 4 chunks).
    python peak_stacks.py --checkpoint PATH --structure NbCl5@5x5x12 [--top 12]
"""

import argparse
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "bench"))
import ab_step_time  # noqa: E402
import activation_profile  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--structure", default="NbCl5@5x5x12")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--chunks", type=int, default=4, help="the combined region's atom chunks (segments.REGION_CHUNKS)")
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload
    from fastmace.rewrites import segments

    segments.REGION_CHUNKS = args.chunks
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    loaded = loading.load_pristine("mh1-r2scan", checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    _, base = ab_step_time.read_structure(args.structure, None)
    rewrites = RewriteConfig(recompute_conv_weights="on", recompute_node_segments="on", recompute_node_region="on")
    field = ForceField(loaded, base, precision="float32", rewrites=rewrites)
    plan = planner.plan(field, base, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        planner.hardware())
    engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
    engine.prepare()
    engine.step()
    engine.flush()
    positions = engine.state.positions.detach().to(torch.float64).clone()
    engine.force_field(positions)
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    engine.force_field(positions)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    trace = [event for device in snapshot["device_traces"] for event in device]
    peak, live = activation_profile.live_at_peak(trace)
    natoms = len(base)
    print(f"{args.structure} {natoms} atoms, {args.chunks} chunks: {peak / 2**30:.3f} GiB live at the recorded "
          f"call's peak ({peak / natoms / 1e3:.1f} kB/atom)")
    for size, frames in sorted(live.values(), key=lambda item: -item[0])[:args.top]:
        print(f"\n{size / natoms / 1e3:6.1f} kB/atom  {size / 2**20:8.1f} MiB")
        for frame in frames[:40]:
            name = frame["filename"]
            if any(skip in name for skip in ("torch/_dynamo", "torch/fx", "torch/nn/modules/module.py",
                                             "torch/utils/_contextlib", "torch/_functorch", "torch/autograd")):
                continue
            print(f"    {name.split('site-packages/')[-1].split('fastmace-capacity/')[-1]}:{frame['line']} "
                  f"{frame['name']}")


if __name__ == "__main__":
    main()
