"""Group-ordered wprime_backward falsifier (plans/capacity-ideas-20261030.md, ninth list, item 1). No library change.

``oeq_projected_backward_wprime`` is edge-owned: one 32-lane group per edge ``e``, per-edge arrays (``sh``,
``weights``, ``lengths``, ``pairs``, ``dsh[e]``, ``dr[e]``), sender gradients added atomically, no ``row_ptr``.
So the edges may run in any order. In the default (receiver) order, neighbouring edges use different ``G'``
groups (species pair, table piece), each ``TERMS x Q`` floats. In group order a block reuses one group's table.

Real operands: the default float32 plan (MinimalPlanner, hybrid), NVT 1 fs at 1000 K, equilibrated, then one eager
call of the engine's force field. Its ``WPrimeBackward`` launches are recorded (every interaction). Each launch is
replayed in a CUDA graph, 15 alternating blocks:
  (a) as recorded;
  (b) per-edge arrays permuted by the group key ``g(e)`` (``wprime_arguments``; unsloped edges last), outputs
      inverse-permuted: dsh and dr must be bitwise equal to (a), dx equal up to the atomic order.
The permutation's gathers are not timed: in a built version the edges would be stored in group order. Also timed:
``torch.argsort`` of the group key at this E (stable and not).
    python wprime_group_probe.py --checkpoint PATH --structures NbCl5 NaNbO3 --output DIR
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


def timed_graph(torch, launch, reps):
    """A CUDA graph of ``reps`` launches (each zeroing dx first, as the op does)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            launch()
    torch.cuda.synchronize()
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NbCl5", "NaNbO3"])
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--blocks", type=int, default=15)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.kernels.oeq_streaming import wprime_backward as wb
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint),
                                   sha256=ab_step_time._sha256(args.checkpoint))
    record = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, blocks=args.blocks,
                  reps=args.reps, temperature_K=args.temperature_K, structures={})
    calls = []
    original = wb.WPrimeBackward.__call__

    def recording(self, *a):
        calls.append((self, [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in a]))
        return original(self, *a)

    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        field = ForceField(loaded, base, precision="float32", rewrites="auto")
        integrator = IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=1.0)
        plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
        engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
        engine.prepare()
        for _ in range(args.equilibrate_steps):
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
        positions = engine.state.positions.detach().to(torch.float64).clone()
        calls.clear()
        wb.WPrimeBackward.__call__ = recording
        try:
            engine.force_field(positions)
            torch.cuda.synchronize()
        finally:
            wb.WPrimeBackward.__call__ = original
        if not calls:
            raise SystemExit(f"{name}: no wprime_backward launch (plan: {plan.describe()[:400]})")
        rows = []
        for index, (kernel, a) in enumerate(calls):
            (x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins, r_lo,
             bin_scale, r_max, dr) = a
            edges = sh.shape[0]
            _, group = wb.wprime_arguments(r, pair, starts, scales, bins, r_lo, bin_scale, r_max)
            key = torch.where(group >= 0, group, torch.full_like(group, int(group.max()) + 1))
            perm = torch.argsort(key, stable=True)
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(edges, device=perm.device)
            permuted = dict(sh=sh[perm].contiguous(), weights=weights[perm].contiguous(),
                            senders=senders[perm].contiguous(), receivers=receivers[perm].contiguous(),
                            r=r[perm].contiguous(), pair=pair[perm].contiguous())
            outs = {side: (torch.empty_like(dx), torch.empty_like(dsh), torch.empty_like(dr)) for side in "ab"}

            def launch_a():
                o = outs["a"]
                o[0].zero_()
                kernel(x, sh, weights, senders, receivers, dout, o[0], o[1], r, pair, gprime, starts, scales,
                       bins, r_lo, bin_scale, r_max, o[2])

            def launch_b():
                o, q = outs["b"], permuted
                o[0].zero_()
                kernel(x, q["sh"], q["weights"], q["senders"], q["receivers"], dout, o[0], o[1], q["r"], q["pair"],
                       gprime, starts, scales, bins, r_lo, bin_scale, r_max, o[2])

            launch_a()
            launch_b()
            torch.cuda.synchronize()
            dsh_equal = torch.equal(outs["a"][1], outs["b"][1][inverse])
            dr_equal = torch.equal(outs["a"][2], outs["b"][2][inverse])
            dx_diff = float((outs["a"][0] - outs["b"][0]).abs().max())
            dx_scale = float(outs["a"][0].abs().max())
            graphs = {"a": timed_graph(torch, launch_a, args.reps), "b": timed_graph(torch, launch_b, args.reps)}
            times = {"a": [], "b": []}
            for block in range(args.blocks):
                for side in ("ab" if block % 2 == 0 else "ba"):
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[side].replay()
                    stop.record()
                    torch.cuda.synchronize()
                    times[side].append(start.elapsed_time(stop) / args.reps * 1e3)
            med = {side: statistics.median(v) for side, v in times.items()}
            sorts = {}
            for stable in (True, False):
                values = []
                for _ in range(args.blocks):
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.reps):
                        torch.argsort(key, stable=stable)
                    stop.record()
                    torch.cuda.synchronize()
                    values.append(start.elapsed_time(stop) / args.reps * 1e3)
                sorts["stable" if stable else "unstable"] = statistics.median(values)
            groups = int((group >= 0).sum() and torch.unique(group[group >= 0]).numel())
            runs = int((key[1:] != key[:-1]).sum()) + 1
            row = dict(launch=index, edges=edges, nodes=int(x.shape[0]), weight_dim=int(weights.shape[1]),
                       terms=int(gprime.shape[1]), groups_used=groups, group_changes_receiver_order=runs,
                       sloped_edges=int((group >= 0).sum()), median_us=med, block_us=times,
                       ratio_b_over_a=med["b"] / med["a"], argsort_us=sorts, dsh_bitwise=bool(dsh_equal),
                       dr_bitwise=bool(dr_equal), dx_max_abs_diff=dx_diff, dx_max_abs=dx_scale)
            rows.append(row)
            print(f"{name} launch {index}: E={edges} Q={row['weight_dim']} groups={groups} "
                  f"(receiver order changes group {runs} times); a {med['a']:.1f} us, b {med['b']:.1f} us, "
                  f"b/a {row['ratio_b_over_a']:.3f}; argsort {sorts['stable']:.1f} us stable, "
                  f"{sorts['unstable']:.1f} us; dsh bitwise {dsh_equal}, dr bitwise {dr_equal}, "
                  f"dx |d| {dx_diff:.2e} of {dx_scale:.2e}", flush=True)
        total = {side: sum(row["median_us"][side] for row in rows) for side in "ab"}
        record["structures"][f"{name}-{len(base)}"] = dict(atoms=len(base), launches=rows, total_us=total,
                                                         ratio_b_over_a=total["b"] / total["a"])
        print(f"{name} {len(base)} atoms: all launches a {total['a']:.1f} us, b {total['b']:.1f} us, "
              f"b/a {total['b'] / total['a']:.3f}", flush=True)
        del engine, field
        calls.clear()
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "wprime_group_probe.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
