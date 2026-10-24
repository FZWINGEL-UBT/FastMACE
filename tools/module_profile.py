"""Where a force call's kernels come from: compiled graphs, eager code and autograd nodes (plan task OPT4-T02).

tools/step_breakdown.py attributes each kernel to the CPU operator that launched it. That tells what a kernel
does, not which code asked for it. This probe builds the same engine (``step_breakdown.build_engine``, every
plan default), equilibrates, then profiles --calls eager force calls (``force_field(positions)``, the calls
whose kernels the B graph replays) with Python stacks, and attributes every kernel to:

- ``compiled forward <n>`` / ``compiled backward <n>``: an Inductor graph (dynamo's frame name, the
  ``Torch-Compiled Region`` or ``CompiledFunction[Backward]`` that encloses the launch);
- ``eager <file>:<line> <function>``: the innermost Python frame in MACE, graph_longrange, cuEquivariance,
  OpenEquivariance or FastMACE outside a compiled region (code dynamo left out: graph breaks, untraced blocks);
- ``autograd <Node>``: an autograd node of eager code run by the backward engine, with the operator's shapes.

It also writes dynamo's graph-break reasons and frame counters (``torch._dynamo.utils.counters``) after the
build. Measurement only; hold the local reservation (tools/ci/local_gpu.py run). Writes sources.csv (one row
per source and kernel group), source_totals.csv (per source), dynamo.json and summary.json.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402

PACKAGES = ("mace/", "graph_longrange/", "cuequivariance", "openequivariance/", "fastmace/", "e3nn/")
FRAME = re.compile(r"^(?P<file>.*?)\((?P<line>\d+)\): (?P<function>.*)$")


def _short(path):
    for package in PACKAGES:
        at = path.rfind(package)
        if at >= 0:
            return path[at:]
    return None


def attribute(events):
    """{(source, kernel, op, dims): [launches, us]} for every kernel in a chrome trace's events."""
    spans = defaultdict(list)  # tid -> CPU spans (ops, annotations, Python frames)
    launches, kernels = {}, []
    for e in events:
        cat = e.get("cat")
        if cat in ("cpu_op", "user_annotation", "python_function") and "dur" in e:
            spans[e["tid"]].append(e)
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
    index = {}
    for tid, items in spans.items():
        items.sort(key=lambda e: e["ts"])
        index[tid] = ([e["ts"] for e in items], items)

    def enclosing(launch):
        """The spans that enclose ``launch``, innermost first."""
        starts, items = index.get(launch["tid"], ([], []))
        found = []
        end = launch["ts"] + launch.get("dur", 0)
        for e in reversed(items[:bisect.bisect_right(starts, launch["ts"])]):
            if e["ts"] + e["dur"] >= end:
                found.append(e)
            if launch["ts"] - e["ts"] > 5e5:
                break
        found.sort(key=lambda e: e["dur"])
        return found

    table = defaultdict(lambda: [0, 0.0])
    for k in kernels:
        launch = launches.get(k.get("args", {}).get("correlation"))
        chain = enclosing(launch) if launch is not None else []
        ops = [e for e in chain if e["cat"] == "cpu_op"]
        op = ops[0]["name"] if ops else "?"
        dims = json.dumps(ops[0].get("args", {}).get("Input Dims", ""))[:200] if ops else ""
        source = None
        for e in chain:
            name = e["name"]
            if name.startswith("autograd::engine::evaluate_function: "):
                node = name.split(": ", 1)[1]
                source = ("compiled backward" if "CompiledFunctionBackward" in node else f"autograd {node}")
                break
            if name.startswith("Torch-Compiled Region") or name.startswith("CompiledFunction") \
                    or name.startswith("## Call CompiledFxGraph"):
                source = f"compiled forward {name[:60]}"
                break
            if e["cat"] == "python_function":
                match = FRAME.match(name)
                if match and _short(match["file"]) and "torch/" not in match["file"].split("site-packages/")[-1][:6]:
                    source = f"eager {_short(match['file'])}:{match['line']} {match['function']}"
                    break
        row = table[(source or "unattributed", k["name"][:90], op, dims)]
        row[0] += 1
        row[1] += k["dur"]
    return table


def dynamo_counters(torch):
    from torch._dynamo.utils import counters

    return {group: {str(key): value for key, value in values.items()} for group, values in counters.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="polar1-s")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--velocity-seed", type=int, default=20261007)
    p.add_argument("--equilibrate-steps", type=int, default=50)
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    args = p.parse_args(argv)
    args.rewrites = step_breakdown.rewrite_overrides(args.rewrites)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.backend == "hybrid":
        ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from torch.profiler import ProfilerActivity, profile

    step_breakdown.label_triton_kernels()
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    _, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.velocity_seed))
    Stationary(atoms)
    engine, plan = step_breakdown.build_engine(args, atoms)
    start = engine.state.positions.detach().clone()  # its outputs are saved last (an eager call before the
    # engine's first capture keeps an autograd graph alive and the capture fails: stale capture stream)
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    (args.output / "dynamo.json").write_text(json.dumps(dynamo_counters(torch), indent=1, default=str))
    field = engine.force_field
    positions = engine.state.positions.detach().clone()
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        for _ in range(args.calls):
            field(positions)
        torch.cuda.synchronize()
    trace = args.output / "trace.json"
    prof.export_chrome_trace(str(trace))
    out = field(start)
    torch.save({"energy": out.energy.detach().cpu(), "forces": out.forces.detach().cpu()},
               args.output / "start_outputs.pt")  # the starting structure's outputs, to compare variants
    table = attribute(step_breakdown.read_trace(trace))
    totals = defaultdict(lambda: [0, 0.0])
    with (args.output / "sources.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "us_per_call", "launches_per_call", "kernel", "op", "input_dims"])
        for (source, kernel, op, dims), (n, us) in sorted(table.items(), key=lambda item: -item[1][1]):
            writer.writerow([source, f"{us / args.calls:.1f}", n / args.calls, kernel, op, dims])
            totals[source][0] += n / args.calls
            totals[source][1] += us / args.calls
    with (args.output / "source_totals.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "us_per_call", "launches_per_call"])
        for source, (n, us) in sorted(totals.items(), key=lambda item: -item[1][1]):
            writer.writerow([source, f"{us:.1f}", n])
    model = getattr(field.model, "_orig_mod", field.model)
    untraced = getattr(model, "_untraced_blocks", None)
    summary = dict(tool="module_profile", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   model=args.model, structure=args.structure, backend=args.backend, natoms=len(atoms),
                   calls=args.calls, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   rewrites=args.rewrites, options=plan["options"], untraced_blocks=untraced,
                   kernels_per_call=sum(n for n, _ in totals.values()),
                   kernel_us_per_call=sum(us for _, us in totals.values()))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    keys = ("kernels_per_call", "kernel_us_per_call", "untraced_blocks")
    print(json.dumps({key: summary[key] for key in keys}, default=str))


if __name__ == "__main__":
    main()
