"""Kernel time per model part of FastMACE's float32 force call, forward and backward (fp32 profile, 28 October 2026).

The graph engine replays one CUDA graph of compiled (Inductor-fused) kernels, so its kernels cannot be
attributed to model modules. This tool builds the same plan on the eager tier (``mode="eager"``: the same
rewrites and custom kernels, no compilation, no CUDA graphs) and wraps every module of the force field's model
in profiler ranges:

- forward: ``fwd <module path>`` from the forward pre-hook to the forward hook;
- backward: ``bwd <module path>``, opened by a pass-through autograd function on the module's outputs (its
  backward runs when the first output gradient arrives) and closed by one on its inputs (when the last input
  gradient has been produced). MACE takes the force gradient inside the model's forward, so the backward
  ranges nest inside ``fwd <root>`` on the autograd thread.

Every kernel is attributed to the innermost range that encloses the CUDA runtime call that launched it
(same thread). "exclusive" is a range's own kernels, "inclusive" adds its submodules'. The marker functions
are views and launch no kernel; they may change the autograd engine's order of independent branches, so
backward ranges of parallel branches can overlap (reported as such: a kernel goes to the innermost range).

Measurement only, on an idle GPU (hold the local reservation). Eager kernel times are not the compiled
graph's: Inductor fuses the elementwise work, so use the shares per part together with step_breakdown's
compiled categories. Writes parts.csv, parts_tree.csv, kernels.csv and summary.json into --output.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402
import step_breakdown  # noqa: E402


class _Token:
    """One module call's backward range: opened by the first output gradient, closed after the last input one."""

    open_tokens: list = []

    def __init__(self, name, inputs, nvtx):
        self.name, self.remaining, self.nvtx = name, inputs, nvtx
        self.handle = self.nvtx_id = None

    def open(self):
        import torch
        if self.handle is None:
            self.handle = torch.ops.profiler._record_function_enter_new(f"bwd {self.name}", None)
            if self.nvtx:
                self.nvtx_id = torch.cuda.nvtx.range_start(f"bwd {self.name}")
            _Token.open_tokens.append(self)

    def done_one(self):
        self.remaining -= 1
        if self.remaining <= 0:
            self.close()

    def close(self):
        import torch
        if self.handle is not None:
            torch.ops.profiler._record_function_exit._RecordFunction(self.handle)
            if self.nvtx_id is not None:
                torch.cuda.nvtx.range_end(self.nvtx_id)
            self.handle = None
            if self in _Token.open_tokens:
                _Token.open_tokens.remove(self)


def _marker_functions():
    import torch

    class Enter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, token):
            ctx.token = token
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            ctx.token.open()
            return grad, None

    class Exit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, token):
            ctx.token = token
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            ctx.token.open()  # a module whose output gradient never arrived through a marker
            ctx.token.done_one()
            return grad, None

    return Enter, Exit


def _map(value, fn):
    """``value`` with ``fn`` applied to every tensor in nested tuples, lists and dicts."""
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    return value


def _grad_tensors(value):
    import torch
    found = []
    _map(value, lambda t: found.append(t) or t)
    return [t for t in found if isinstance(t, torch.Tensor) and t.requires_grad and t.is_floating_point()]


def attach_ranges(model, nvtx=False, max_depth=6):
    """Hooks on every module of ``model`` up to ``max_depth`` levels; returns the handles."""
    import torch
    Enter, Exit = _marker_functions()
    handles, pending = [], defaultdict(list)

    def pre(name):
        def hook(module, args, kwargs):
            if torch.is_grad_enabled():
                inputs = _grad_tensors((args, kwargs))
                token = _Token(name, len(inputs), nvtx) if inputs else None
                if token is not None:
                    ids = {id(t) for t in inputs}
                    args, kwargs = _map((args, kwargs), lambda t: Exit.apply(t, token) if id(t) in ids else t)
            else:
                token = None
            handle = torch.ops.profiler._record_function_enter_new(f"fwd {name}", None)
            nvtx_id = torch.cuda.nvtx.range_start(f"fwd {name}") if nvtx else None
            pending[id(module)].append((handle, nvtx_id, token))
            return args, kwargs
        return hook

    def post(name):
        def hook(module, args, kwargs, output):
            handle, nvtx_id, token = pending[id(module)].pop()
            torch.ops.profiler._record_function_exit._RecordFunction(handle)
            if nvtx_id is not None:
                torch.cuda.nvtx.range_end(nvtx_id)
            if token is not None and torch.is_grad_enabled() and _grad_tensors(output):
                return _map(output, lambda t: Enter.apply(t, token) if t.requires_grad and t.is_floating_point()
                            else t)
            return output
        return hook

    for name, module in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        label = name or "<root>"
        handles.append(module.register_forward_pre_hook(pre(label), with_kwargs=True))
        handles.append(module.register_forward_hook(post(label), with_kwargs=True))
    return handles


def attribute(trace_path):
    """[(kernel name, us, innermost range or "", direction)] for every GPU kernel/memcpy/memset of the trace."""
    events = json.loads(Path(trace_path).read_text())
    events = events.get("traceEvents", events)
    ranges = defaultdict(list)  # tid -> [(start, end, name)]
    launches = {}  # correlation -> (tid, ts)
    gpu = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "user_annotation" and e["name"].startswith(("fwd ", "bwd ", "force call")):
            ranges[e["tid"]].append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e["name"]))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = (e["tid"], float(e["ts"]))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(e)
    index = {}
    for tid, rs in ranges.items():
        rs.sort()
        index[tid] = ([r[0] for r in rs], rs)
    out = []
    for e in gpu:
        corr = e.get("args", {}).get("correlation")
        where = launches.get(corr)
        label = ""
        if where is not None and where[0] in index:
            starts, rs = index[where[0]]
            ts = where[1]
            i = bisect.bisect_right(starts, ts)
            best = None
            for j in range(i - 1, -1, -1):  # innermost: the latest-starting enclosing range
                s, end, name = rs[j]
                if end >= ts:
                    best = name
                    break
                if ts - s > 5e6:
                    break
            label = best or ""
        out.append((e["name"], float(e.get("dur", 0)), label, e.get("cat")))
    return out


def build_eager_engine(args, base):
    """The minimal planner's plan for ``base`` on the eager tier, with the graph tier's enabled options beside it."""
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import fixed, minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint),
                                   sha256=step_breakdown.sha256(args.checkpoint))
    force_field = ForceField(loaded, base, precision="float32", rewrites="auto")
    integrator = IntegratorConfig(temperature_K=args.temperature_K, timestep_fs=args.timestep_fs)
    workload = Workload(max_temperature_K=args.temperature_K)
    graph_plan = planner.plan(force_field, base, integrator, workload, hardware)
    plan = planner.plan(force_field, base, integrator, workload, hardware, mode="eager")
    engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperature_K)
    return engine, dict(eager=sorted(fixed.enabled_options(plan)), graph=sorted(fixed.enabled_options(graph_plan)),
                        describe=plan.describe(full=True))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--velocity-seed", type=int, default=20261007)
    p.add_argument("--equilibrate-steps", type=int, default=100)
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--nvtx", action="store_true", help="Also NVTX ranges (for an nsys run of this tool)")
    p.add_argument("--allow-busy-gpu", action="store_true", help="CPU checks of the tool only (no timing)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    oeq_runtime = ab_step_time.prepare_fastmace_hybrid()
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from torch.profiler import ProfilerActivity, profile

    if not args.allow_busy_gpu and ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.velocity_seed))
    Stationary(atoms)
    engine, options = build_eager_engine(args, atoms)
    started = time.perf_counter()
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    equilibrate_s = time.perf_counter() - started
    field = engine.force_field
    model = field.model
    positions = engine.state.positions.detach().clone()
    edges = int(field.data["edge_index"].shape[1])
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    # Plain eager kernel time first (no hooks), then the same calls with the ranges.
    with profile(activities=[ProfilerActivity.CUDA]) as plain:
        for _ in range(args.calls):
            field(positions)
        torch.cuda.synchronize()
    plain_us = sum(e.time_range.elapsed_us() for e in plain.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA) / args.calls
    handles = attach_ranges(model, nvtx=args.nvtx, max_depth=args.max_depth)
    field(positions)
    torch.cuda.synchronize()
    trace = args.output / "trace.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.calls):
            with torch.profiler.record_function("force call"):
                field(positions)
            for token in list(_Token.open_tokens):
                token.close()
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    prof.export_chrome_trace(str(trace))
    kernels = attribute(trace)
    exclusive = defaultdict(lambda: [0.0, 0])
    per_kernel = defaultdict(lambda: [0.0, 0])
    for name, us, label, cat in kernels:
        exclusive[label][0] += us
        exclusive[label][1] += 1
        per_kernel[(label, name)][0] += us
        per_kernel[(label, name)][1] += 1
    total_us = sum(v[0] for v in exclusive.values()) / args.calls
    with (args.output / "parts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["range", "direction", "module", "exclusive_us_per_call", "launches_per_call", "share"])
        for label, (us, n) in sorted(exclusive.items(), key=lambda item: -item[1][0]):
            direction, _, module = label.partition(" ")
            writer.writerow([label or "(no range)", direction, module, f"{us / args.calls:.1f}",
                             f"{n / args.calls:g}", f"{us / args.calls / total_us:.4f}"])
    # Inclusive per module path and direction: the module's own ranges plus its submodules'.
    inclusive = defaultdict(lambda: [0.0, 0])
    for label, (us, n) in exclusive.items():
        direction, _, module = label.partition(" ")
        if direction not in ("fwd", "bwd"):
            continue
        parts = [] if module == "<root>" else module.split(".")
        for depth in range(len(parts) + 1):
            key = (direction, ".".join(parts[:depth]) or "<root>")
            inclusive[key][0] += us
            inclusive[key][1] += n
    with (args.output / "parts_tree.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "module", "depth", "inclusive_us_per_call", "launches_per_call", "share"])
        for (direction, module), (us, n) in sorted(inclusive.items(), key=lambda item: (item[0][1], item[0][0])):
            depth = 0 if module == "<root>" else module.count(".") + 1
            writer.writerow([direction, module, depth, f"{us / args.calls:.1f}", f"{n / args.calls:g}",
                             f"{us / args.calls / total_us:.4f}"])
    with (args.output / "kernels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["range", "kernel", "us_per_call", "launches_per_call"])
        for (label, name), (us, n) in sorted(per_kernel.items(), key=lambda item: -item[1][0]):
            writer.writerow([label or "(no range)", name[:200], f"{us / args.calls:.1f}", f"{n / args.calls:g}"])
    properties = torch.cuda.get_device_properties(0)
    summary = dict(
        tool="fp32_parts", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
        fastmace_commit=ab_step_time.fastmace_commit(), packages=step_breakdown.package_versions(),
        gpu=torch.cuda.get_device_name(), compute_capability=f"{properties.major}.{properties.minor}",
        oeq_runtime=oeq_runtime, structure=structure, natoms=len(atoms), padded_or_exact_edges=edges,
        engine_mode=engine.mode, options=options, equilibrate_s=equilibrate_s, calls=args.calls,
        eager_kernel_us_per_call_without_hooks=plain_us, eager_kernel_us_per_call_with_hooks=total_us,
        unattributed_us_per_call=exclusive.get("", [0.0])[0] / args.calls,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("structure", "engine_mode", "eager_kernel_us_per_call_without_hooks",
                                              "eager_kernel_us_per_call_with_hooks", "unattributed_us_per_call")},
                     indent=2))
    return summary


if __name__ == "__main__":
    main()
