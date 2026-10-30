"""Where the force call's memory peak sits, by model segment (plans/capacity.md CAP-T08: the atom-chunk falsifier).

Atom-chunking a node-local segment (every row an atom, nothing across atoms) with recomputation in the backward
lowers the force call's peak only if the peak is reached while that segment's saved tensors are alive, or inside its
own backward. This tool finds out where the peak is. In the eager tier (``compile_mode=none``, so the modules run as
modules) it builds the default plan's ``fm.Simulation`` for one supercell, runs two steps, then makes one force call
(forward and the force backward) of the engine's force field with

- markers at every marked module's forward entry and exit, and at the start (its output's gradient) and end (its
  input's gradient) of its backward; each marker records the allocated bytes and the peak since the previous
  marker, so every interval between two markers has its own peak;
- ``saved_tensors_hooks`` attributing every saved storage to the innermost marked module running when it was saved.

Marked modules: each interaction and its ``skip_tp``, ``linear_up`` and ``gated_rows`` (or ``linear_1``,
``linear_2``), each product block, each readout.

``--chunk MODULE...`` is the prototype (eager only): each named module runs on ``--chunks`` atom chunks without
saving anything but its inputs, and its backward recomputes one chunk at a time (``AtomChunked``). Every tensor
input whose first dimension is the atom count is chunked, the others pass whole; the output must be one per-atom
tensor. A plain call first gives the reference forces, then the measured call runs chunked.

Writes ``<output>/segments-<name>-<atoms><tag>.json`` and prints the timeline with its peaks. Run inside a local GPU
reservation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402

PARTS = ("skip_tp", "linear_up", "gated_rows", "linear_1", "linear_2")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE", help="RewriteConfig overrides")
    p.add_argument("--chunk", nargs="*", default=[], metavar="MODULE", help="e.g. interaction1.gated_rows product1")
    p.add_argument("--chunks", type=int, default=8)
    return p.parse_args(argv)


def forces_of(out):
    return out["forces"] if isinstance(out, dict) else out[1] if isinstance(out, tuple) else out


def atom_chunked(module, natoms, chunks):
    """``module``'s forward on atom chunks, saving only its inputs; the backward recomputes chunk by chunk."""
    import inspect

    import torch

    forward = module.forward
    names = list(inspect.signature(forward).parameters)
    size = -(-natoms // chunks)
    bounds = [(start, min(start + size, natoms)) for start in range(0, natoms, size)]

    def pieces(inputs, start, stop):
        return [x[start:stop] if torch.is_tensor(x) and x.dim() and x.shape[0] == natoms else x for x in inputs]

    class AtomChunked(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            ctx.save_for_backward(*[x if torch.is_tensor(x) else None for x in inputs])
            ctx.others = [None if torch.is_tensor(x) else x for x in inputs]
            with torch.no_grad():
                out = torch.cat([call(pieces(inputs, a, b)) for a, b in bounds])
            assert out.shape[0] == natoms
            return out

        @staticmethod
        def backward(ctx, grad):
            inputs = [x if x is not None else other for x, other in zip(ctx.saved_tensors, ctx.others)]
            wanted = [torch.is_tensor(x) and ctx.needs_input_grad[i] for i, x in enumerate(inputs)]
            grads = [torch.zeros_like(x) if want else None for x, want in zip(inputs, wanted)]
            for a, b in bounds:
                with torch.enable_grad():
                    part = [x.detach().requires_grad_(want) if torch.is_tensor(x) else x
                            for x, want in zip(pieces(inputs, a, b), wanted)]
                    out = call(part)
                    leaves = [x for x, want in zip(part, wanted) if want]
                    got = iter(torch.autograd.grad(out, leaves, grad[a:b]))
                for i, want in enumerate(wanted):
                    if want:
                        g = next(got)
                        if grads[i].shape[0] == natoms:
                            grads[i][a:b] += g
                        else:
                            grads[i] += g
            return tuple(grads)

    def call(values):
        return forward(**dict(zip(names, values)))

    def run(*args, **kwargs):
        bound = inspect.signature(forward).bind(*args, **kwargs)
        bound.apply_defaults()
        return AtomChunked.apply(*(bound.arguments[name] for name in names))

    return run


def tensors(value):
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import importlib.util

    import torch

    import fastmace as fm
    from fastmace.config import EngineConfig, RewriteConfig

    name, cell = capacity_scan.base_cell(args.structure)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.lower().split("x")))
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", activation_profile.TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    rewrites = {k: capacity_scan._value(v) for k, v in (item.split("=", 1) for item in args.rewrites)}
    field = fm.ForceField(fm.load(args.model), atoms, precision="float32",
                          **({"rewrites": RewriteConfig(**rewrites)} if rewrites else {}))
    sim = fm.Simulation(atoms, field, seed=1, engine=EngineConfig(compile_mode="none"),
                        integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                           friction_per_fs=0.002))
    sim.run(steps=2)
    engine = sim._engine
    model = engine.force_field.model
    model = getattr(model, "_orig_mod", model)
    natoms = len(atoms)
    reference = None
    if args.chunk:
        reference = forces_of(engine.force_field(engine.state.positions)).detach().clone()
        for label in args.chunk:
            module = model
            for part in label.replace("interaction", "interactions.").replace("product", "products.").split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            module.forward = atom_chunked(module, natoms, args.chunks)

    marked = []
    for i, interaction in enumerate(model.interactions):
        marked.append((f"interaction{i}", interaction))
        marked += [(f"interaction{i}.{part}", getattr(interaction, part)) for part in PARTS
                   if isinstance(getattr(interaction, part, None), torch.nn.Module)]
        marked.append((f"product{i}", model.products[i]))
        marked.append((f"readout{i}", model.readouts[i]))

    timeline, stack, saved = [], [], {}
    active = {"on": False}

    def mark(label):
        if not active["on"]:
            return
        timeline.append(dict(label=label, allocated=torch.cuda.memory_allocated(),
                             peak_since_previous=torch.cuda.max_memory_allocated()))
        torch.cuda.reset_peak_memory_stats()

    handles = []
    for label, module in marked:
        def pre(module, inputs, label=label):
            stack.append(label)
            mark(f"{label} forward start")

        def post(module, inputs, output, label=label):
            mark(f"{label} forward end")
            stack.pop()
            if not active["on"]:
                return
            outs = [t for t in tensors(output) if t.requires_grad]
            ins = [t for t in tensors(inputs) if t.requires_grad]
            if outs:
                outs[0].register_hook(lambda grad, label=label: mark(f"{label} backward start"))
            for t in ins[:1]:
                t.register_hook(lambda grad, label=label: mark(f"{label} backward end"))

        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

    def pack(tensor):
        if active["on"] and tensor.device.type == "cuda":
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key not in saved:
                saved[key] = dict(bytes=storage.nbytes(), shape=list(tensor.shape),
                                  module=stack[-1] if stack else "outside", per_atom=bool(
                                      tensor.dim() and tensor.shape[0] and tensor.shape[0] % natoms == 0))
        return tensor

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    active["on"] = True
    mark("call start")
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = engine.force_field(engine.state.positions)
    torch.cuda.synchronize()
    mark("call end")
    active["on"] = False
    difference = None if reference is None else float((forces_of(out) - reference).abs().max())
    del out
    for handle in handles:
        handle.remove()

    by_module = {}
    for entry in saved.values():
        row = by_module.setdefault(entry["module"], dict(bytes=0, per_atom_bytes=0))
        row["bytes"] += entry["bytes"]
        if entry["per_atom"]:
            row["per_atom_bytes"] += entry["bytes"]
    peak = max(event["peak_since_previous"] for event in timeline)
    record = dict(tool="segment_peaks", structure=name, natoms=natoms, edges=int(engine.force_field.edge_count),
                  rewrites=rewrites, chunk=args.chunk, chunks=args.chunks if args.chunk else None,
                  force_max_abs_difference=difference, gpu=torch.cuda.get_device_name(), allocated_before=base,
                  peak=peak, peak_over_before=peak - base, timeline=timeline,
                  saved_by_module={key: dict(value, per_atom_kB_per_atom=value["per_atom_bytes"] / natoms / 1e3)
                                   for key, value in by_module.items()},
                  plan=sim.plan.describe())
    args.output.mkdir(parents=True, exist_ok=True)
    tag = "".join(f"-{k}={v}" for k, v in rewrites.items())
    tag += (f"-chunks{args.chunks}-" + "+".join(args.chunk)) if args.chunk else ""
    path = args.output / f"segments-{name}-{natoms}{tag}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    gib = 2**30
    print(f"{name} {natoms} atoms, {record['edges']} edges, eager: before {base / gib:.3f} GiB, peak "
          f"{peak / gib:.3f} GiB ({(peak - base) / gib:.3f} above) ({path})")
    if args.chunk:
        print(f"chunked on {args.chunks} atom chunks: {', '.join(args.chunk)}; max |dF| {difference:.2e} eV/A")
    for event in timeline:
        flag = " <== peak" if event["peak_since_previous"] == peak else ""
        print(f"  {(event['peak_since_previous'] - base) / gib:7.3f} GiB peak before | "
              f"{(event['allocated'] - base) / gib:7.3f} GiB at  {event['label']}{flag}")
    print("saved by module (GiB, of it per atom, kB per atom):")
    for key, value in sorted(by_module.items(), key=lambda item: -item[1]["bytes"]):
        print(f"  {key:28s} {value['bytes'] / gib:7.3f} {value['per_atom_bytes'] / gib:7.3f} "
              f"{value['per_atom_bytes'] / natoms / 1e3:8.1f}")
    sim.close()
    sys.stdout.flush()  # os._exit skips the buffers of a redirected stdout
    os._exit(0)


if __name__ == "__main__":
    main()
