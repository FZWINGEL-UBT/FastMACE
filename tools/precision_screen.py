"""Per-stage precision sensitivity of a MACE force call (plan opt-4, OPT4-T08; accuracy only, no kernels).

Question: if one stage of the force call ran at reduced precision and every other stage stayed FP32, would the
forces still pass the force gate (production error <= 2x the float32 calculator's, both against float64)?

One process per model and structure, on one GPU:

1. the gate's copies of the structure (``selfcheck.make_copies`` with a variable cell: 2 rattled, 2 strained
   and rattled);
2. the float64 reference and the float32 yardstick of ``fastmace.reference`` (MACE's calculator, the stock
   record: cuEquivariance, float64 energy accumulation for the yardstick), each built, evaluated and freed
   before the next;
3. the screened path: the same float32 yardstick calculator on MACE's e3nn modules (``backend="e3nn"``), so
   that every GEMM is a torch matmul that TF32 and autocast reach, and the scatter sums are separate calls;
4. for each stage and precision mode, that path with the stage perturbed and nothing else, scored with the
   gate's own ``selfcheck.copy_errors`` and ``selfcheck.verdict`` against the float64 reference, with the
   cuEq float32 yardstick as the ratio's denominator.

Modes (``MODES``; ``dt`` is bf16 or fp16):

- ``out-dt``: the stage's outputs rounded through ``dt`` in the forward; gradients pass unchanged.
- ``grad-dt``: the gradients the stage's backward produces (with respect to its tensor inputs) rounded.
- ``saved-dt``: the tensors autograd saves inside the stage for its backward stored in ``dt``.
- ``tf32-fwd`` / ``tf32-bwd``: TF32 for the stage's GEMMs in the forward / in its backward. ``tf32-bwd`` runs
  the stage as one autograd node that recomputes it in FP32 in the backward and differentiates it with TF32 on
  for that nested call only, so no other node's backward is touched.
- ``autocast-dt``: the stage under ``torch.autocast`` (its GEMMs in ``dt`` in forward and backward), outputs
  cast back to float32.
- ``cast-dt``: the whole stage in ``dt``: a copy of its modules in ``dt``, inputs cast in, outputs cast out;
  forward and backward arithmetic in ``dt``.

A mode that does not change the stage (no GEMM for TF32 or autocast, nothing saved) gives bitwise the screened
path's own errors and is reported as ``unchanged``. The scatter stage wraps ``mace.modules.blocks.scatter_sum``
for the duration of its interaction's forward; ``cast`` runs the sum itself in ``dt``.

Output: ``<output>/<model>-<structure>.json`` with every row and the case's metadata. Accuracy only: eager,
uncompiled, untimed.
"""

from __future__ import annotations

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import argparse
import contextlib
import copy
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DTYPES = ("bf16", "fp16")
MODES = tuple(f"{kind}-{dt}" for kind in ("out", "grad", "saved") for dt in DTYPES) + (
    "tf32-fwd", "tf32-bwd") + tuple(f"{kind}-{dt}" for kind in ("autocast", "cast") for dt in DTYPES)
STRUCTURES_ROOT = "~/fastmace-baseline/1a-T18"
BENCHMARK = Path("Structures", "Benchmark", "R2SCAN")


def torch_dtype(name):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


# --------------------------------------------------------------------------- stages


def stages(model):
    """``{name: (owner, [modules])}``: the stages of a MACE model, per interaction where they repeat.

    ``owner`` is the module whose forward a scatter stage wraps (else None).
    """
    out = {
        "radial-basis": (None, [model.radial_embedding]),
        "spherical-harmonics": (None, [model.spherical_harmonics]),
        "node-embedding": (None, [model.node_embedding]),
    }
    if getattr(model, "pair_repulsion_fn", None) is not None:
        out["zbl"] = (None, [model.pair_repulsion_fn])
    for i, (block, product, readout) in enumerate(zip(model.interactions, model.products, model.readouts)):
        def present(*names, block=block):
            return [getattr(block, n) for n in names if getattr(block, n, None) is not None]

        out[f"i{i}-radial-mlp"] = (None, present("conv_tp_weights", "density_fn"))
        out[f"i{i}-species-embeddings"] = (None, present("source_embedding", "target_embedding"))
        out[f"i{i}-linears"] = (None, present("linear_up", "linear_1", "linear_2", "linear"))
        out[f"i{i}-conv-tp"] = (None, present("conv_tp"))
        out[f"i{i}-scatter"] = (block, [])
        out[f"i{i}-gate"] = (None, present("equivariant_nonlin"))
        out[f"i{i}-skip-residual"] = (None, present("skip_tp", "linear_res"))
        out[f"i{i}-symmetric-contraction"] = (None, [product.symmetric_contractions])
        out[f"i{i}-product-linear"] = (None, [product.linear])
        out[f"i{i}-readout"] = (None, [readout])
    return {name: value for name, value in out.items() if value[0] is not None or value[1]}


def selected(name, wanted):
    """``name`` is one of ``wanted``, or starts with one of them and a dash (``i0``), or is global (``global``)."""
    interaction = name[:1] == "i" and name[1:2].isdigit()
    return any(name == w or name.startswith(w + "-") or (w == "global" and not interaction) for w in wanted)


def _map_tensors(value, fn):
    import torch

    if torch.is_tensor(value):
        return fn(value) if torch.is_floating_point(value) else value
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


def _functions():
    import torch

    class RoundForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dtype):
            return x.to(dtype).to(x.dtype)

        @staticmethod
        def backward(ctx, g):
            return g, None

    class RoundGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dtype):
            ctx.dtype = dtype
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return g.to(ctx.dtype).to(g.dtype), None

    class Tf32Backward(torch.autograd.Function):
        """The stage as one autograd node: forward in FP32 without a graph; backward recomputes the stage in
        FP32 and differentiates it with TF32 on for that nested call only (as reentrant checkpointing does).

        A TF32 flag left on between hooks would also reach every node of other branches that the autograd
        engine runs in between (it runs higher sequence numbers first, so an early stage's window spans the
        whole later backward).
        """

        @staticmethod
        def forward(ctx, run, *tensors):
            ctx.run = run
            ctx.save_for_backward(*tensors)
            with torch.no_grad():
                out = run(*tensors)
            ctx.single = torch.is_tensor(out)
            return out if ctx.single else tuple(out)

        @staticmethod
        def backward(ctx, *grads):
            inputs = [t.detach().requires_grad_(t.requires_grad) for t in ctx.saved_tensors]
            with torch.enable_grad():
                out = ctx.run(*inputs)
            outs = (out,) if ctx.single else tuple(out)
            pairs = [(o, g) for o, g in zip(outs, grads) if torch.is_tensor(o) and o.requires_grad and g is not None]
            wanted = [t for t in inputs if t.requires_grad]
            set_tf32(True)
            try:
                found = torch.autograd.grad([o for o, _ in pairs], wanted, [g for _, g in pairs], allow_unused=True)
            finally:
                set_tf32(False)
            found = iter(found)
            return (None,) + tuple(next(found) if t.requires_grad else None for t in inputs)

    return RoundForward, RoundGradient, Tf32Backward


def set_tf32(on):
    import torch

    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        matmul.fp32_precision = "tf32" if on else "ieee"
    else:
        matmul.allow_tf32 = bool(on)


class Tf32BackwardWrapper:
    """``forward`` of a stage module through ``Tf32Backward``: its tensor arguments become the node's inputs."""

    def __init__(self, module, function):
        self.original, self.function = module.forward, function

    def __call__(self, *args, **kwargs):
        import torch

        positions = [i for i, a in enumerate(args) if torch.is_tensor(a)]

        def run(*tensors):
            full = list(args)
            for i, t in zip(positions, tensors):
                full[i] = t
            return self.original(*full, **kwargs)

        return self.function.apply(run, *(args[i] for i in positions))


class CastWrapper:
    """``forward`` of a stage module replaced by a ``dt`` copy (``cast``) or run under autocast (``autocast``)."""

    def __init__(self, module, dtype, kind):
        import torch

        self.module, self.dtype, self.kind = module, dtype, kind
        self.original = module.forward
        self.copy = copy.deepcopy(module).to(dtype) if kind == "cast" else None
        self.torch = torch

    def __call__(self, *args, **kwargs):
        torch = self.torch
        if self.kind == "cast":
            args = _map_tensors(args, lambda t: t.to(self.dtype))
            kwargs = _map_tensors(kwargs, lambda t: t.to(self.dtype))
            out = self.copy(*args, **kwargs)
        else:
            with torch.autocast("cuda", dtype=self.dtype):
                out = self.original(*args, **kwargs)
        return _map_tensors(out, lambda t: t.to(torch.float32))


@contextlib.contextmanager
def perturbed(model, stage, mode):
    """The model with ``stage`` (a ``stages`` entry) under ``mode``, restored afterwards."""
    import torch
    from mace.modules import blocks

    owner, modules = stage
    kind, _, dt_name = mode.partition("-")
    dtype = torch_dtype(dt_name) if dt_name in DTYPES else None
    RoundForward, RoundGradient, Tf32Backward = _functions()
    handles, restore = [], []

    if owner is not None:  # the scatter sums inside this interaction's forward
        original = blocks.scatter_sum

        def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
            if kind == "out":
                return RoundForward.apply(original(src, index, dim=dim, out=out, dim_size=dim_size), dtype)
            if kind == "grad":
                return original(RoundGradient.apply(src, dtype), index, dim=dim, out=out, dim_size=dim_size)
            if kind == "cast":
                return original(src.to(dtype), index, dim=dim, dim_size=dim_size).to(src.dtype)
            return original(src, index, dim=dim, out=out, dim_size=dim_size)

        def enter(module, args):
            blocks.scatter_sum = scatter

        def leave(module, args, output):
            blocks.scatter_sum = original

        handles += [owner.register_forward_pre_hook(enter), owner.register_forward_hook(leave)]
        restore.append(lambda: setattr(blocks, "scatter_sum", original))
    for module in modules:
        if kind == "out":
            handles.append(module.register_forward_hook(
                lambda m, a, o: _map_tensors(o, lambda t: RoundForward.apply(t, dtype))))
        elif kind == "grad":
            def round_inputs(m, a, kw):
                def fn(t):
                    return RoundGradient.apply(t, dtype) if t.requires_grad else t

                return _map_tensors(a, fn), _map_tensors(kw, fn)

            handles.append(module.register_forward_pre_hook(round_inputs, with_kwargs=True))
        elif kind == "saved":
            def pack(t):
                return t.to(dtype) if t.dtype == torch.float32 else t

            def unpack(t):
                return t.to(torch.float32) if t.dtype == dtype else t

            scopes = []

            def enter_saved(m, a, scopes=scopes):
                scope = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
                scope.__enter__()
                scopes.append(scope)

            def leave_saved(m, a, o, scopes=scopes):
                scopes.pop().__exit__(None, None, None)

            handles += [module.register_forward_pre_hook(enter_saved), module.register_forward_hook(leave_saved)]
        elif mode == "tf32-fwd":
            handles += [module.register_forward_pre_hook(lambda m, a: set_tf32(True)),
                        module.register_forward_hook(lambda m, a, o: set_tf32(False))]
        elif kind in ("tf32", "autocast", "cast"):  # tf32-bwd here; tf32-fwd took the branch above
            wrapper = Tf32BackwardWrapper(module, Tf32Backward) if kind == "tf32" else CastWrapper(module, dtype, kind)
            module.forward = wrapper
            restore.append(lambda module=module, wrapper=wrapper: setattr(module, "forward", wrapper.original))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for undo in restore:
            undo()
        for module in modules:  # an instance forward left by a wrapper would bypass the class forward
            module.__dict__.pop("forward", None)
        set_tf32(False)


# --------------------------------------------------------------------------- evaluation


def errors(reference, candidate, natoms):
    """The gate's ``copy_errors`` plus the RMS force error (eV/A) and the largest stress component (eV/A^3)."""
    from fastmace.planner import selfcheck

    row = selfcheck.copy_errors(reference, candidate, natoms)
    diff = np.asarray(candidate[1], float) - np.asarray(reference[1], float)
    row["force_rms_eV_per_A"] = float(np.sqrt(np.mean(diff ** 2)))
    return row


def score(references, yardstick, values, natoms):
    from fastmace.planner import selfcheck

    rows = [{"production": errors(r, v, natoms), "yardstick": errors(r, y, natoms)}
            for r, y, v in zip(references, yardstick, values)]
    gate = selfcheck.verdict(rows)
    worst = {q: max(row["production"][q] for row in rows) for q in rows[0]["production"]}
    return rows, gate, worst


def evaluate_all(calculator, configs):
    return [calculator.evaluate(atoms) for atoms in configs]


def _free():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_case(args, label, structure, repeat):
    import torch
    from ase.io import read

    from fastmace.config import IntegratorConfig
    from fastmace.model import loading
    from fastmace.planner import selfcheck
    from fastmace.reference import calculators

    set_tf32(False)
    torch.backends.cudnn.allow_tf32 = False
    # e3nn's scripted modules run TorchScript's profiling graph in their first calls and an optimised (possibly
    # fused) graph afterwards, which rounds differently; without the optimisation every call is the same.
    torch._C._set_graph_executor_optimize(False)
    root =Path(os.path.expanduser(args.structures_root))
    atoms = read(root / BENCHMARK / f"{structure}.cif").repeat(repeat)
    copies = selfcheck.make_copies(atoms, IntegratorConfig(ensemble="npt", cell_mode="full"), seed=args.seed)
    configs = [c.atoms for c in copies]
    natoms = len(atoms)
    t0 = time.time()
    timings = {}
    ref = calculators.load(label, atoms, dtype="float64", device="cuda:0")
    references = evaluate_all(ref, configs)
    del ref
    _free()
    timings["float64_s"] = round(time.time() - t0, 1)
    yard = calculators.load(label, atoms, dtype="float32", device="cuda:0")
    yardstick = evaluate_all(yard, configs)
    yard_info = {"model_dtype": yard.dtype, "role": yard.role}
    del yard
    _free()
    # The screened path only: deterministic scatter sums (CUDA index_add uses atomics otherwise), so that a mode
    # that changes nothing gives the unperturbed numbers bit for bit.
    torch.use_deterministic_algorithms(True, warn_only=True)
    loaded = loading.load(label, backend="e3nn", dtype="float32", device="cuda:0")
    version = calculators.records.mace_version()
    record = calculators.records.record_for(version, calculators.records.family_of(loaded.model, loaded.family))
    screened = calculators.ReferenceCalculator(loaded.calculator, record, atoms, loaded=loaded)
    model = screened.model
    base_values = evaluate_all(screened, configs)
    again = evaluate_all(screened, configs)
    repeatable = all(np.array_equal(a[1], b[1]) and a[0] == b[0] for a, b in zip(again, base_values))
    _, base_gate, base_worst = score(references, yardstick, base_values, natoms)
    timings["setup_s"] = round(time.time() - t0, 1)
    stage_map = stages(model)
    if args.stages:
        stage_map = {k: v for k, v in stage_map.items() if selected(k, args.stages)}
    modes = args.modes or MODES
    results = []
    partial = args.output / f"{label}-{structure}-{'x'.join(map(str, repeat))}.partial.json"
    for name, stage in stage_map.items():
        for mode in modes:
            t1 = time.time()
            entry = {"stage": name, "mode": mode,
                     "modules": [type(m).__name__ for m in stage[1]] or ["scatter_sum"]}
            try:
                with perturbed(model, stage, mode):
                    values = evaluate_all(screened, configs)
                rows, gate, worst = score(references, yardstick, values, natoms)
                same = all(np.array_equal(v[1], b[1]) and v[0] == b[0] for v, b in zip(values, base_values))
                entry.update({"unchanged": bool(same), "passed": gate["passed"], "nonfinite": gate["nonfinite"],
                              "ratios": {q: v.get("ratio") for q, v in gate["quantities"].items()},
                              "worst": worst, "rows": rows})
            except Exception as error:  # noqa: BLE001  (a mode that cannot run is a row, not a stop)
                entry.update({"error": f"{type(error).__name__}: {error}"[:500]})
                _free()
            entry["seconds"] = round(time.time() - t1, 2)
            results.append(entry)
            # Each row as it comes: a CUDA error is sticky and ends the process (OOM on large cells).
            partial.write_text(json.dumps(results) + "\n", encoding="utf-8")
            r = entry.get("ratios", {})
            print(f"{label} {structure} {name:28} {mode:12} "
                  + ("ERROR " + entry["error"][:80] if "error" in entry else
                     f"F {r.get('force_eV_per_A') or math.nan:9.3g}  "
                     f"E {r.get('energy_meV_per_atom') or math.nan:9.3g}  "
                     f"S {r.get('stress_eV_per_A3') or math.nan:9.3g}  {'pass' if entry['passed'] else 'FAIL'}"
                     f"{'  unchanged' if entry['unchanged'] else ''}"), flush=True)
    meta = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label,
            "structure": structure, "repeat": list(repeat), "natoms": natoms, "copies": selfcheck.describe_copies(
                copies, "full"), "seed": args.seed, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "mace": version, "record": record.id, "yardstick": yard_info, "screened_backend": "e3nn",
            "timings": timings, "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    baseline = {"passed": base_gate["passed"], "repeatable_bitwise": bool(repeatable),
                "ratios": {q: v.get("ratio") for q, v in base_gate["quantities"].items()}, "worst": base_worst,
                "yardstick_worst": {q: max(errors(r, y, natoms)[q] or 0 for r, y in zip(references, yardstick))
                                    for q in base_worst}}
    print(f"{label} {structure} baseline (e3nn float32, unperturbed): {baseline['ratios']}", flush=True)
    return {"meta": meta, "baseline": baseline, "results": results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="model label (mh1-r2scan, omat0-medium, ...)")
    p.add_argument("--structures", nargs="+", default=["NaCl"])
    p.add_argument("--repeat", nargs=3, type=int, default=(3, 3, 3))
    p.add_argument("--structures-root", default=STRUCTURES_ROOT, help="holds Structures/Benchmark/R2SCAN")
    p.add_argument("--stages", nargs="*", help="restrict to these stages: names, prefixes (i0, i1) or 'global'")
    p.add_argument("--modes", nargs="*", choices=MODES)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for structure in args.structures:
        repeat = tuple(args.repeat)
        if "@" in structure:
            structure, own = structure.split("@")
            repeat = tuple(int(n) for n in own.split("x"))
        result = run_case(args, args.model, structure, repeat)
        path = args.output / f"{args.model}-{structure}-{'x'.join(map(str, repeat))}.json"
        path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}", flush=True)
        _free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
