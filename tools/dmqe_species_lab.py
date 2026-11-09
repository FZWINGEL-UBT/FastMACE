"""Lab: remove the first interaction's species row plan from CUDA-graph replay.

This is an isolated consumer A/B, not a force-call or MD-step measurement. Both
sides use the current species_mm_rows forward and its input-gradient kernel.
Side a recomputes argmax, species_plan and joined_rows. Side b reads their
identical precomputed tensors. Counts, widths and path sizes are explicit;
the default is an OMAT-0-like synthetic case, not checkpoint qualification.
E0 reductions, skip-select maps and contraction-weight clones are not timed.

The immutable snapshot belongs to one model instance, atom layout, device and
dtype. validate() deliberately synchronizes and compares the entire species
input OUTSIDE the timed/captured region. Production adoption needs an event
contract that refreshes the plan before every changed-species evaluation,
including MC trial, acceptance and rejection, and invalidates/re-captures all
graphs referring to old buffers. Matching species counts alone is insufficient.

Run only through the repository GPU reservation/idle-check wrapper, e.g.:
  python tools/dmqe_species_lab.py --output results/species-plan-local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

# isort: split
import torch

from fastmace.kernels.triton import species_gemm as sg


def build_plan(onehot, sizes, counting):
    """The current JOINED_LAYOUT call's species-derived inputs, flattened."""
    kind = onehot.argmax(dim=1)
    order, counts = sg.species_plan(kind, onehot.shape[1], counting=counting)
    _, rows = sg.joined_rows(order, sizes)
    return kind, order, counts, *rows


@dataclass
class FrozenSpeciesPlan:
    """Lab snapshot with explicit host-side stale-state validation."""

    snapshot: torch.Tensor
    tensors: tuple[torch.Tensor, ...]
    sizes: tuple[int, ...]
    counting: bool
    model_token: str

    @classmethod
    def bind(cls, onehot, sizes, counting, model_token):
        sizes = tuple(sizes)
        if onehot.ndim != 2 or not sizes or min(sizes) < 1:
            raise ValueError("expected [atoms, species] and positive path sizes")
        snapshot = onehot.detach().clone()
        tensors = tuple(t.detach().clone() for t in build_plan(onehot, sizes, counting))
        return cls(snapshot, tensors, sizes, bool(counting), model_token)

    def validate(self, onehot, model_token):
        """Call before replay, outside capture; a stale snapshot requires rebind."""
        metadata = (onehot.shape, onehot.dtype, onehot.device)
        expected = (self.snapshot.shape, self.snapshot.dtype, self.snapshot.device)
        if metadata != expected or model_token != self.model_token:
            raise ValueError("stale species plan: model, shape, dtype or device changed; rebind and recapture")
        if not torch.equal(onehot, self.snapshot):
            raise ValueError("stale species plan: species rows changed; rebind and recapture")


def bitwise_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _timed(graph, rounds):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / rounds


def _summary(samples):
    med = {name: statistics.median(values) for name, values in samples.items()}
    return {"median_us": med, "removed_us": med["dynamic"] - med["frozen"],
            "frozen_over_dynamic": med["frozen"] / med["dynamic"], "samples_us": samples}


def _same_outputs(a, b):
    return {"output_bitwise_equal": bitwise_equal(a[0], b[0]),
            "input_gradient_bitwise_equal": bitwise_equal(a[1], b[1]),
            "output_max_abs": float((a[0] - b[0]).abs().max()),
            "input_gradient_max_abs": float((a[1] - b[1]).abs().max())}


def _consumer(x, weights, transposed, grad, sizes, precision, plan):
    kind, order, counts, *rows = plan
    y = sg.species_mm_rows(x, list(sizes), weights, transposed, kind, order, counts, rows, precision)
    gx = sg.species_mm_rows_grad(grad, list(sizes), transposed, kind, order, counts, rows, precision)
    return y, gx


def _invalidation_checks(attrs, frozen, model_token, dynamic_plan, consume):
    records = {}
    original = attrs.clone()
    # Find unlike rows, so the swap preserves counts while changing row maps.
    first = int(attrs.argmax(1)[0])
    unlike = torch.nonzero(attrs.argmax(1) != first).flatten()
    if unlike.numel() == 0:
        return {"skipped": "requires at least two nonempty species"}
    other = int(unlike[0])
    for name in ("same_count_swap", "composition_change"):
        attrs.copy_(original)
        if name == "same_count_swap":
            attrs[[0, other]] = attrs[[other, 0]]
        else:
            attrs[0] = attrs[other]
        try:
            frozen.validate(attrs, model_token)
        except ValueError:
            rejected = True
        else:
            rejected = False
        refreshed = FrozenSpeciesPlan.bind(attrs, frozen.sizes, frozen.counting, model_token)
        refreshed.validate(attrs, model_token)
        actual = consume(refreshed.tensors)
        expected = consume(dynamic_plan())
        records[name] = {"stale_rejected": rejected, **_same_outputs(expected, actual)}
    attrs.copy_(original)
    frozen.validate(attrs, model_token)
    return records


def _run_case(args, attrs, x, weights, transposed, grad, counting):
    sizes = tuple(args.sizes)
    model_token = "synthetic-fixed-weight-instance-0"

    def make():
        return build_plan(attrs, sizes, counting)

    dynamic_plan = torch.compile(make, fullgraph=True) if args.compile_plan else make
    frozen = FrozenSpeciesPlan.bind(attrs, sizes, counting, model_token)
    frozen.validate(attrs, model_token)

    def consume(plan):
        return _consumer(x, weights, transposed, grad, sizes, args.precision, plan)

    functions = {"dynamic": lambda: consume(dynamic_plan()), "frozen": lambda: consume(frozen.tensors)}
    compiled_plan = dynamic_plan()
    plan_equal = all(bitwise_equal(a, b) for a, b in zip(compiled_plan, frozen.tensors))
    initial = _same_outputs(functions["dynamic"](), functions["frozen"]())
    if not plan_equal or not all(initial[key] for key in ("output_bitwise_equal", "input_gradient_bitwise_equal")):
        raise RuntimeError(f"initial exactness failed: plan={plan_equal}, {initial}")
    invalidation = _invalidation_checks(attrs, frozen, model_token, dynamic_plan, consume)
    if any(not all(row[key] for key in ("stale_rejected", "output_bitwise_equal", "input_gradient_bitwise_equal"))
           for row in invalidation.values() if isinstance(row, dict)):
        raise RuntimeError(f"invalidation checks failed: {invalidation}")
    builds = {}
    for order in ("ab", "ba"):
        graphs, outputs = {}, {}
        for name in (("dynamic", "frozen") if order == "ab" else ("frozen", "dynamic")):
            graphs[name], outputs[name] = _capture(functions[name])
        # Verify captured outputs after replay as well, not only eager calls.
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        captured = _same_outputs(outputs["dynamic"], outputs["frozen"])
        if not captured["output_bitwise_equal"] or not captured["input_gradient_bitwise_equal"]:
            raise RuntimeError(f"captured exactness failed: {captured}")
        samples = {name: [] for name in graphs}
        for block in range(args.blocks):
            names = ("dynamic", "frozen") if block % 2 == 0 else ("frozen", "dynamic")
            for name in names:
                samples[name].append(_timed(graphs[name], args.rounds))
        builds[order] = {**_summary(samples), "captured_checks": captured}
        print(f"counting={counting} build={order}: {json.dumps(builds[order]['median_us'])}, "
              f"b/a={builds[order]['frozen_over_dynamic']:.6f}", flush=True)
        del graphs, outputs
    plan_graph, plan_output = _capture(dynamic_plan)
    plan_times = [_timed(plan_graph, args.rounds) for _ in range(args.blocks)]
    return {"counting": counting, "plan_bitwise_equal": plan_equal, "initial_checks": initial,
            "invalidation_checks": invalidation, "build_orders": builds,
            "dynamic_plan_only_us": {"median": statistics.median(plan_times), "samples": plan_times},
            "cached_plan_bytes": sum(t.numel() * t.element_size() for t in frozen.tensors),
            "validation_snapshot_bytes": frozen.snapshot.numel() * frozen.snapshot.element_size()}


def _csv_ints(text):
    return tuple(int(item) for item in text.split(","))


def source_commit(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    marker = root / ".source_commit"
    return marker.read_text().strip() if marker.exists() else "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--counts", type=_csv_ints, default=(1000, 1000, 3000))
    parser.add_argument("--sizes", type=_csv_ints, default=(1, 3, 5, 7))
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--precision", choices=("ieee", "tf32x3", "tf32x3-cutlass"), default="tf32x3")
    parser.add_argument("--plan", choices=("both", "counting", "sort"), default="both")
    parser.add_argument("--compile-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args(argv)
    if min(args.counts) < 0 or sum(args.counts) < 2 or min(args.sizes) < 1:
        parser.error("need nonnegative species counts, at least two atoms and positive path sizes")
    if min(args.input_width, args.channels, args.blocks, args.rounds) < 1:
        parser.error("widths and timing counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    kind = torch.cat([torch.full((n,), s, device=device, dtype=torch.int64) for s, n in enumerate(args.counts)])
    kind = kind[torch.randperm(kind.numel(), device=device, generator=generator)]
    attrs = torch.nn.functional.one_hot(kind, len(args.counts)).float()
    x = torch.randn(kind.numel() * sum(args.sizes), args.input_width, device=device, generator=generator)
    weights = [torch.randn(len(args.counts), args.input_width, args.channels, device=device, generator=generator)
               / args.input_width ** 0.5 for _ in args.sizes]
    transposed = [w.transpose(1, 2).contiguous() for w in weights]
    grad = torch.randn(kind.numel(), sum(args.sizes), args.channels, device=device, generator=generator)
    commit = source_commit()
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": sys.executable,
              "fastmace_commit": commit, "baseline_commit": commit,
              "source_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model": "synthetic OMAT-0-like row-product inputs; no checkpoint",
              "backend": "species_mm_rows forward and input-gradient, CUDA graph replay",
              "precision": args.precision, "counts": args.counts, "sizes": args.sizes,
              "input_width": args.input_width, "channels": args.channels, "compile_plan": args.compile_plan,
              "scope": "isolated plan removal with unchanged consumers; not a force-call or MD-step speedup",
              "excluded": ["E0 and graph sums", "contraction-weight clones", "other skip-select plans"],
              "lifecycle": "validate outside capture; refresh and recapture on species, model, shape, dtype "
                           "or device change",
              "cases": {}}
    modes = (False, True) if args.plan == "both" else (args.plan == "counting",)
    for counting in modes:
        result["cases"]["counting" if counting else "sort"] = _run_case(
            args, attrs, x, weights, transposed, grad, counting)
        (args.output / "species_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
