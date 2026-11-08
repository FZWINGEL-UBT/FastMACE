"""Lab for composing interaction 1's reassociated projection with the receiver-species skip (H200 ideas after
SPLINE_W, idea 1): times, at OMAT-0 medium's shapes on SrTiO3 10x10x10, the production chain against the composed one.

Production (H200): per degree l the reassociation's product ``left_l [n (2l+1), 192] @ right_l [192, 128]``
(``tf32x3_matmul``; the linear is folded into ``right_l``), the blocks joined and divided by ``density + 1``, then
``tf32x3_skip_select`` (``[n (2l+1), 128] @ B_l [128, species 128]``, each atom's own-species columns kept).

Composed: ``C_l,z = right_l B_l,z`` ([species, 192, 128], built once per model), then ``species_mm(left_l, C_l)`` per
l (the atoms grouped by species inside the kernel), the blocks joined and divided. One product per l instead of two,
and the skip's three-species product (3x the columns) is gone.

Variants: ``old``; ``composed-<launch>`` for species_mm's launches (``triton``, ``tc``: mma.sync, ``cutlass``);
``sorted``: the atoms already sorted by species (three plain products per l on contiguous rows; the floor a
species-sorted layout would reach). Each is captured with its backward (the gradient of ``left``, as forces need) in
one CUDA graph and replayed; per variant the median of ``--repeats`` timings of ``--replays`` replays. Each variant's
output and gradient are compared with a float64 reference (max |error| / max |reference|).

Output: JSON on stdout and in ``--output``."""
from __future__ import annotations

import argparse
import json
import statistics
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=5000)
    p.add_argument("--pattern", default="0,1,2,2,2", help="species of one unit cell, repeated (SrTiO3: Sr Ti O O O)")
    p.add_argument("--degrees", default="0,1,2,3")
    p.add_argument("--k", type=int, default=192, help="slots x radial width (3 x 64)")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", default="old,composed-triton,composed-tc,composed-cutlass,sorted")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def build(args, device):
    import torch

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    pattern = [int(s) for s in args.pattern.split(",")]
    species = max(pattern) + 1
    n, k, c = args.atoms, args.k, args.channels
    kind = torch.tensor(pattern * (n // len(pattern)) + pattern[:n % len(pattern)], dtype=torch.int64)
    degrees = [int(d) for d in args.degrees.split(",")]
    sizes = [2 * d + 1 for d in degrees]
    lefts = [torch.randn(n * s, k, generator=g) for s in sizes]
    rights = [torch.randn(k, c, generator=g) / k ** 0.5 for _ in sizes]
    skips = [torch.randn(c, species * c, generator=g) / c ** 0.5 for _ in sizes]  # columns species-major
    density = torch.rand(n, generator=g) * 20
    grad = torch.randn(n, sum(sizes) * c, generator=g)
    onehot = torch.nn.functional.one_hot(kind, species).to(torch.float32)
    # C_l,z = right_l B_l,z in float64, stored as float32 (what a build would keep)
    composed = [torch.stack([(r.double() @ b.double()[:, z * c:(z + 1) * c]) for z in range(species)]).float()
                for r, b in zip(rights, skips)]
    case = dict(n=n, c=c, species=species, sizes=sizes, kind=kind, lefts=lefts, rights=rights, skips=skips,
                density=density, grad=grad, onehot=onehot, composed=composed)
    def move(value):
        if isinstance(value, list):
            return [move(v) for v in value]
        return value.to(device) if torch.is_tensor(value) else value

    return {key: move(value) for key, value in case.items()}


def reference(case):
    """float64 output [n, width] and gradient of each left, from the unfused products."""
    import torch

    n, c, kind = case["n"], case["c"], case["kind"]
    lefts = [left.double().requires_grad_() for left in case["lefts"]]
    blocks = []
    for left, right, skip, size in zip(lefts, case["rights"], case["skips"], case["sizes"]):
        y = (left @ right.double()).view(n, size, c) / (case["density"].double() + 1)[:, None, None]
        full = (y.reshape(n * size, c) @ skip.double()).view(n, size, case["species"], c)
        blocks.append(full[torch.arange(n, device=kind.device), :, kind].reshape(n, size * c))
    out = torch.cat(blocks, dim=1)
    (out * case["grad"].double()).sum().backward()
    return out.detach(), [left.grad for left in lefts]


def old_chain(case):
    import torch

    from fastmace.kernels.cutlass import skip_select
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

    n, c = case["n"], case["c"]
    kind32 = case["kind"].to(torch.int32)

    def run(lefts):
        blocks = [tf32x3_matmul(left, right).view(n, -1) for left, right in zip(lefts, case["rights"])]
        joined = torch.cat(blocks, dim=1) / (case["density"] + 1)[:, None]
        offsets, views, start = [], [], 0
        for size in case["sizes"]:
            views.append(joined[:, start:start + size * c].reshape(n * size, c))
            offsets.append(start)
            start += size * c
        if not skip_select.applies(views, case["skips"], case["onehot"], offsets, start):
            raise RuntimeError("tf32x3_skip_select does not take these operands")
        return skip_select.tf32x3_skip_select(views, case["skips"], case["onehot"], kind32, offsets, start)

    return run


def composed_chain(case, launch):
    import torch

    from fastmace.kernels.triton import species_gemm

    species_gemm.CUTLASS = launch == "cutlass"
    species_gemm.TENSOR_CORES = launch == "tc"
    n = case["n"]
    order, counts = species_gemm.species_plan(case["kind"], case["species"])
    transposed = [w.transpose(1, 2).contiguous() for w in case["composed"]]

    def run(lefts):
        blocks = [species_gemm.species_mm(left.view(n, size, -1), w, wt, case["kind"], order, counts, "tf32x3")
                  .reshape(n, -1) for left, w, wt, size in zip(lefts, case["composed"], transposed, case["sizes"])]
        return torch.cat(blocks, dim=1) / (case["density"] + 1)[:, None]

    return run


def sorted_chain(case):
    """The atoms sorted by species already: per l and species one product on contiguous rows. Its input is the
    sorted left (built outside the timing) and its output stays sorted: the floor, not a drop-in."""
    import torch

    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

    n, species = case["n"], case["species"]
    counts = torch.bincount(case["kind"], minlength=species).tolist()
    order = torch.argsort(case["kind"], stable=True)
    case["sorted_order"] = order
    density = (case["density"][order] + 1)[:, None]

    def run(lefts):
        blocks = []
        for left, w, size in zip(lefts, case["composed"], case["sizes"]):
            parts, start = [], 0
            for z, count in enumerate(counts):
                rows = left[start * size:(start + count) * size]
                parts.append(tf32x3_matmul(rows, w[z]).view(count, -1))
                start += count
            blocks.append(torch.cat(parts, dim=0))
        return torch.cat(blocks, dim=1) / density

    return run


def sort_rows(case, tensors):
    order = case["sorted_order"]
    return [t.view(case["n"], -1, t.shape[-1])[order].reshape(t.shape) for t in tensors]


def measure(run, lefts, grad, args):
    """(median ms of one forward + backward, output, gradients): captured in one CUDA graph with its backward."""
    import torch

    inputs = [left.clone().requires_grad_() for left in lefts]

    def step():
        for x in inputs:
            x.grad = None
        out = run(inputs)
        out.backward(grad)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    for x in inputs:
        x.grad = None
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / args.replays)
    return statistics.median(times), times, out.detach().clone(), [x.grad.detach().clone() for x in inputs]


def relative(a, b):
    return float((a.double() - b).abs().max() / b.abs().max())


def main(argv=None):
    import torch

    import fastmace  # noqa: F401  (before torch use: soft file locks)

    args = parse_args(argv)
    device = torch.device("cuda")
    case = build(args, device)
    ref_out, ref_grads = reference(case)
    record = dict(gpu=torch.cuda.get_device_name(0), atoms=args.atoms, pattern=args.pattern, sizes=case["sizes"],
                  k=args.k, channels=args.channels, replays=args.replays, repeats=args.repeats, variants={})
    for name in args.variants.split(","):
        try:
            if name == "old":
                run, lefts, grad = old_chain(case), case["lefts"], case["grad"]
            elif name.startswith("composed-"):
                run, lefts, grad = composed_chain(case, name.split("-", 1)[1]), case["lefts"], case["grad"]
            elif name == "sorted":
                run = sorted_chain(case)
                lefts, grad = sort_rows(case, case["lefts"]), case["grad"][case["sorted_order"]]
            else:
                raise ValueError(f"unknown variant {name}")
            ms, times, out, grads = measure(run, lefts, grad, args)
            if name == "sorted":  # back to the atoms' order for the comparison
                inverse = torch.argsort(case["sorted_order"])
                out = out[inverse]
                grads = [g.view(case["n"], -1, g.shape[-1])[inverse].reshape(g.shape) for g in grads]
            row = dict(ms=ms, times_ms=times, out_rel_err=relative(out, ref_out),
                       grad_rel_err=max(relative(g, r) for g, r in zip(grads, ref_grads)))
        except Exception as error:  # noqa: BLE001  (a lab: record and go on)
            row = dict(error=f"{type(error).__name__}: {error}")
        record["variants"][name] = row
        print(name, json.dumps(row), flush=True)
    base = record["variants"].get("old", {}).get("ms")
    if base:
        for row in record["variants"].values():
            if "ms" in row:
                row["saving_us"] = (base - row["ms"]) * 1e3
    text = json.dumps(record, indent=1)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
