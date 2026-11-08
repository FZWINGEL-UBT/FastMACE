"""Kernel-level lab for the POLAR k-space megakernel (docs/design/research/2026-11-07-polar-megakernel.md).

One field-feature call of the charge recursion for both spins (C = 2, J = 4 density and F = 8 feature components, as
POLAR-1), forward and backward (positions and sources; ``--stress`` also the k-vectors and the per-k factors),
captured in a CUDA graph and replayed, in alternating blocks:

- ``mega``: ``kspace_megakernel.kspace_field``, one launch per direction (Triton);
- ``cuda``: the same with ``CUDA = True`` (fastmace.kernels.cuda.kspace, CUDA C++);
- ``fused``: off-polar-alex's FUSED_FOURIER path (``fourier.structure_factor`` and ``fourier.fourier_project`` with
  ``JOINT_BACKWARD``, the per-k glue in eager PyTorch), one call per spin as the model makes it;
- ``dense``: graph_longrange's multichannel sums on dense cosines and sines (SPIN_BATCHED), the cosines included.

Eager glue: Inductor fuses some of it in the model, so ``fused`` and ``dense`` are upper bounds here; the MD-step
A/B decides. Usage: python tools/bench/kspace_megakernel_lab.py [--sizes 243x216,2989x1944] [--stress]
[--forward-only]
"""

import argparse
import json
import math

import torch

from fastmace.kernels.triton import fourier
from fastmace.kernels.triton import kspace_megakernel as mk

C, J, F = 2, 4, 8


def inputs(K, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    k = (1.5 * torch.randn((K, 3), generator=g)).to(dev)
    r = (25.0 * torch.rand((N, 3), generator=g)).to(dev)
    mask = (torch.rand(K, generator=g) > 0.2).float().to(dev)
    x = torch.randn((N, C, J), generator=g).to(dev)
    dbf = torch.randn((K, 1, J, 2), generator=g).to(dev)
    fbf = torch.randn((K, 2, F // 2, 2), generator=g).to(dev)
    vol = torch.full((K,), 1000.0, device=dev)
    kfc = (torch.rand(K, generator=g) + 0.1).to(dev)
    kfp = torch.ones(K, device=dev)
    return k, r, mask, x, dbf, fbf, vol, kfc, kfp


def mega(k, r, mask, x, dbf, fbf, vol, kfc, kfp):
    K = k.shape[0]
    pr = fbf[..., 0].reshape(K, -1) * kfp[:, None]
    pi = fbf[..., 1].reshape(K, -1) * kfp[:, None]
    w = (2 * math.pi) ** 3 / vol * kfc * 14.399645
    return mk.kspace_field(k, r, mask, x, dbf[..., 0].reshape(K, -1), dbf[..., 1].reshape(K, -1), w, pr, pi)


def cuda(*args):
    return mega(*args)  # step_fn holds mk.CUDA for the whole step: the backward runs inside autograd.grad


def fused(k, r, mask, x, dbf, fbf, vol, kfc, kfp):
    from fastmace.longrange.polar import _fused_density, _fused_projection

    outs = []
    for c in range(C):
        density = _fused_density(x[:, c].unsqueeze(-2), k, r, mask, dbf, vol)
        potential = density * kfc[:, None] * 14.399645
        outs.append(_fused_projection(potential, fbf, k, r, mask, kfp).reshape(r.shape[0], -1))
    return torch.stack(outs, dim=1)


def dense(k, r, mask, x, dbf, fbf, vol, kfc, kfp):
    from graph_longrange.features import assemble_fourier_series_batch_channels, project_to_features_batch_channels

    theta = k @ r.t()
    cos, sin = mask[:, None] * torch.cos(theta), mask[:, None] * torch.sin(theta)
    density = assemble_fourier_series_batch_channels(source_feats=x, cosines=cos, sines=sin, density_basis_fs=dbf,
                                                     volume_per_k=vol)
    potential = density * kfc[:, None, None] * 14.399645
    out = project_to_features_batch_channels(potential=potential, feature_basis_fs=fbf, cosines=cos, sines=sin,
                                             k_factor_proj=kfp)
    return out.reshape(r.shape[0], C, -1)


def step_fn(fn, args, stress, forward_only=False):
    want = [1, 3] + ([0, 4, 5, 6, 7, 8] if stress else [])
    leaves = [t.clone().requires_grad_(i in want) for i, t in enumerate(args)]
    weight = torch.randn((args[1].shape[0], C, F), device="cuda")

    def run():
        mk.CUDA = fn is cuda
        try:
            out = fn(*leaves)
            if forward_only:
                return out
            return torch.autograd.grad((weight * out).sum(), [leaves[i] for i in want])
        finally:
            mk.CUDA = False

    return run


def capture(run):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def time_graph(graph, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps  # µs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="243x216,2989x1944")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--forward-only", action="store_true", help="the forward alone (no autograd graph kept)")
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--reps", type=int, default=200)
    a = p.parse_args()
    fourier.JOINT_BACKWARD = True
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "stress": a.stress,
                      "forward_only": a.forward_only}))
    for size in a.sizes.split(","):
        K, N = map(int, size.split("x"))
        args = inputs(K, N)
        graphs = {name: capture(step_fn(fn, args, a.stress, a.forward_only)) for name, fn in
                  (("mega", mega), ("cuda", cuda), ("fused", fused), ("dense", dense))}
        times = {name: [] for name in graphs}
        for block in range(a.blocks):
            order = list(graphs) if block % 2 == 0 else list(reversed(graphs))
            for name in order:
                times[name].append(time_graph(graphs[name], a.reps))
        med = {name: sorted(v)[len(v) // 2] for name, v in times.items()}
        print(json.dumps({"K": K, "N": N, "us_median": {n: round(v, 2) for n, v in med.items()},
                          "mega_over_fused": round(med["mega"] / med["fused"], 3),
                          "mega_over_dense": round(med["mega"] / med["dense"], 3),
                          "cuda_over_dense": round(med["cuda"] / med["dense"], 3),
                          "cuda_over_mega": round(med["cuda"] / med["mega"], 3)}))


if __name__ == "__main__":
    main()
