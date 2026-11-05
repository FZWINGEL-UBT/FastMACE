"""Edge spherical harmonics lab (round 5, 5 November 2026): ``fastmace.kernels.triton.edge_sh.edge_sh3`` against
e3nn's ``SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component")`` compiled by Inductor, forward plus the
vectors' gradient, on Benchmark_loup's edge count; accuracy of both against e3nn in float64.

    python tools/edge_sh_lab.py [--edges 373760] [--rounds 51]
"""

from __future__ import annotations

import argparse
import json
import statistics


def main():
    import torch

    torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4 loads its own Wigner constants with torch.load
    from e3nn import o3

    from fastmace.kernels.triton.edge_sh import edge_sh3

    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=int, default=373760)
    p.add_argument("--rounds", type=int, default=51)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    e = args.edges
    v64 = torch.randn(e, 3, dtype=torch.float64, device=dev)
    v64 = v64 / v64.norm(dim=1, keepdim=True) * (1.0 + 5.3 * torch.rand(e, 1, dtype=torch.float64, device=dev))
    v64[:64] = 0.0  # the neighbour list's padding rows
    g64 = torch.randn(e, 16, dtype=torch.float64, device=dev)
    sh = o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component").to(dev)

    def reference(v, g):
        v = v.detach().requires_grad_(True)
        out = sh(v)
        (gv,) = torch.autograd.grad(out, v, g)
        return out.detach(), gv

    ref_out, ref_grad = reference(v64, g64)
    v32, g32 = v64.float(), g64.float()
    compiled = torch.compile(lambda v: sh(v))

    def e3nn_compiled(v, g):
        v = v.detach().requires_grad_(True)
        out = compiled(v)
        (gv,) = torch.autograd.grad(out, v, g)
        return out, gv

    def fused(v, g):
        v = v.detach().requires_grad_(True)
        out = edge_sh3(v)
        (gv,) = torch.autograd.grad(out, v, g)
        return out, gv

    result = {"gpu": torch.cuda.get_device_name(), "edges": e}
    for name, fn in (("e3nn-compiled", e3nn_compiled), ("edge_sh3", fused)):
        out, gv = fn(v32, g32)
        err_out = float((out.double() - ref_out).abs().max() / ref_out.abs().max())
        err_grad = float((gv.double() - ref_grad).abs().max() / ref_grad.abs().max())
        for _ in range(5):
            fn(v32, g32)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.rounds):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn(v32, g32)
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1000.0)
        result[name] = dict(us=statistics.median(times), err_out=err_out, err_grad=err_grad)
        print(f"{name:14s} {statistics.median(times):8.1f} us  out {err_out:.2e}  grad {err_grad:.2e}", flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
