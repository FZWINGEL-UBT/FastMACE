"""wprime_backward (opt-9 OPT9-T03) on CUDA against the chain it replaces.

The chain: the radial table gives ``a`` and the edge density, ``folded_streaming_convolution`` builds ``w = a
[W, b]^T`` and the message, and its chunked backward gives ``dx``, ``dsh`` and ``da``, which the table's backward
turns into ``dL/dr`` together with the density's gradient. The record must give the same message and density
(the same forward), ``dsh`` as the vendored backward's (the same sums), ``dx`` to the atomics' order, and
``dL/dr`` to float32 rounding; both are compared with a float64 reference built from the same weight gradient.
Edges at and beyond the cutoff and padding edges past every row get a zero radial slope. Both layouts, the
cuBLAS and the fused forward, a CUDA graph. Runs in a fresh interpreter like test_fused_folded_forward_cuda.py.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastmace.kernels import streaming_available

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(not streaming_available(), reason=streaming_available().reason)

CHILD = r"""
import fastmace  # before torch: soft file locks (SPEC §14)
import json, os, sys
from types import SimpleNamespace

if "openequivariance" not in sys.modules and not os.environ.get("FASTMACE_OEQ_PREPARED"):
    from fastmace._cache import cache_dir
    from fastmace.kernels import _oeq_runtime
    _oeq_runtime.prepare_in_process(cache_dir("oeq"))

import numpy as np
import openequivariance as oeq
import torch

from fastmace.kernels.oeq_streaming import StreamingKernels, folded_streaming_convolution
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward
from fastmace.kernels.triton.radial_table import radial_table, radial_table_backward

torch.backends.cuda.matmul.allow_tf32 = False
C = 64
irreps_in1, irreps_in2 = oeq.Irreps(f"{C}x0e+{C}x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o")
irreps_out = oeq.Irreps(f"{C}x0e+{C}x0e+{C}x1o+{C}x1o+{C}x2e+{C}x2e+{C}x3o")
instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True), (0, 2, 4, "uvu", True), (1, 1, 5, "uvu", True), (0, 3, 6, "uvu", True)]
problem = oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False,
                        internal_weights=False)
g = torch.Generator(device="cuda").manual_seed(0)
cpu = torch.Generator().manual_seed(0)

# A float32 table: degree 10, 3 x 3 species pairs, 12 pieces over [0.3, 6), the derivative series the value
# series' derivative, the last column the edge density; K = 65 folded inputs.
pairs, pieces, degree, k, r_lo, r_max = 9, 12, 10, 65, 0.3, 6.0
coef = torch.randn(pairs, pieces, degree + 1, k + 1, generator=cpu, dtype=torch.float64) / (
    1.0 + torch.arange(degree + 1, dtype=torch.float64))[None, None, :, None] ** 2
scale = 2 * pieces / (r_max - r_lo)
dcoef = torch.from_numpy(np.polynomial.chebyshev.chebder(coef.numpy(), axis=2)) * scale
table = SimpleNamespace(coef=coef.float().cuda(), dcoef=dcoef.float().cuda(),
                        starts=torch.linspace(r_lo, r_max, pieces + 1, dtype=torch.float64)[:-1].float().cuda(),
                        scales=torch.full((pieces,), scale, dtype=torch.float32, device="cuda"),
                        bins=torch.arange(pieces, dtype=torch.int32, device="cuda"), r_lo=r_lo,
                        bin_scale=pieces / (r_max - r_lo), r_max=r_max)

nodes, edges, padding = 53, 1500, 70
receiver = torch.sort(torch.randint(0, nodes, (edges,), device="cuda", generator=g)).values
receiver[nodes // 2 == receiver] = nodes // 2 + 1  # one receiver without edges
receiver = torch.sort(receiver).values
sender = torch.randint(0, nodes, (edges,), device="cuda", generator=g)
key = torch.where(torch.arange(edges, device="cuda") < edges - padding, receiver, torch.full_like(receiver, nodes))
rows = torch.searchsorted(key, torch.arange(nodes + 1, device="cuda")).to(torch.int32)
r = 1.0 + (r_max - 1.0) * torch.rand((edges, 1), device="cuda", generator=g)
r[::17] = r_max + 0.3 * torch.rand((r[::17].shape[0], 1), device="cuda", generator=g)  # the skin
r[edges - padding:] = r_max + 1.0  # padding past every row
r[5] = r_max
pair = torch.randint(0, pairs, (edges,), device="cuda", generator=g, dtype=torch.int32)
outside = (r.reshape(-1) >= r_max)
report = {}


def run(x, y, gout, dd, kernels, weight, fused, w_prime):
    rr, xr, yr = (t.clone().requires_grad_() for t in (r, x, y))
    s32, r32 = sender.int(), receiver.int()
    if w_prime is None:
        a, density = radial_table(rr, pair, table.coef, table.dcoef, table.starts, table.scales, table.bins,
                                  table.r_lo, table.bin_scale, table.r_max)
        message = folded_streaming_convolution(a, weight, xr, yr, s32, r32, rows, kernels, 256, fused=fused)
    else:
        message, density = w_prime.convolution(rr, pair, table, weight, xr, yr, s32, r32, rows, kernels, fused)
    grads = torch.autograd.grad([message, density], (rr, xr, yr), [gout, dd])
    return [message.detach(), density.detach(), *grads]


for layout in ("ir_mul", "mul_ir"):
    kernels = StreamingKernels(problem, np.float32, layout=layout)
    weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g) / k ** 0.5
    w_prime = wp.WPrime(table, weight, kernels)
    assert w_prime.gprime.shape == (pairs * pieces, degree, problem.weight_numel)
    X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
    Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
    G = torch.randn((nodes, irreps_out.dim), device="cuda", generator=g)
    D = torch.randn((edges, 1), device="cuda", generator=g)
    for fused in (None, FusedFoldedForward(kernels, k)):
        name = f"{layout}-{'fused' if fused is not None else 'cublas'}"
        chain = run(X, Y, G, D, kernels, weight, fused, None)
        new = run(X, Y, G, D, kernels, weight, fused, w_prime)
        assert torch.equal(new[0], chain[0]) and torch.equal(new[1], chain[1]), name  # the same forward
        torch.testing.assert_close(new[3], chain[3], rtol=1e-5, atol=1e-6 * float(chain[3].abs().max()))  # dx
        torch.testing.assert_close(new[4], chain[4], rtol=1e-6, atol=1e-7 * float(chain[4].abs().max()))  # dsh
        dr_new, dr_chain = new[2].reshape(-1), chain[2].reshape(-1)
        assert bool((dr_new[outside] == 0).all()) and bool((dr_chain[outside] == 0).all()), name
        # Float64 reference from the vendored backward's weight gradient: the w'-dot and the chain in float64.
        a, _ = radial_table(r, pair, table.coef, table.dcoef, table.starts, table.scales, table.bins, r_lo,
                            table.bin_scale, r_max)
        w = a @ weight.t()
        _, _, dw = streaming_conv_backward(X, Y, w, sender.int(), receiver.int(), rows, G, kernels.id)
        dw64 = dw.double()
        dd64 = D.double()
        ref = (wp.wprime_reference(dw64, r.double(), pair, wp.derivative_table(table.dcoef.double().cpu(),
                                                                                weight.double().cpu()).cuda(),
                                   table.starts.double(), table.scales.double(), table.bins, r_lo,
                                   table.bin_scale, r_max)
               + wp.density_backward(r.double().cpu(), pair.cpu(), wp.density_column(table.dcoef.double().cpu()),
                                     table.starts.double().cpu(), table.scales.double().cpu(), table.bins.cpu(),
                                     dd64.cpu(), r_lo, table.bin_scale, r_max).cuda())
        largest = float(ref.abs().max())
        err_new = float((dr_new.double() - ref).abs().max()) / largest
        err_chain = float((dr_chain.double() - ref).abs().max()) / largest
        assert err_new <= 1e-5, (name, err_new, err_chain)
        report[name] = {"dr error (new)": err_new, "dr error (chain)": err_chain,
                        "dsh bitwise": bool(torch.equal(new[4], chain[4])),
                        "dx max rel": float((new[3] - chain[3]).abs().max() / chain[3].abs().max())}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(X, Y, G, D, kernels, weight, fused, w_prime)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0], new[0]) and torch.equal(captured[4], new[4]), name
        torch.testing.assert_close(captured[2], new[2], rtol=0, atol=0)  # no atomics in dr
        torch.testing.assert_close(captured[3], new[3], rtol=1e-5, atol=1e-6 * float(new[3].abs().max()))
print(json.dumps(report))
"""


def test_wprime_backward_matches_the_chain():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(report, indent=1))
    assert set(report) == {"ir_mul-cublas", "ir_mul-fused", "mul_ir-cublas", "mul_ir-fused"}
