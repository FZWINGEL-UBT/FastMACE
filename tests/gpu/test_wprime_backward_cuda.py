"""wprime_backward (opt-9 OPT9-T03) on CUDA against the chain it replaces.

The chain: the radial table gives ``a`` and the edge density, ``folded_streaming_convolution`` builds ``w = a
[W, b]^T`` and the message, and its chunked backward gives ``dx``, ``dsh`` and ``da``, which the table's backward
turns into ``dL/dr`` together with the density's gradient. The record must give the same message and density
(the same forward), ``dsh`` as the vendored backward's (the same sums), ``dx`` to the atomics' order, and
``dL/dr`` to float32 rounding; both are compared with a float64 reference built from the same weight gradient.
Edges at and beyond the cutoff and padding edges past every row get a zero radial slope. Both layouts, the
cuBLAS and the fused forward, a CUDA graph, and the q-major derivative table (``SLOPE_VECTOR`` 2 and 4) bit for bit
against the term-major one. ``TABLE_W`` (w and its slope from the value table, no stored w): the same message and
density, ``dx`` and ``dsh`` to rounding, ``dL/dr`` against the float64 reference, also in a CUDA graph. Runs in a fresh
interpreter like test_fused_folded_forward_cuda.py.
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
table = SimpleNamespace(coef=coef.float().cuda(), dcoef=dcoef.float().contiguous().cuda(),  # the kernels' layout
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
        assert err_new <= 1e-5 and err_chain <= 1e-5, (name, err_new, err_chain)
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

# SLOPE_VECTOR: the q-major derivative table loaded as float2/float4 (float4: two zero padding terms) gives the
# same dr and dsh bit for bit (the same sums in the same order) and dx to the atomics' order.
kernels = StreamingKernels(problem, np.float32, layout="ir_mul")
weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g) / k ** 0.5
X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
G = torch.randn((nodes, irreps_out.dim), device="cuda", generator=g)
D = torch.randn((edges, 1), device="cuda", generator=g)
fused = FusedFoldedForward(kernels, k)
base = run(X, Y, G, D, kernels, weight, fused, wp.WPrime(table, weight, kernels))
for vector in (2, 4):
    wp.SLOPE_VECTOR = vector
    try:
        w_prime = wp.WPrime(table, weight, kernels)
    finally:
        wp.SLOPE_VECTOR = 0
    name = f"slope-vector-{vector}"
    assert w_prime.gprime.shape == (pairs * pieces, problem.weight_numel, wp.padded_terms(degree, vector)), name
    new = run(X, Y, G, D, kernels, weight, fused, w_prime)
    assert torch.equal(new[0], base[0]) and torch.equal(new[1], base[1]), name
    assert torch.equal(new[2], base[2]) and torch.equal(new[4], base[4]), name  # dr, dsh
    torch.testing.assert_close(new[3], base[3], rtol=1e-5, atol=1e-6 * float(base[3].abs().max()))  # dx
    report[name] = {"dr and dsh bitwise": True,
                    "dx max rel": float((new[3] - base[3]).abs().max() / base[3].abs().max())}

# TABLE_W: the backward evaluates w and w' from the q-major value table (degree 10: 11 terms, padded to 12); the
# forward (cuBLAS, or the fused forward's message-only kernel) stores no w. The message and density are the stored-w
# path's bit for bit; dx and dsh differ by the w rounding; dr is checked against the float64 reference.
a, _ = radial_table(r, pair, table.coef, table.dcoef, table.starts, table.scales, table.bins, r_lo, table.bin_scale,
                    r_max)
_, _, dw = streaming_conv_backward(X, Y, a @ weight.t(), sender.int(), receiver.int(), rows, G, kernels.id)
ref = (wp.wprime_reference(dw.double(), r.double(), pair, wp.derivative_table(table.dcoef.double().cpu(),
                                                                              weight.double().cpu()).cuda(),
                           table.starts.double(), table.scales.double(), table.bins, r_lo, table.bin_scale, r_max)
       + wp.density_backward(r.double().cpu(), pair.cpu(), wp.density_column(table.dcoef.double().cpu()),
                             table.starts.double().cpu(), table.scales.double().cpu(), table.bins.cpu(),
                             D.double().cpu(), r_lo, table.bin_scale, r_max).cuda())
wp.TABLE_W = True
try:
    w_prime = wp.WPrime(table, weight, kernels)
finally:
    wp.TABLE_W = None
assert w_prime.gprime.shape == (pairs * pieces, problem.weight_numel, wp.padded_terms(degree + 1, 2))
for forward in (None, fused):
    name = f"table-w-{'fused' if forward is not None else 'cublas'}"
    stored = run(X, Y, G, D, kernels, weight, forward, wp.WPrime(table, weight, kernels))
    new = run(X, Y, G, D, kernels, weight, forward, w_prime)
    assert torch.equal(new[0], stored[0]) and torch.equal(new[1], stored[1]), name  # the same forward
    torch.testing.assert_close(new[3], stored[3], rtol=1e-5, atol=2e-6 * float(stored[3].abs().max()))  # dx
    torch.testing.assert_close(new[4], stored[4], rtol=1e-5, atol=2e-6 * float(stored[4].abs().max()))  # dsh
    dr_new = new[2].reshape(-1)
    assert bool((dr_new[outside] == 0).all()), name
    err = float((dr_new.double() - ref).abs().max()) / float(ref.abs().max())
    assert err <= 1e-5, (name, err)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(X, Y, G, D, kernels, weight, forward, w_prime)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], new[0]) and torch.equal(captured[2], new[2]), name
    report[name] = {"dr error": err, "dsh max rel": float((new[4] - stored[4]).abs().max() / stored[4].abs().max()),
                    "dx max rel": float((new[3] - stored[3]).abs().max() / stored[3].abs().max())}

# FACTORED: each path's three trilinear sums regrouped. Against the same record unfactored: the same forward, dx and
# dsh to rounding, dL/dr against the float64 reference (stored w and TABLE_W, both layouts), also in a CUDA graph.
for layout in ("ir_mul", "mul_ir"):
    kernels = StreamingKernels(problem, np.float32, layout=layout)
    weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g) / k ** 0.5
    X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
    Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
    G = torch.randn((nodes, irreps_out.dim), device="cuda", generator=g)
    D = torch.randn((edges, 1), device="cuda", generator=g)
    fused = FusedFoldedForward(kernels, k)
    a, _ = radial_table(r, pair, table.coef, table.dcoef, table.starts, table.scales, table.bins, r_lo,
                        table.bin_scale, r_max)
    _, _, dw = streaming_conv_backward(X, Y, a @ weight.t(), sender.int(), receiver.int(), rows, G, kernels.id)
    ref = (wp.wprime_reference(dw.double(), r.double(), pair, wp.derivative_table(table.dcoef.double().cpu(),
                                                                                  weight.double().cpu()).cuda(),
                               table.starts.double(), table.scales.double(), table.bins, r_lo, table.bin_scale,
                               r_max)
           + wp.density_backward(r.double().cpu(), pair.cpu(), wp.density_column(table.dcoef.double().cpu()),
                                 table.starts.double().cpu(), table.scales.double().cpu(), table.bins.cpu(),
                                 D.double().cpu(), r_lo, table.bin_scale, r_max).cuda())
    for table_w, scale in ((False, False), (True, False), (True, True)):
        name = f"factored-{layout}-{'table-w' if table_w else 'stored-w'}{'-scaled' if scale else ''}"
        wp.TABLE_W = table_w
        try:
            wp.FACTORED = False  # plain on every GPU (FACTORED is a per-GPU default)
            plain = wp.WPrime(table, weight, kernels)
            wp.FACTORED, wp.FACTORED_SCALE = True, scale
            try:
                factored = wp.WPrime(table, weight, kernels)
            finally:
                wp.FACTORED, wp.FACTORED_SCALE = False, False
        finally:
            wp.TABLE_W = None
        assert factored.kernel.factored and not plain.kernel.factored, name
        assert "u_0" in factored.kernel.source and "u_0" not in plain.kernel.source, name
        base = run(X, Y, G, D, kernels, weight, fused, plain)
        new = run(X, Y, G, D, kernels, weight, fused, factored)
        assert torch.equal(new[0], base[0]) and torch.equal(new[1], base[1]), name  # the same forward
        torch.testing.assert_close(new[3], base[3], rtol=1e-5, atol=2e-6 * float(base[3].abs().max()))  # dx
        torch.testing.assert_close(new[4], base[4], rtol=1e-5, atol=2e-6 * float(base[4].abs().max()))  # dsh
        dr_new = new[2].reshape(-1)
        assert bool((dr_new[outside] == 0).all()), name
        err = float((dr_new.double() - ref).abs().max()) / float(ref.abs().max())
        err_base = float((base[2].reshape(-1).double() - ref).abs().max()) / float(ref.abs().max())
        assert err <= 1e-5, (name, err)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(X, Y, G, D, kernels, weight, fused, factored)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0], new[0]) and torch.equal(captured[2], new[2]), name
        report[name] = {"dr error": err, "dr error (unfactored)": err_base,
                        "dsh max rel": float((new[4] - base[4]).abs().max() / base[4].abs().max()),
                        "dx max rel": float((new[3] - base[3]).abs().max() / base[3].abs().max())}
    # Round 2 (5 November 2026): HORNER (the value table in the power basis), with SCALAR_PATHS (l1 = 0 paths without
    # the v_j sums, FACTORED) and with MAX_REGISTERS 80. dL/dr against the float64 reference, the rest against TABLE_W.
    for horner_case in ("horner", "horner-scalar", "horner-80"):
        name = f"{horner_case}-{layout}"
        wp.TABLE_W = True
        try:
            wp.HORNER, wp.SCALAR_PATHS, wp.MAX_REGISTERS = False, False, 0  # plain on every GPU (per-GPU defaults)
            plain = wp.WPrime(table, weight, kernels)
            wp.HORNER = True
            wp.FACTORED = True if horner_case == "horner-scalar" else None
            wp.SCALAR_PATHS = horner_case == "horner-scalar"
            wp.MAX_REGISTERS = 80 if horner_case == "horner-80" else 0
            try:
                new_w = wp.WPrime(table, weight, kernels)
            finally:
                wp.HORNER = wp.FACTORED = wp.SCALAR_PATHS = wp.MAX_REGISTERS = None
        finally:
            wp.TABLE_W = None
        assert new_w.horner and new_w.kernel.horner and not plain.horner, name
        assert new_w.kernel.scalar_paths == (horner_case == "horner-scalar"), name
        assert ("maxrregcount=80" in new_w.kernel.source) == (horner_case == "horner-80"), name
        base = run(X, Y, G, D, kernels, weight, fused, plain)
        new = run(X, Y, G, D, kernels, weight, fused, new_w)
        assert torch.equal(new[0], base[0]) and torch.equal(new[1], base[1]), name  # the same forward
        torch.testing.assert_close(new[3], base[3], rtol=1e-5, atol=2e-6 * float(base[3].abs().max()))  # dx
        torch.testing.assert_close(new[4], base[4], rtol=1e-5, atol=2e-6 * float(base[4].abs().max()))  # dsh
        dr_new = new[2].reshape(-1)
        assert bool((dr_new[outside] == 0).all()), name
        err = float((dr_new.double() - ref).abs().max()) / float(ref.abs().max())
        assert err <= 1e-5, (name, err)
        report[name] = {"dr error": err}
    # Round 5 (5 November 2026): BUTTERFLY, the end-of-edge reduction as a butterfly reduce-scatter. The same tree of
    # pairwise sums, so dsh and dL/dr are bitwise those of the shfl_down reduction (dx: atomics' order).
    for table_w in (False, True):
        name = f"butterfly-{layout}-{'table-w' if table_w else 'stored-w'}"
        wp.TABLE_W = table_w
        try:
            wp.BUTTERFLY, wp.REDUX = False, False
            plain = wp.WPrime(table, weight, kernels)
            wp.BUTTERFLY = True
            new_w = wp.WPrime(table, weight, kernels)
        finally:
            wp.TABLE_W = wp.BUTTERFLY = wp.REDUX = None
        assert new_w.kernel.butterfly and not plain.kernel.butterfly, name
        assert "__shfl_xor_sync" in new_w.kernel.source and "__shfl_xor_sync" not in plain.kernel.source, name
        base = run(X, Y, G, D, kernels, weight, fused, plain)
        new = run(X, Y, G, D, kernels, weight, fused, new_w)
        assert torch.equal(new[0], base[0]) and torch.equal(new[1], base[1]), name  # the same forward
        assert torch.equal(new[4], base[4]) and torch.equal(new[2], base[2]), name  # dsh, dL/dr bitwise
        torch.testing.assert_close(new[3], base[3], rtol=1e-5, atol=2e-6 * float(base[3].abs().max()))  # dx
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(X, Y, G, D, kernels, weight, fused, new_w)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[2], new[2]) and torch.equal(captured[4], new[4]), name
        report[name] = {"bitwise": True}
    # REDUX: the end-of-edge sums in fixed point (redux.sync): FP32 class against the shuffle tree, deterministic,
    # and a NaN partial still gives NaN where the shuffle tree gives NaN.
    for table_w in (False, True):
        name = f"redux-{layout}-{'table-w' if table_w else 'stored-w'}"
        wp.TABLE_W = table_w
        try:
            wp.REDUX = wp.BUTTERFLY = False
            plain = wp.WPrime(table, weight, kernels)
            wp.REDUX = True
            new_w = wp.WPrime(table, weight, kernels)
        finally:
            wp.TABLE_W = wp.BUTTERFLY = wp.REDUX = None
        assert new_w.kernel.redux and not plain.kernel.redux and not new_w.kernel.butterfly, name
        assert "__reduce_add_sync" in new_w.kernel.source and "__reduce_add_sync" not in plain.kernel.source, name
        base = run(X, Y, G, D, kernels, weight, fused, plain)
        new = run(X, Y, G, D, kernels, weight, fused, new_w)
        assert torch.equal(new[0], base[0]) and torch.equal(new[1], base[1]), name  # the same forward
        torch.testing.assert_close(new[4], base[4], rtol=1e-5, atol=2e-6 * float(base[4].abs().max()))  # dsh
        torch.testing.assert_close(new[2], base[2], rtol=1e-5, atol=2e-6 * float(base[2].abs().max()))  # dL/dr
        torch.testing.assert_close(new[3], base[3], rtol=1e-5, atol=2e-6 * float(base[3].abs().max()))  # dx
        again = run(X, Y, G, D, kernels, weight, fused, new_w)
        assert torch.equal(again[4], new[4]) and torch.equal(again[2], new[2]), name  # deterministic
        bad = X.clone()
        bad[0] = float("nan")
        base_nan = run(bad, Y, G, D, kernels, weight, fused, plain)
        new_nan = run(bad, Y, G, D, kernels, weight, fused, new_w)
        assert bool(torch.isnan(base_nan[4]).any()), name
        assert torch.equal(torch.isnan(new_nan[4]), torch.isnan(base_nan[4])), name
        assert torch.equal(torch.isnan(new_nan[2]), torch.isnan(base_nan[2])), name
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(X, Y, G, D, kernels, weight, fused, new_w)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[2], new[2]) and torch.equal(captured[4], new[4]), name
        report[name] = {"dsh": float((new[4] - base[4]).abs().max() / base[4].abs().max()),
                        "dr": float((new[2] - base[2]).abs().max() / base[2].abs().max())}
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
    assert set(report) == {"ir_mul-cublas", "ir_mul-fused", "mul_ir-cublas", "mul_ir-fused", "slope-vector-2",
                           "slope-vector-4", "table-w-cublas", "table-w-fused", "factored-ir_mul-stored-w",
                           "factored-ir_mul-table-w", "factored-mul_ir-stored-w", "factored-mul_ir-table-w",
                           "factored-ir_mul-table-w-scaled", "factored-mul_ir-table-w-scaled"} | {
                               f"{case}-{layout}" for case in ("horner", "horner-scalar", "horner-80")
                               for layout in ("ir_mul", "mul_ir")} | {
                               f"{case}-{layout}-{w}" for case in ("butterfly", "redux")
                               for layout in ("ir_mul", "mul_ir") for w in ("stored-w", "table-w")}
