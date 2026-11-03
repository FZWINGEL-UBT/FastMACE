"""fused_folded_forward (opt-8 OPT8-T02) on CUDA against the chain it replaces.

The chain: ``w = a @ W.T`` (cuBLAS), then the streaming forward. The fused kernel must store the same ``w``
(FP32 rounding; on the RTX 5060 Ti bitwise) and a message bitwise equal to the streaming forward on the ``w`` it
stored (the same angular sums in the same edge order), including the padding edges past every row; the gradients
through ``folded_streaming_convolution`` equal the chain's (the backward is the same). Both layouts, two folded
widths, a CUDA graph, and ``SKIP_ZERO_COLUMN``'s kernel bit for bit on a weight with a zero last column. Runs in a
fresh interpreter like test_kernels_oeq_streaming_cuda.py.
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

if "openequivariance" not in sys.modules and not os.environ.get("FASTMACE_OEQ_PREPARED"):
    from fastmace._cache import cache_dir
    from fastmace.kernels import _oeq_runtime
    _oeq_runtime.prepare_in_process(cache_dir("oeq"))

import numpy as np
import openequivariance as oeq
import torch

from fastmace.kernels.oeq_streaming import StreamingKernels, folded_streaming_convolution
from fastmace.kernels.oeq_streaming import fused_forward as ff_module
from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward, fused_folded_forward
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward

torch.backends.cuda.matmul.allow_tf32 = False
C = 64
irreps_in1, irreps_in2 = oeq.Irreps(f"{C}x0e+{C}x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o")
irreps_out = oeq.Irreps(f"{C}x0e+{C}x0e+{C}x1o+{C}x1o+{C}x2e+{C}x2e+{C}x3o")
instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True), (0, 2, 4, "uvu", True), (1, 1, 5, "uvu", True), (0, 3, 6, "uvu", True)]
problem = oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False,
                        internal_weights=False)
g = torch.Generator(device="cuda").manual_seed(0)
nodes, edges, padding = 53, 1500, 70
receiver = torch.sort(torch.randint(0, nodes, (edges,), device="cuda", generator=g)).values
receiver[nodes // 2 == receiver] = nodes // 2 + 1  # one receiver without edges
receiver = torch.sort(receiver).values
sender = torch.randint(0, nodes, (edges,), device="cuda", generator=g)
key = torch.where(torch.arange(edges, device="cuda") < edges - padding, receiver, torch.full_like(receiver, nodes))
rows = torch.searchsorted(key, torch.arange(nodes + 1, device="cuda")).to(torch.int32)
report = {}
for layout in ("ir_mul", "mul_ir"):
    kernels = StreamingKernels(problem, np.float32, layout=layout)
    X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
    Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
    G = torch.randn((nodes, irreps_out.dim), device="cuda", generator=g)
    for k in (9, 65):
        a = torch.randn((edges, k), device="cuda", generator=g)
        weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g)
        fused = FusedFoldedForward(kernels, k)
        args = (sender.int(), receiver.int(), rows)
        message, w = fused_folded_forward(a, weight, X, Y, *args, fused.id)
        chain_w = a @ weight.t()
        torch.testing.assert_close(w, chain_w, rtol=2e-6, atol=2e-6)  # every row, the padding included
        assert torch.equal(message, streaming_conv_forward(X, Y, w, sender.int(), rows, kernels.id)), (layout, k)
        name = f"{layout}-{k}"
        report[name] = {"w bitwise cuBLAS": bool(torch.equal(w, chain_w)),
                        "message bitwise chain": bool(torch.equal(
                            message, streaming_conv_forward(X, Y, chain_w, sender.int(), rows, kernels.id)))}

        def grads(f):
            xr, yr, ar = (t.clone().requires_grad_() for t in (X, Y, a))
            out = folded_streaming_convolution(ar, weight, xr, yr, *args, kernels, 256, fused=f)
            return (out.detach(),) + torch.autograd.grad(out, (xr, yr, ar), G)

        chain, new = grads(None), grads(fused)
        assert torch.equal(new[0], chain[0]) or not report[name]["w bitwise cuBLAS"]
        for n, o in zip(new, chain):  # dx: atomic sender sums, the order differs between any two runs
            torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-6 * float(o.abs().max()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = grads(fused)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0], new[0])  # the forward has no atomics: replay-stable
        for n, o in zip(captured[1:], new[1:]):
            torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-6 * float(o.abs().max()))
        # SKIP_ZERO_COLUMN: with an all-zero last weight column the kernel that reads one input fewer stores the same
        # w and message bit for bit (the skipped term adds an exact zero).
        zero = weight.clone()
        zero[:, -1] = 0.0
        skip = FusedFoldedForward(kernels, k, used=k - 1)
        reference = fused_folded_forward(a, zero, X, Y, *args, fused.id)
        skipped = fused_folded_forward(a, zero, X, Y, *args, skip.id)
        assert torch.equal(skipped[0], reference[0]) and torch.equal(skipped[1], reference[1]), (layout, k)
        report[name]["skip zero column bitwise"] = True
        # STREAM_STORE_W: the w store with the evict-first hint stores the same values.
        ff_module.STREAM_STORE_W = True
        try:
            streamed = FusedFoldedForward(kernels, k)
        finally:
            ff_module.STREAM_STORE_W = False
        assert "__stcs(&wout" in streamed.source, (layout, k)
        hinted = fused_folded_forward(a, weight, X, Y, *args, streamed.id)
        assert torch.equal(hinted[0], message) and torch.equal(hinted[1], w), (layout, k)
        report[name]["stream store bitwise"] = True
        # TENSOR_CORES (sm_80+): w in 3xTF32 within twice the FP32 chain's error against float64 (plus a few units
        # of FP32 rounding of the row's scale), the message bitwise the streaming forward's on that w; with the
        # zero-column skip too.
        if torch.cuda.get_device_capability() >= (8, 0):
            exact = (a.double() @ weight.double().t())
            scale = (a.double().abs() @ weight.double().abs().t())
            ff_module.TENSOR_CORES = True
            try:
                tc = FusedFoldedForward(kernels, k)
                tc_skip = FusedFoldedForward(kernels, k, used=k - 1)
            finally:
                ff_module.TENSOR_CORES = False
            assert tc.tc and "mma.sync" in tc.source, (layout, k)
            tc_message, tc_w = fused_folded_forward(a, weight, X, Y, *args, tc.id)
            chain_error = float(((chain_w.double() - exact).abs() / scale).max())
            tc_error = float(((tc_w.double() - exact).abs() / scale).max())
            assert tc_error <= 2 * chain_error + 4 * 2.0 ** -24, (layout, k, tc_error, chain_error)
            assert torch.equal(tc_message, streaming_conv_forward(X, Y, tc_w, sender.int(), rows, kernels.id))
            skip_message, skip_w = fused_folded_forward(a, zero, X, Y, *args, tc_skip.id)
            zero_exact = a.double() @ zero.double().t()
            assert float(((skip_w.double() - zero_exact).abs() / scale).max()) <= 2 * chain_error + 4 * 2.0 ** -24
            report[name]["tensor cores"] = {"max error / scale": tc_error, "chain": chain_error}
print(json.dumps(report))
"""


def test_fused_forward_matches_the_chain():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(report, indent=1))
    assert set(report) == {"ir_mul-9", "ir_mul-65", "mul_ir-9", "mul_ir-65"}
