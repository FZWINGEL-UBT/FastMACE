"""fused_folded_forward (opt-8 OPT8-T02) on CUDA against the chain it replaces.

The chain: ``w = a @ W.T`` (cuBLAS), then the streaming forward. The fused kernel must store the same ``w``
(FP32 rounding; on the RTX 5060 Ti bitwise) and a message bitwise equal to the streaming forward on the ``w`` it
stored (the same angular sums in the same edge order), including the padding edges past every row; the gradients
through ``folded_streaming_convolution`` equal the chain's (the backward is the same). Both layouts, two folded
widths, a CUDA graph. Runs in a fresh interpreter like test_kernels_oeq_streaming_cuda.py.
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
        for n, o in zip(new, chain):
            torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = grads(fused)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0], new[0])  # the forward has no atomics: replay-stable
        for n, o in zip(captured[1:], new[1:]):
            torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-5)
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
