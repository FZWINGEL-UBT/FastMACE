"""fastmace.kernels.oeq_streaming on CUDA: the streaming convolution against OEq's TensorProductConv and the baseline.

Ported from the research engine's tests/core/test_oeq_streaming.py. Needs OpenEquivariance, NVRTC through
cuda.bindings and CUDA (the hybrid environment). Each check runs in a fresh interpreter, because the OEq
runtime must be prepared (fastmace.kernels._oeq_runtime.prepare_in_process) before anything imports
OpenEquivariance or torch.utils.cpp_extension; the child prepares it under the FastMACE cache root.

With FASTMACE_BASELINE set, the child also builds the research engine's kernels from the export
(ResearchModule tag research-engine-baseline-20261017): the rendered CUDA source must be identical, the
forward (one thread per receiver, no atomics) bitwise equal, and the backward (sender gradients added
atomically) equal at FP32 rounding.
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

PRELUDE = r"""
import fastmace  # before torch: soft file locks (SPEC §14)
import json, os, sys, types
from pathlib import Path

if "openequivariance" not in sys.modules and not os.environ.get("FASTMACE_OEQ_PREPARED"):
    from fastmace._cache import cache_dir
    from fastmace.kernels import _oeq_runtime
    _oeq_runtime.prepare_in_process(cache_dir("oeq"))

import numpy as np
import openequivariance as oeq
import torch

from fastmace.kernels.oeq_streaming import StreamingKernels, render_kernels, streaming_convolution

torch.backends.cuda.matmul.allow_tf32 = False

irreps_in1, irreps_in2 = oeq.Irreps("8x0e+8x1o"), oeq.Irreps("1x0e+1x1o+1x2e")
# MH-1-like paths: 0e x 0e -> 0e, 1o x 1o -> 0e, 0e x 1o -> 1o, 1o x 0e -> 1o, 0e x 2e -> 2e, 1o x 1o -> 2e
irreps_out = oeq.Irreps("8x0e+8x0e+8x1o+8x1o+8x2e+8x2e")
instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True), (0, 2, 4, "uvu", True), (1, 1, 5, "uvu", True)]
problem = oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False,
                        internal_weights=False)
g = torch.Generator(device="cuda").manual_seed(0)
nodes, edges = 37, 400
receiver = torch.sort(torch.randint(0, nodes, (edges,), device="cuda", generator=g)).values
sender = torch.randint(0, nodes, (edges,), device="cuda", generator=g)
row_ptr = torch.searchsorted(receiver, torch.arange(nodes + 1, device="cuda")).to(torch.int32)
X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
W = torch.randn((edges, problem.weight_numel), device="cuda", generator=g)
G = torch.randn((nodes, irreps_out.dim), device="cuda", generator=g)


def to_ir_mul(tensor, irreps):
    pieces, start = [], 0
    for mul, ir in irreps:
        width = mul * ir.dim
        block = tensor[:, start:start + width].reshape(-1, mul, ir.dim).transpose(1, 2)
        pieces.append(block.reshape(-1, width))
        start += width
    return torch.cat(pieces, dim=1)


def run(convolution, kernels, layout="mul_ir"):
    x = X if layout == "mul_ir" else to_ir_mul(X, irreps_in1)
    grad = G if layout == "mul_ir" else to_ir_mul(G, irreps_out)
    xr, yr, wr = (t.clone().requires_grad_() for t in (x, Y, W))
    out = convolution(xr, yr, wr, sender.int(), receiver.int(), row_ptr, kernels)
    return (out.detach(),) + torch.autograd.grad(out, (xr, yr, wr), grad)
"""

AGAINST_OEQ = r"""
conv = oeq.TensorProductConv(problem, deterministic=False)
xr, yr, wr = (t.clone().requires_grad_() for t in (X, Y, W))
out = conv(xr, yr, wr, receiver, sender)
reference = (out.detach(),) + torch.autograd.grad(out, (xr, yr, wr), G)
new = run(streaming_convolution, StreamingKernels(problem, np.float32))
for n, o in zip(new, reference):
    torch.testing.assert_close(n, o, rtol=1e-4, atol=1e-4)

# The ir_mul variant on transposed inputs gives the transposed OEq results, with the edge-owned
# (PR) and the receiver-owned reverse pass.
for receiver_backward in (False, True):
    kernels = StreamingKernels(problem, np.float32, layout="ir_mul", receiver_backward=receiver_backward)
    out, dx, dy, dw = run(streaming_convolution, kernels, "ir_mul")
    torch.testing.assert_close(out, to_ir_mul(reference[0], irreps_out), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dx, to_ir_mul(reference[1], irreps_in1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dy, reference[2], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, reference[3], rtol=1e-4, atol=1e-4)

# Capturable: a CUDA graph replays forward and backward.
kernels = StreamingKernels(problem, np.float32)
eager = run(streaming_convolution, kernels)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    captured = run(streaming_convolution, kernels)
graph.replay()
torch.cuda.synchronize()
assert torch.equal(captured[0], eager[0])
for n, o in zip(captured[1:], eager[1:]):
    torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-5)
print(json.dumps({"ok": True}))
"""

FOLDED_CHUNKED = r"""
from fastmace.kernels.oeq_streaming import folded_streaming_convolution
from fastmace.kernels.triton.skinny_linear import folded_linear

# chunked_conv_backward (opt-6 OPT6-T02): the per-edge weights a @ [W, b].T built inside the convolution and
# the backward in edge chunks equal folded_linear followed by streaming_convolution. The last 40 edges are
# padding past every row, as the engine's edges after the compacted count.
a = torch.randn((edges, 9), device="cuda", generator=g)
weight = torch.randn((problem.weight_numel, 9), device="cuda", generator=g)
key = torch.where(torch.arange(edges, device="cuda") < edges - 40, receiver, torch.full_like(receiver, nodes))
rows = torch.searchsorted(key, torch.arange(nodes + 1, device="cuda")).to(torch.int32)


def reference_path(x, y, ar, kernels):
    return streaming_convolution(x, y, folded_linear(ar, weight), sender.int(), receiver.int(), rows, kernels)


def chunked_path(chunk, parts=None):
    def path(x, y, ar, kernels):
        return folded_streaming_convolution(ar, weight, x, y, sender.int(), receiver.int(), rows, kernels, chunk,
                                            parts=parts)
    return path


def gradients(path, kernels):
    xr, yr, ar = (t.clone().requires_grad_() for t in (X, Y, a))
    out = path(xr, yr, ar, kernels)
    return (out.detach(),) + torch.autograd.grad(out, (xr, yr, ar), G)


kernels = StreamingKernels(problem, np.float32)
reference = gradients(reference_path, kernels)
halves = (weight[:, :8].contiguous(), weight[:, 8].contiguous())
for chunk, parts in ((64, None), (150, halves), (399, None), (4096, None)):
    out, dx, dy, da = gradients(chunked_path(chunk, parts), kernels)
    assert torch.equal(out, reference[0]), chunk  # the same GEMM and forward kernel
    assert torch.equal(dy, reference[2]), chunk  # per edge, no atomics
    assert torch.equal(da, reference[3]), chunk  # the same input-gradient reduction order per row
    torch.testing.assert_close(dx, reference[1], rtol=1e-5, atol=1e-5)  # atomic sender sums

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    captured = gradients(chunked_path(64), kernels)
graph.replay()
torch.cuda.synchronize()
for n, o in zip(captured, reference):
    torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-5)
assert torch.equal(captured[3], reference[3])
print(json.dumps({"ok": True}))
"""

AGAINST_BASELINE = r"""
directory = Path(os.environ["FASTMACE_BASELINE"]).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
package = types.ModuleType("langevin")
package.__path__ = [str(directory)]
sys.modules["langevin"] = package  # langevin/__init__.py is not run
from langevin.oeq_streaming import StreamingKernels as BaselineKernels
from langevin.oeq_streaming import streaming_convolution as baseline_convolution

report = {}
for layout, receiver_backward, dtype in (("mul_ir", False, np.float32), ("ir_mul", False, np.float32),
                                         ("ir_mul", True, np.float32), ("mul_ir", False, np.float64)):
    new_kernels = StreamingKernels(problem, dtype, layout=layout, receiver_backward=receiver_backward)
    old_kernels = BaselineKernels(problem, dtype, layout=layout, receiver_backward=receiver_backward)
    name = f"{layout}-{'receiver' if receiver_backward else 'edge'}-{np.dtype(dtype).name}"
    assert new_kernels.source == old_kernels.source, name
    assert new_kernels.source == render_kernels(problem, dtype, layout=layout)[1].jit_kernel, name
    if dtype is np.float64:
        report[name] = "source identical"
        continue
    new = run(streaming_convolution, new_kernels, layout)
    old = run(baseline_convolution, old_kernels, layout)
    assert torch.equal(new[0], old[0]), name  # forward: no atomics
    differences = [float((n - o).abs().max()) for n, o in zip(new[1:], old[1:])]
    for n, o in zip(new[1:], old[1:]):
        torch.testing.assert_close(n, o, rtol=1e-5, atol=1e-5)
    report[name] = {"forward": "bitwise", "backward max |difference|": differences}
print(json.dumps(report))
"""


def run_child(body):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(PRELUDE) + textwrap.dedent(body)],
                            cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_message_and_gradients_match_oeq():
    assert run_child(AGAINST_OEQ) == {"ok": True}


def test_chunked_folded_backward_matches_the_unchunked_path():
    assert run_child(FOLDED_CHUNKED) == {"ok": True}


def test_kernels_equal_baseline():
    if not os.environ.get("FASTMACE_BASELINE"):
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    report = run_child(AGAINST_BASELINE)
    print(json.dumps(report, indent=1))
    assert len(report) == 4
