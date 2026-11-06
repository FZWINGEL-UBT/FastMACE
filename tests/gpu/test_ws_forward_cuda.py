"""WS_FORWARD (H200 ideas plan, idea 30; ``ws_forward.apply_ws_forward``) on Hopper against today's kernel.

The H200 plan's message kernel (tensor cores, TC_WFRAG, W_GROUP, VEC_A, VEC_SH, X_LATE, CG_FOLD, DEGREE_PARTS,
LANE_INDICES; ``store_w=False``) with and without the warp-specialized wgmma forward, on a synthetic MACE-like
problem with an uneven neighbourhood (empty rows, empty receiver chunks, a few receivers with 150 edges, padding
edges), in both layouts: the messages agree to the 3xTF32 rounding class (max |d| / max |ref| < 1e-5, as CG_FOLD's
check), three launches each. The block layout (BLOCK_MESSAGE, the MD's) is covered by the force/stress gate.
Compute capability 9.0 only (sm_90a: wgmma, mbarrier); skipped elsewhere. The protocol itself (ring, barriers,
chunk walk, zero rows, block orders) is checked on any GPU by ``tools/check_idea30_local.py --run`` (an FP32-SIMT
producer, bitwise the reference). Fresh interpreter like test_fused_folded_forward_cuda.py.
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


def _hopper():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    except Exception:  # noqa: BLE001
        return False


pytestmark = [pytest.mark.skipif(not streaming_available(), reason=streaming_available().reason),
              pytest.mark.skipif(not _hopper(), reason="WS_FORWARD builds only on compute capability 9.0 (sm_90a)")]

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

from fastmace.kernels.oeq_streaming import StreamingKernels
from fastmace.kernels.oeq_streaming import fused_forward as ff_module
from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward, fused_folded_message

PLAN = dict(TENSOR_CORES=True, TC_WFRAG=True, W_GROUP=True, VEC_A=True, VEC_SH=True, X_LATE=True, CG_FOLD=True,
            DEGREE_PARTS=True, LANE_INDICES=True, SH_LDS4=False)


def build(kernels, k, used, ws):
    saved = {name: getattr(ff_module, name) for name in (*PLAN, "WS_FORWARD")}
    try:
        for name, value in PLAN.items():
            setattr(ff_module, name, value)
        ff_module.WS_FORWARD = ws
        return FusedFoldedForward(kernels, k, used=used, store_w=False)
    finally:
        for name, value in saved.items():
            setattr(ff_module, name, value)


C = 128
irreps_in1, irreps_in2 = oeq.Irreps(f"{C}x0e+{C}x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o")
irreps_out = oeq.Irreps(f"{C}x0e+{C}x0e+{C}x1o+{C}x1o+{C}x2e+{C}x2e+{C}x3o")
instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True), (0, 2, 4, "uvu", True), (1, 1, 5, "uvu", True), (0, 3, 6, "uvu", True)]
problem = oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False,
                        internal_weights=False)
g = torch.Generator(device="cuda").manual_seed(30)
nodes, padding = 3000, 70
degree = torch.randint(0, 40, (nodes,), device="cuda", generator=g)
degree[torch.rand(nodes, device="cuda", generator=g) < 0.15] = 0  # empty rows
degree[100:110] = 0  # whole empty receiver chunks
degree[::37] = 150
real = torch.repeat_interleave(torch.arange(nodes, device="cuda"), degree)
receiver = torch.cat([real, torch.full((padding,), nodes - 1, device="cuda")])
key = torch.cat([real, torch.full((padding,), nodes, device="cuda")])
rows = torch.searchsorted(key, torch.arange(nodes + 1, device="cuda")).to(torch.int32)
edges = receiver.numel()
sender = torch.randint(0, nodes, (edges,), device="cuda", generator=g)
args = (sender.int(), receiver.int(), rows)
report = {}
for layout in ("ir_mul", "mul_ir"):
    kernels = StreamingKernels(problem, np.float32, layout=layout)
    X = torch.randn((nodes, irreps_in1.dim), device="cuda", generator=g)
    Y = torch.randn((edges, irreps_in2.dim), device="cuda", generator=g)
    k, used = 65, 64
    a = torch.randn((edges, k), device="cuda", generator=g)
    weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g) * 0.2
    weight[:, -1] = 0.0
    off, on = build(kernels, k, used, False), build(kernels, k, used, True)
    assert not off.ws_forward, layout
    assert on.ws_forward, (layout, "WS_FORWARD did not apply to the H200 plan's render")
    assert on.rule.warps == 12 and on.source != off.source, layout
    ref = fused_folded_message(a, weight, X, Y, *args, off.id)
    scale = ref.abs().max().clamp_min(1e-30)
    errors = []
    for _ in range(3):
        out = fused_folded_message(a, weight, X, Y, *args, on.id)
        assert not torch.isnan(out).any(), layout
        errors.append(((out - ref).abs().max() / scale).item())
    assert max(errors) < 1e-5, (layout, errors)
    report[layout] = {"max rel": max(errors), "shared": on.rule.shared_bytes}
print(json.dumps(report))
"""


def test_ws_forward_matches_todays_kernel():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(report, indent=1))
    assert set(report) == {"ir_mul", "mul_ir"}
