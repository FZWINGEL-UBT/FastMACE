"""The owner-pass forward (ideas plan idea 14(A), ``fused_forward.OWNER_PASS``) on CUDA.

Pass 1 (the fused forward over each receiver's owned edges, each owned edge's w stored into its mirror's row) and
pass 2 (the product-free forward over the partner edges, continuing each receiver's message) must give the message of
the one-pass message-only kernel on the same owned-first edge order, bit for bit. The edges are mirror pairs with
equal a rows (as SYMMETRIC_PAIRS' tables give them), self-image pairs, edges without a mirror, an uneven
neighbourhood (some receivers with 150 edges, others with none) and padding edges past the last row. Every kernel
form the H200 plan builds (tensor cores in fragment order, VEC_A, W_GROUP, LANE_INDICES, VEC_SH, SH_LDS4,
DEGREE_PARTS, the zero-column skip, the block-layout message) and the FP32-SIMT product, in both layouts; the pass
kernels inside a CUDA graph too. Runs in a fresh interpreter like test_fused_folded_forward_cuda.py.
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

from fastmace.kernels.oeq_streaming import StreamingKernels
from fastmace.kernels.oeq_streaming import fused_forward as ff
from fastmace.neighbors.owner import mirror_pairs, owned_first_order, owner_flags, split_rows

torch.backends.cuda.matmul.allow_tf32 = False
C = 64
irreps_in1, irreps_in2 = oeq.Irreps(f"{C}x0e+{C}x1o"), oeq.Irreps("1x0e+1x1o+1x2e+1x3o")
irreps_out = oeq.Irreps(f"{C}x0e+{C}x0e+{C}x1o+{C}x1o+{C}x2e+{C}x2e+{C}x3o")
instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (0, 1, 2, "uvu", True),
                (1, 0, 3, "uvu", True), (0, 2, 4, "uvu", True), (1, 1, 5, "uvu", True), (0, 3, 6, "uvu", True)]
problem = oeq.TPProblem(irreps_in1, irreps_in2, irreps_out, instructions, shared_weights=False,
                        internal_weights=False)
g = torch.Generator(device="cuda").manual_seed(3)
gc = torch.Generator().manual_seed(3)

# The graph: mirror pairs i <- j / j <- i with an image shift s / -s (several images for some pairs), self-image pairs
# (i <- i with s and -s), edges without a mirror, and receivers of very different degree.
N = 700
canonical = set()  # (lo, hi, shift of the lo <- hi edge); a self-image pair with a positive shift
for i in range(N):
    many = i % 41 == 0  # a dense neighbourhood
    for _ in range(int(torch.randint(60 if many else 0, 90 if many else 7, (1,), generator=gc))):
        j = int(torch.randint(0, N, (1,), generator=gc))
        s = (int(torch.randint(-1, 2, (1,), generator=gc)), int(torch.randint(-1, 2, (1,), generator=gc)), 0)
        if i == j and s == (0, 0, 0):
            continue
        if i == j and not (s[0] > 0 or (s[0] == 0 and s[1] > 0)):
            s = (-s[0], -s[1], 0)
        canonical.add((i, j, s) if i <= j else (j, i, (-s[0], -s[1], 0)))
for i in range(0, N, 13):
    canonical.add((i, i, (0, 1, 0)))  # self-image pairs
senders, receivers, shifts, mirror_list = [], [], [], []
for lo, hi, s in sorted(canonical):  # the edge lo <- hi with s, its mirror hi <- lo with -s
    n0 = len(senders)
    senders += [hi, lo]
    receivers += [lo, hi]
    shifts += [s, tuple(-v for v in s)]
    mirror_list += [n0 + 1, n0]
lone = {(int(torch.randint(0, N, (1,), generator=gc)), int(torch.randint(0, N, (1,), generator=gc))) for _ in range(90)}
for i, j in sorted(lone):  # edges without a mirror (a pair split at the cutoff + skin): shift (7, 0, 0) is unpaired
    senders.append(j)
    receivers.append(i)
    shifts.append((7, 0, 0))
    mirror_list.append(-1)
E0 = len(senders)
perm = torch.randperm(E0, generator=gc)  # then receiver-sorted (stable), as the neighbour list gives them
perm = perm[torch.sort(torch.tensor(receivers)[perm], stable=True).indices]
back = torch.empty_like(perm)
back[perm] = torch.arange(E0)
sender = torch.tensor(senders)[perm].cuda()
receiver = torch.tensor(receivers)[perm].cuda()
shift = torch.tensor(shifts)[perm].cuda()
mirror0 = torch.tensor(mirror_list)[perm]
mirror0 = torch.where(mirror0 >= 0, back[mirror0.clamp(min=0)], -1).cuda()
valid = torch.ones(E0, dtype=torch.bool, device="cuda")
assert torch.equal(mirror_pairs(sender, receiver, shift, valid, N), mirror0)  # the sort finds the generator's pairs
owned = owner_flags(sender, receiver, shift, mirror0)
order = owned_first_order(receiver, owned)
inverse = torch.empty_like(order)
inverse[order] = torch.arange(E0, device="cuda")
sender, receiver, owned = sender[order], receiver[order], owned[order]
mirror = torch.where(mirror0[order] >= 0, inverse[mirror0[order].clamp(min=0)], -1)
padding = 70
E = E0 + padding
sender = torch.cat([sender, torch.zeros(padding, dtype=torch.int64, device="cuda")]).int()
receiver_all = torch.cat([receiver, torch.zeros(padding, dtype=torch.int64, device="cuda")])
owned = torch.cat([owned, torch.ones(padding, dtype=torch.bool, device="cuda")])
mirror = torch.cat([mirror, torch.full((padding,), -1, dtype=torch.int64, device="cuda")]).int()
count = torch.tensor(E0, device="cuda")
row_ptr, mid = split_rows(receiver_all, owned, count, N)
receiver = receiver_all.int()
stats = {"edges": E0, "owned": int(owned[:E0].sum()), "without mirror": int((mirror[:E0] < 0).sum()),
         "self edges": int((sender[:E0] == receiver[:E0]).sum()), "max degree": int((row_ptr[1:] - row_ptr[:-1]).max())}
# The order checks: within each receiver its owned edges first; every partner edge's mirror is owned.
assert bool((receiver[1:E0] >= receiver[:E0 - 1]).all())
same = receiver[1:E0] == receiver[:E0 - 1]
assert not bool((same & ~owned[:E0 - 1] & owned[1:E0]).any())
partner = ~owned[:E0]
assert bool((mirror[:E0][partner] >= 0).all()) and bool(owned[mirror[:E0][partner].long()].all())
assert bool((mirror[mirror[:E0].clamp(min=0).long()][mirror[:E0] >= 0] == torch.arange(E0, device="cuda")[
    mirror[:E0] >= 0].int()).all())
for i in torch.randint(0, N, (20,), generator=g, device="cuda").tolist():
    lo, m, hi = int(row_ptr[i]), int(mid[i]), int(row_ptr[i + 1])
    assert bool(owned[lo:m].all()) and not bool(owned[m:hi].any()) and bool((receiver[lo:hi] == i).all())

report = {"graph": stats}


def mirrored(t):
    # Equal rows for both edges of a pair (the owner's), as the symmetric radial table gives them.
    t = t.clone()
    src = torch.where(owned[:E0], torch.arange(E0, device="cuda"), mirror[:E0].long())
    t[:E0] = t[src]
    return t


def blocks_of(schedule):
    dims = {}
    for slot in schedule.output_slots:
        dims.setdefault(int(slot.irrep_dim), set()).add(int(slot.start))
    return tuple((d, tuple(sorted(dims[d]))) for d in sorted(dims))


settings = [dict(TENSOR_CORES=False), dict(TENSOR_CORES=False, VEC_SH=True)]
if torch.cuda.get_device_capability() >= (8, 0):
    settings += [dict(TENSOR_CORES=True),
                 dict(TENSOR_CORES=True, TC_WFRAG=True, VEC_A=True, VEC_SH=True, LANE_INDICES=True),
                 dict(TENSOR_CORES=True, TC_WFRAG=True, VEC_A=True, VEC_SH=True, LANE_INDICES=True, W_GROUP=True,
                      DEGREE_PARTS=True),
                 dict(TENSOR_CORES=True, TC_WFRAG=True, VEC_A=True, VEC_SH=True, SH_LDS4=True, W_GROUP=True)]
names = ("TENSOR_CORES", "TC_WFRAG", "VEC_A", "VEC_SH", "LANE_INDICES", "W_GROUP", "DEGREE_PARTS", "SH_LDS4")
for layout in ("ir_mul", "mul_ir"):
    kernels = StreamingKernels(problem, np.float32, layout=layout)
    X = torch.randn((N, irreps_in1.dim), device="cuda", generator=g)
    Y = torch.randn((E, irreps_in2.dim), device="cuda", generator=g)
    for k, used in ((9, 9), (64, 64), (65, 64)):
        a = mirrored(torch.randn((E, k), device="cuda", generator=g))
        weight = torch.randn((problem.weight_numel, k), device="cuda", generator=g)
        if used < k:
            weight[:, -1] = 0.0
        for setting in settings:
            saved = {n: getattr(ff, n) for n in names}
            label = f"{layout}-{k}/{used}-" + "+".join(n.lower() for n, v in setting.items() if v)
            blocks = blocks_of(kernels.schedule) if layout == "ir_mul" else None
            try:
                for n in names:
                    setattr(ff, n, setting.get(n, False if n == "TENSOR_CORES" else None))
                one = ff.FusedFoldedForward(kernels, k, store_w=False, used=used, blocks=blocks)
                p1 = ff.FusedFoldedForward(kernels, k, store_w=False, used=used, blocks=blocks, owner="pass1")
                p2 = ff.FusedFoldedForward(kernels, k, store_w=False, used=used, blocks=blocks, owner="pass2")
            finally:
                for n, v in saved.items():
                    setattr(ff, n, v)
            if setting.get("W_GROUP") and one.rule.warps % 2 == 0:
                assert p1.w_group and "group_any(" in p1.source, label
            assert not p2.w_group and "mma_tf32(" not in p2.source.split('extern "C"')[1], label
            args = (a, weight, X, Y, sender, receiver, row_ptr)
            if blocks:
                base = ff.fused_folded_message_blocks(*args, one.id)
                two = ff.fused_folded_message_blocks_owner(*args, mid, mirror, p1.id, p2.id)
            else:
                base = [ff.fused_folded_message(*args, one.id)]
                two = [ff.fused_folded_message_owner(*args, mid, mirror, p1.id, p2.id)]
            for b_, t_ in zip(base, two):
                if not torch.equal(b_, t_):
                    d = (b_ - t_).abs().reshape(b_.shape[0], -1).amax(1)
                    raise AssertionError(f"{label}: not bitwise, {int((d > 0).sum())} rows differ, "
                                         f"max {float(d.max())}")
            # The passes alone: pass 1 is the one-pass kernel on the owned edges only (partner w rows from it).
            w = X.new_full((E, weight.shape[0]), float("nan"))
            out = [torch.full_like(t_, float("nan")) for t_ in base]
            p1(a, weight, X, Y, sender, receiver, row_ptr, out if blocks else out[0], w, mid, mirror)
            assert not bool(torch.isnan(torch.cat([o.flatten() for o in out])).any()), label  # every row written
            filled = (mirror[:E0] >= 0) & ~owned[:E0]
            exact = (a[:E0][filled].double() @ weight.double().t())
            got = w[:E0][filled].double()
            assert bool(torch.isfinite(got).all()), label
            assert bool(torch.isnan(w[:E0][owned[:E0]]).all()), label  # owned rows are not written
            report[label] = dict(bitwise=True, max_w_error=float(((got - exact).abs() / (a[:E0][filled].double().abs()
                                 @ weight.double().abs().t())).max()))
        # A CUDA graph of the two passes replays the same message.
        if layout == "ir_mul" and k == 64:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = ff.fused_folded_message_blocks_owner(*args, mid, mirror, p1.id, p2.id)
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(c_, t_) for c_, t_ in zip(captured, two))
            report[f"{layout}-graph"] = True
print(json.dumps(report))
"""


def test_owner_pass_is_bitwise_the_one_pass_forward():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=1500, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(report, indent=1))
    assert report["ir_mul-graph"] is True
    assert sum(1 for key in report if key.count("/")) >= 12
