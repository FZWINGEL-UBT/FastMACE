"""Why is the block wprime kernel slower on the RTX 5060 Ti? Times the dense and block kernels alone, on tensors
captured from a real step, with the block gradients placed in different ways. Alternating rounds, one process."""
import hashlib
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = ["/mnt/d/Margraf/FastMACE", "/mnt/d/Margraf/FastMACE/tools"]
import fastmace  # noqa: F401
import torch  # noqa: E402

import ab_step_time  # noqa: E402
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig  # noqa: E402
from fastmace.kernels.oeq_streaming import wprime_backward as wb  # noqa: E402
from fastmace.model import loading  # noqa: E402
from fastmace.model.forcefield import ForceField  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

wb.TABLE_W = True
wb.BLOCK_MESSAGE = True
captured = {}
original = wb.WPrimeBackward.__call__


def capture(self, x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins,
            r_lo, bin_scale, r_max, dr):
    if self.blocks and id(self) not in captured:
        captured[id(self)] = (self, [t.clone() for t in (x, sh, weights, senders, receivers)],
                              [d.clone() for d in dout],
                              [t.clone() for t in (dx, dsh, r, pair, gprime, starts, scales, bins)],
                              (r_lo, bin_scale, r_max), dr.clone())
    return original(self, x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins,
                    r_lo, bin_scale, r_max, dr)


wb.WPrimeBackward.__call__ = capture
ck = Path("/home/fzwin/models/mace-omat-0-medium.model")
loaded = loading.load_pristine("omat0-medium", checkpoint=str(ck), sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
name, base = ab_step_time.read_structure("results/loup-a100-20261103/SrTiO3.cif@10x10x10", None)
field = ForceField(loaded, base, precision="float32", rewrites="auto")
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=NeighbourConfig(skin=0.5))
integ = IntegratorConfig(temperature_K=300.0, timestep_fs=1.0)
plan = planner.plan(field, base, integ, Workload(max_temperature_K=300.0), planner.hardware())
engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
engine.prepare()
for _ in range(3):
    engine.step()
engine.flush()
torch.cuda.synchronize()
wb.WPrimeBackward.__call__ = original
print(torch.cuda.get_device_name(0), "captured kernels", len(captured), flush=True)

owners = {}
import gc  # noqa: E402

for o in gc.get_objects():
    try:
        if isinstance(o, wb.WPrime) and o.block is not None:
            owners[id(o.block[3])] = o
    except Exception:
        pass


def placed(groups, gap, align):
    """The groups in one allocation: each start at a multiple of ``align`` bytes, then ``gap(i)`` bytes on."""
    start, spans = 0, []
    for i, g in enumerate(groups):
        start = -(-start // align) * align + gap(i)
        spans.append(start)
        start += g.numel() * 4
    store = torch.empty(start // 4 + align // 4 + 4096, dtype=groups[0].dtype, device=groups[0].device)
    shift = (-store.data_ptr() // 4) % (align // 4) if align <= 2 << 20 else 0
    views = []
    for g, s in zip(groups, spans):
        v = store[shift + s // 4:shift + s // 4 + g.numel()].view(g.shape)
        v.copy_(g)
        views.append(v)
    return store, views


for key, (kernel, head, groups, tail, floats, dr) in captured.items():
    owner = owners[key]
    k = owner.block
    dense_kernel = wb.WPrimeBackward(owner.kernels, owner.terms, groups[0].device, vector=owner.vector, table_w=True)
    dense = wb.dense_message(groups, kernel.blocks, kernel.channels, kernel.output_dim)
    dx, dsh, r, pair, gprime, starts, scales, bins = tail
    print("edges", head[1].shape[0], "x", head[0].shape, "dense", dense.shape, "groups",
          [tuple(g.shape) for g in groups], "addrs MB", [round(g.data_ptr() / 2**20, 3) for g in groups], flush=True)
    sides = {"dense": (dense_kernel, dense, None), "block": (kernel, groups, None)}
    for label, gap, align in (("joined", lambda i: 0, 4), ("align2M", lambda i: 0, 2 << 20),
                              ("align2M+pad", lambda i: i * 1536 + i * 64, 2 << 20),
                              ("align4K+pad", lambda i: i * 1536 + i * 64, 4096)):
        store, views = placed(groups, gap, align)
        print(label, "addr mod 64K", [g.data_ptr() % 65536 for g in views], flush=True)
        sides[label] = (kernel, views, store)
    saved = {n: ([v.clone() for v in s[1]] if isinstance(s[1], list) else s[1].clone()) for n, s in sides.items()}
    times = {(n, m): [] for n in sides for m in ("reread", "rewritten")}
    for round_ in range(45):
        for mode in ("reread", "rewritten"):
            for n, (ker, msg, _) in sides.items():
                if mode == "rewritten":  # as after the GEMM: the gradient was just written on the GPU
                    if isinstance(msg, list):
                        for v, s in zip(msg, saved[n]):
                            v.copy_(s)
                    else:
                        msg.copy_(saved[n])
                dx.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ker(*head, msg, dx, dsh, r, pair, gprime, starts, scales, bins, *floats, dr)
                torch.cuda.synchronize()
                if round_ >= 5:
                    times[n, mode].append((time.perf_counter() - t0) * 1e3)
    ref = {m: statistics.median(times["dense", m]) for m in ("reread", "rewritten")}
    for (n, m), v in times.items():
        med = statistics.median(v)
        print(f"{n:14s} {m:10s} {med:7.3f} ms  /dense {med / ref[m]:.3f}  spread {min(v):.3f}-{max(v):.3f}", flush=True)
print("done", flush=True)
