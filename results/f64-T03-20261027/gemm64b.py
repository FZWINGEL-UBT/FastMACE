"""Float64 follow-up (f64-T03): the package's remaining float64 matrix products, native vs cuBLAS emulation.

One process per setting (the emulation is read when cuBLAS creates its handle): python gemm64b.py LABEL.
1. The chunked backward's input gradient da = dw_chunk @ W, dw_chunk [n, 1280], W [1280, 65]: chunk sizes
   n = 768 (today's float64 L2 rule), 1536, 3072, 6144, 12288, 24576 and one product over all 56,320 edges (NbCl5);
   time per 56,320 edges (all chunks), one stream.
2. The folded forward w = a @ W^T, a [56320, 65], W [1280, 65]: as is; transposed (W @ a^T); split into 4 or 10
   column blocks of W; the inner dimension padded with zeros to 128.
Errors on 64 sampled rows against long double, scaled by |A||B|. One JSON line.
"""
import json
import os
import sys
import time

import numpy as np
import torch

E, Q, P = 56320, 1280, 65
label = sys.argv[1]
gen = torch.Generator(device="cuda").manual_seed(5)


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / reps


def err(c, a, b, rows):
    a64, b64 = a.cpu().numpy().astype(np.longdouble), b.cpu().numpy().astype(np.longdouble)
    ref = a64[rows] @ b64
    scale = np.abs(a64[rows]) @ np.abs(b64)
    return float((np.abs(c.cpu().numpy().astype(np.longdouble)[rows] - ref) / scale).max())


out = {"label": label, "env": {k: v for k, v in os.environ.items() if k.startswith("CUBLAS")},
       "gpu": torch.cuda.get_device_name()}
dw = torch.randn(E, Q, dtype=torch.float64, device="cuda", generator=gen)
W = torch.randn(Q, P, dtype=torch.float64, device="cuda", generator=gen) / Q ** 0.5
rows = torch.randperm(E, device="cuda", generator=gen)[:64].cpu().numpy()
da = torch.empty(E, P, dtype=torch.float64, device="cuda")
chunks = {}
for n in (768, 1536, 3072, 6144, 12288, 24576, E):
    def run(n=n):
        for s in range(0, E, n):
            torch.mm(dw[s:s + n], W, out=da[s:s + n])
    ms = timed(run)
    chunks[n] = {"ms_all_edges": ms, "err": err(da, dw, W, rows)}
out["da_chunks"] = chunks

a = torch.randn(E, P, dtype=torch.float64, device="cuda", generator=gen)
a[:, -1] = 1.0
Wt = torch.randn(Q, P, dtype=torch.float64, device="cuda", generator=gen) / P ** 0.5
B = Wt.t().contiguous()
variants = {}
w = torch.empty(E, Q, dtype=torch.float64, device="cuda")
variants["as_is"] = (lambda: torch.mm(a, B, out=w), lambda: w)
wT = torch.empty(Q, E, dtype=torch.float64, device="cuda")
aT = a.t().contiguous()
variants["transposed"] = (lambda: torch.mm(Wt, aT, out=wT), lambda: wT.t())
for parts in (4, 10):
    step = Q // parts

    def split(step=step):
        for s in range(0, Q, step):
            torch.mm(a, B[:, s:s + step], out=w[:, s:s + step])
    variants[f"columns_{parts}"] = (split, lambda: w)
a128 = torch.zeros(E, 128, dtype=torch.float64, device="cuda")
a128[:, :P] = a
B128 = torch.zeros(128, Q, dtype=torch.float64, device="cuda")
B128[:P] = B
variants["k_padded_128"] = (lambda: torch.mm(a128, B128, out=w), lambda: w)
fwd = {}
for name, (fn, result) in variants.items():
    ms = timed(fn)
    fwd[name] = {"ms": ms, "err": err(result().contiguous(), a, B, rows)}
out["folded_forward"] = fwd
print(json.dumps(out), flush=True)
