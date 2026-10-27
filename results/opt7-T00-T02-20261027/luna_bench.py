"""OPT7-T00 (LunaTP side): LunaEquivariance's TensorProductConv on the same recorded operands (mul_ir).

Usage (lunatp_screen venv): python luna_bench.py ops-*.pt ...  -> one JSON line per file. Checks the forward
against the recorded message and the gradients against the streaming kernels' (grads-<file>), then times the
graph build (create_graph), the forward and the backward (forward + backward minus forward, autograd).
"""
import json
import sys
import time

import torch
from LunaEquivariance import TensorProductConv

from common import graph_time, load, rel, to_mul_ir

for path in sys.argv[1:]:
    r = load(path)
    p = r["problem"]
    ref = {k: v.cuda() for k, v in torch.load("grads-" + path).items()}
    conv = TensorProductConv(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                             [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                              for i in p["instructions"]],
                             irrep_normalization="none", path_normalization="none", internal_weights=False,
                             shared_weights=False, dtype=torch.float32).cuda()
    x = to_mul_ir(r["x"], p["irreps_in1"], p["layout"]).requires_grad_()
    sh, w = r["sh"].clone().requires_grad_(), r["w"].clone().requires_grad_()
    dout = to_mul_ir(r["dout"], p["irreps_out"], p["layout"])
    src, dst = r["senders"].long(), r["receivers"].long()
    graph = conv.create_graph(src, dst, x.shape[0])  # warm-up (first-call compilation)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        graph = conv.create_graph(src, dst, x.shape[0])
    torch.cuda.synchronize()
    row = dict(file=path, create_graph_ms=1e3 * (time.perf_counter() - t) / 10)
    y = conv(x, sh, w, graph=graph)
    dx, dsh, dw = torch.autograd.grad(y, (x, sh, w), dout)
    row.update(forward_vs_streaming=rel(y, ref["out"]), dx_vs=rel(dx, ref["dx"]), dsh_vs=rel(dsh, ref["dsh"]),
               dw_vs=rel(dw, ref["dw"]))

    def forward():
        with torch.no_grad():
            conv(x, sh, w, graph=graph)

    x0, sh0, w0 = x.detach(), sh.detach(), w.detach()

    def both():
        xx, ss, ww = x0.requires_grad_(False).clone().requires_grad_(), sh0.clone().requires_grad_(),             w0.clone().requires_grad_()
        yy = conv(xx, ss, ww, graph=graph)
        torch.autograd.grad(yy, (xx, ss, ww), dout)

    row["forward_ms"] = graph_time(forward)
    try:
        row["forward_backward_ms"] = graph_time(both)
        row["clone_ms"] = graph_time(lambda: (x0.clone(), sh0.clone(), w0.clone()))
        row["backward_ms"] = row["forward_backward_ms"] - row["forward_ms"] - row["clone_ms"]
    except Exception as error:  # autograd capture may be unsupported
        row["backward_error"] = repr(error)[:300]
    print(json.dumps(row), flush=True)
