"""One fused forward and backward under the profiler (ncu --profile-from-start off). argv: ops file, config."""
import sys

sys.path.insert(0, "/mnt/d/Margraf/fm-opt8/tools/bench")
import fastmace  # noqa: F401
import torch

import opt8_fused_conv as fc
import opt8_fused_kernels as fk

r = fc.load(__import__("pathlib").Path(sys.argv[1]))
kernels = fc.make_kernels(r["problem"])
cfg = dict(kv.split("=") for kv in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if kv)
f = fk.FusedConv(kernels.schedule, kernels.layout, r["weight"].contiguous(), cfg)
x, sh, s, rcv, rp, dout, a = (r[k] for k in ("x", "sh", "senders", "receivers", "row_ptr", "dout", "a"))
out = torch.empty((x.shape[0], kernels.output_dim), device="cuda")
dx, dsh, da = torch.zeros_like(x), torch.empty_like(sh), torch.empty_like(a)
for _ in range(2):
    f.forward(a, x, sh, s, rcv, rp, out)
    f.backward(a, x, sh, s, rcv, rp, dout, dx, dsh, da)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
f.forward(a, x, sh, s, rcv, rp, out)
f.backward(a, x, sh, s, rcv, rp, dout, dx, dsh, da)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
