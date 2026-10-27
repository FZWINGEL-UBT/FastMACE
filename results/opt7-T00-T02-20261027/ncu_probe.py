"""OPT7-T02: one forward and one backward of the streaming convolution between cudaProfilerStart/Stop (for ncu).

Usage: ncu --profile-from-start off ... python ncu_probe.py ops-X.pt
Also prints the compulsory bytes of every operand (each tensor read or written once).
"""
import json
import sys

import fastmace  # noqa: F401
import numpy as np
import torch
from openequivariance.core.e3nn_lite import TPProblem

from common import load
from fastmace.kernels.oeq_streaming import StreamingKernels
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward, streaming_conv_forward

path = sys.argv[1]
r = load(path)
p = r["problem"]
problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                    [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                     for i in p["instructions"]],
                    irrep_normalization="none", path_normalization="none", shared_weights=False, internal_weights=False)
k = StreamingKernels(problem, np.float32, layout=p["layout"])
x, sh, w, s, rcv, rp, dout = (r[key] for key in ("x", "sh", "w", "senders", "receivers", "row_ptr", "dout"))
for _ in range(3):
    streaming_conv_forward(x, sh, w, s, rp, k.id)
    streaming_conv_backward(x, sh, w, s, rcv, rp, dout, k.id)
torch.cuda.synchronize()
size = {name: t.numel() * t.element_size() for name, t in dict(x=x, sh=sh, w=w, senders=s, receivers=rcv,
                                                                  row_ptr=rp, dout=dout).items()}
out_bytes = x.shape[0] * k.output_dim * 4
print(json.dumps(dict(file=path, compulsory_bytes=size, out_bytes=out_bytes)), flush=True)
torch.cuda.cudart().cudaProfilerStart()
streaming_conv_forward(x, sh, w, s, rp, k.id)
streaming_conv_backward(x, sh, w, s, rcv, rp, dout, k.id)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
