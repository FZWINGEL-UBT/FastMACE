"""OPT7-T00 (FastMACE side): the vendored OEq streaming convolution on recorded operands, forward and backward.

Usage (mace_cueq_oeq, fm-f64 on PYTHONPATH): python stream_bench.py ops-*.pt ...  -> one JSON line per file and
grads-<file> with the reference gradients (mul_ir) for the LunaTP side.
"""
import json
import sys

import fastmace  # noqa: F401  (before torch: soft file locks)
import numpy as np
import torch
from openequivariance.core.e3nn_lite import TPProblem

from common import graph_time, load, rel, to_mul_ir
from fastmace.kernels.oeq_streaming import StreamingKernels
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward, streaming_conv_forward

for path in sys.argv[1:]:
    r = load(path)
    p = r["problem"]
    problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                        [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                         for i in p["instructions"]],
                        irrep_normalization="none", path_normalization="none", shared_weights=False,
                        internal_weights=False)
    kernels = StreamingKernels(problem, np.float32, layout=p["layout"], receiver_backward=False)
    x, sh, w, s, rcv, rp, dout = (r[k] for k in ("x", "sh", "w", "senders", "receivers", "row_ptr", "dout"))
    out = streaming_conv_forward(x, sh, w, s, rp, kernels.id)
    dx, dsh, dw = streaming_conv_backward(x, sh, w, s, rcv, rp, dout, kernels.id)
    row = dict(file=path, nodes=x.shape[0], edges=sh.shape[0], forward_vs_record=rel(out, r["out"]))
    row["forward_ms"] = graph_time(lambda: streaming_conv_forward(x, sh, w, s, rp, kernels.id))
    row["backward_ms"] = graph_time(lambda: streaming_conv_backward(x, sh, w, s, rcv, rp, dout, kernels.id))
    torch.save(dict(out=to_mul_ir(out, p["irreps_out"], p["layout"]).cpu(),
                    dx=to_mul_ir(dx, p["irreps_in1"], p["layout"]).cpu(), dsh=dsh.cpu(), dw=dw.cpu()),
               "grads-" + path)
    print(json.dumps(row), flush=True)
