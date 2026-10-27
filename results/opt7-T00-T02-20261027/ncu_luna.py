"""LunaTP's backward once between cudaProfilerStart/Stop (for ncu), on recorded operands (scratch venv)."""
import sys

import torch
from LunaEquivariance import TensorProductConv
from LunaEquivariance.lunatp.uvu_conv.interface import run_backward

from common import load, to_mul_ir

r = load(sys.argv[1])
p = r["problem"]
conv = TensorProductConv(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                         [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                          for i in p["instructions"]],
                         irrep_normalization="none", path_normalization="none", internal_weights=False,
                         shared_weights=False, dtype=torch.float32).cuda()
x = to_mul_ir(r["x"], p["irreps_in1"], p["layout"])
dout = to_mul_ir(r["dout"], p["irreps_out"], p["layout"])
graph = conv.create_graph(r["senders"].long(), r["receivers"].long(), x.shape[0])
tables = graph._tables[("uvu", conv.op.index_bits)]
needs = (True, True, True, False)
for _ in range(3):
    run_backward(conv.op, x, r["sh"], r["w"], dout, tables, None, None, needs=needs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
run_backward(conv.op, x, r["sh"], r["w"], dout, tables, None, None, needs=needs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
