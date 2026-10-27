import fastmace  # noqa: F401
import numpy as np
import torch
from openequivariance.core.e3nn_lite import TPProblem

from common import irreps_blocks, load, rel
from fastmace.kernels.oeq_streaming import StreamingKernels
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward

r = load("ops-mh1-r2scan-NaCl.pt")
p = r["problem"]
for irrep_norm, path_norm in (("none", "none"), ("component", "element"), ("component", "none"), ("none", "element")):
    problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                        [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"])
                         for i in p["instructions"]],
                        irrep_normalization=irrep_norm, path_normalization=path_norm, shared_weights=False,
                        internal_weights=False)
    print(irrep_norm, path_norm, [round(i.path_weight, 4) for i in problem.instructions])
    k = StreamingKernels(problem, np.float32, layout=p["layout"])
    out = streaming_conv_forward(r["x"], r["sh"], r["w"], r["senders"], r["row_ptr"], k.id)
    ratios, start = [], 0
    for mul, dim in irreps_blocks(p["irreps_out"]):
        a = out[:, start:start + mul * dim].double()
        b = r["out"][:, start:start + mul * dim].double()
        ratios.append(round(float((a * b).sum() / (a * a).sum()), 4))
        start += mul * dim
    print("   rel", rel(out, r["out"]), "per-path record/ours", ratios)
print("recorded", [round(i["path_weight"], 4) for i in p["instructions"]])
