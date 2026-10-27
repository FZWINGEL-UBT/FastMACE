"""ideas D3 screen: the float64 package without recomputation of transcendental pointwise ops in backward graphs.

AOTAutograd's min-cut partitioner (``torch._functorch.partitioners``) may recompute ops from its default list in
the backward instead of saving their outputs; the list includes exp, expm1, tanh, sigmoid, log and the other
pointwise ops. In float32 that saves memory traffic; in float64 on reduced-rate GPUs these ops run on the FP64
pipe, which the profiles show 84-89 % busy (f64-T03). D3_MODE=save removes the transcendental ops from the
recomputable list, so their outputs are saved instead; D3_MODE=base changes nothing. Then it runs
``tools/bench/sb64b.py`` with the remaining arguments.

Run both sides with the Inductor FX-graph and AOTAutograd caches off (TORCHINDUCTOR_FX_GRAPH_CACHE=0,
TORCHINDUCTOR_AUTOGRAD_CACHE=0): a cached partition would bypass the change.
"""
import os
import runpy
import sys

import fastmace  # noqa: F401  (before torch)
import torch
from torch._functorch import partitioners

TRANSCENDENTAL = ("exp", "expm1", "exp2", "log", "log1p", "log2", "tanh", "sigmoid", "sin", "cos", "erf", "rsqrt",
                  "sqrt", "pow", "reciprocal", "silu")

if os.environ.get("D3_MODE") == "save":
    original = partitioners.get_default_op_list

    def _without_transcendentals():
        op_types = original()
        aten = torch.ops.aten
        banned = {getattr(aten, name) for name in TRANSCENDENTAL if hasattr(aten, name)}
        removed = [op for op in list(op_types.recomputable_ops) if op in banned]
        for op in removed:
            op_types.recomputable_ops.discard(op)
        print(f"D3 save: {len(removed)} transcendental ops not recomputable", flush=True)
        return op_types

    partitioners.get_default_op_list = _without_transcendentals

tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sb64b.py")
sys.argv = [tool] + sys.argv[1:]
runpy.run_path(tool, run_name="__main__")
