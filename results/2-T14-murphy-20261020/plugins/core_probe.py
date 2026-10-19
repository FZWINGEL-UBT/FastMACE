"""2-T14: the cached cuEq symmetric-contraction descriptor after a float32 MACE conversion."""
import sys
import fastmace  # noqa: F401
import numpy as np
import torch
sys.path[:0] = ["tests", "tests/cpu"]
from _support.tiny_models import FAMILIES, tiny_mace_model
import cuequivariance as cue
from cuequivariance.group_theory.descriptors import symmetric_contractions as sc
from mace.cli.convert_e3nn_cueq import run

calls = []
original = sc._symmetric_contraction_core


def coefficients(poly):
    return [(p.coefficients.dtype, np.array(p.coefficients, dtype=np.float64)) for _, d in poly.operations for p in d.paths]


def snapshot(args):
    return coefficients(sc._symmetric_contraction_core(*args))


m32 = tiny_mace_model([11, 17], **FAMILIES["mh1"]).float()
torch.set_default_dtype(torch.float32)
captured = []
real = cue.descriptors.symmetric_contraction
cue.descriptors.symmetric_contraction = lambda *a: (captured.append(a), real(*a))[1]
import mace.cli.convert_e3nn_cueq as conv  # noqa: E402
run(m32, device="cpu", return_model=True)
torch.set_default_dtype(torch.float64)
args = (captured[0][0], captured[0][1], tuple(captured[0][2]))
cached = snapshot(args)
print("args", args, "dtypes in cache:", sorted({str(d) for d, _ in cached}))
sc._symmetric_contraction_core.cache_clear()
fresh = snapshot(args)
print("max diff cached vs recomputed:", max(np.abs(a - b).max() for (_, a), (_, b) in zip(cached, fresh)))
# the same in a float64 default dtype from the start of a recomputation: does the dtype of the basis matter?
basis = cue.reduced_symmetric_tensor_product_basis(args[0].set_mul(1), 2, keep_ir=args[1].set_mul(1), layout=cue.ir_mul)
print("basis dtype", np.asarray(basis.array).dtype)
