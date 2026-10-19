"""2-T14: record what MACE's cuEq conversion helpers return for the polar model, plain vs after a float32 conversion."""
import pickle
import sys
import fastmace  # noqa: F401
import numpy as np
import torch
sys.path[:0] = ["tests", "tests/cpu"]
import test_forces as tf
from _support.tiny_models import FAMILIES, tiny_mace_model
import cuequivariance as cue
import mace.cli.convert_e3nn_cueq as conv
import mace.tools.cg as cg
import mace.tools.cg_cueq_tools as cgt
from mace.cli.convert_e3nn_cueq import run

if sys.argv[2] == "after32":
    dtype = torch.get_default_dtype()
    m32 = tiny_mace_model([11, 17], **FAMILIES["mh1"]).float()
    torch.set_default_dtype(torch.float32)
    run(m32, device="cpu", return_model=True)
    torch.set_default_dtype(dtype)

record = {}


def wrap(module, name, convert):
    original = getattr(module, name)

    def wrapper(*a, **k):
        out = original(*a, **k)
        record.setdefault(f"{module.__name__}.{name}", []).append(convert(out))
        return out
    setattr(module, name, wrapper)


def arrays(x):
    if isinstance(x, torch.Tensor):
        return x.detach().double().numpy().copy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float64).copy()
    if isinstance(x, (list, tuple)):
        return [arrays(v) for v in x]
    if hasattr(x, "array"):
        return np.asarray(x.array, dtype=np.float64).copy()
    if hasattr(x, "polynomial"):
        return [np.asarray(p.coefficients, dtype=np.float64).copy() for _, d in x.polynomial.operations for p in d.paths]
    return None


wrap(cgt, "symmetric_contraction_proj", arrays)
wrap(conv, "symmetric_contraction_proj", arrays)
wrap(cue, "reduced_symmetric_tensor_product_basis", arrays)
wrap(cg, "U_matrix_real", arrays)
wrap(cue.descriptors, "symmetric_contraction", arrays)
calc, cueq = tf.polar_calculators()
pickle.dump(record, open(sys.argv[1], "wb"))
print({k: len(v) for k, v in record.items()})
