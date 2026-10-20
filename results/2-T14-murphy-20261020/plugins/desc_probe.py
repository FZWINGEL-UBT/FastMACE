"""2-T14: the polar symmetric-contraction descriptor, plain vs after a float32 conversion (optionally with caches cleared)."""
import functools
import gc
import pickle
import sys
import fastmace  # noqa: F401
import numpy as np
import torch
sys.path[:0] = ["tests", "tests/cpu"]
from _support.tiny_models import FAMILIES, tiny_mace_model
import cuequivariance as cue
from mace.tools.cg import O3_e3nn
from mace.cli.convert_e3nn_cueq import run

mode, path = sys.argv[1], sys.argv[2]
if mode != "plain":
    m32 = tiny_mace_model([11, 17], **FAMILIES["mh1"]).float()
    torch.set_default_dtype(torch.float32)
    run(m32, device="cpu", return_model=True)
    torch.set_default_dtype(torch.float64)
if mode == "after32-dict":
    from cuequivariance.group_theory.irreps_array import reduced_tensor_product as rtp
    print("cleared", len(rtp._cache_reduce_basis_product), "entries of _cache_reduce_basis_product")
    rtp._cache_reduce_basis_product.clear()
if mode in ("after32-clear", "after32-dict"):
    n = 0
    for obj in gc.get_objects():
        if isinstance(obj, functools._lru_cache_wrapper):
            obj.cache_clear()
            n += 1
    print("cleared", n, "functools caches (all modules)")
torch.set_default_dtype(torch.float64)
irreps_in = cue.Irreps(O3_e3nn, "8x0e+8x1o+8x2e")
irreps_out = cue.Irreps(O3_e3nn, "8x0e+8x1o")
poly = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, (1, 2))
values = [np.array(p.coefficients, dtype=np.float64) for _, d in poly.polynomial.operations for p in d.paths]
basis = np.asarray(cue.reduced_symmetric_tensor_product_basis(cue.Irreps(O3_e3nn, "0e+1o+2e"), 2,
                                                              keep_ir=cue.Irreps(O3_e3nn, "0e+1o"), layout=cue.ir_mul).array)
cg = O3_e3nn.clebsch_gordan(O3_e3nn(l=1, p=-1), O3_e3nn(l=1, p=-1), O3_e3nn(l=2, p=1))
if mode == "plain":
    pickle.dump((values, basis, cg), open(path, "wb"))
else:
    ref_values, ref_basis, ref_cg = pickle.load(open(path, "rb"))
    print(mode, "descriptor max diff", max(np.abs(a - b).max() for a, b in zip(values, ref_values)),
          "| basis max diff", np.abs(basis - ref_basis).max(), "| CG(1o,1o,2e) dtype", cg.dtype,
          "max diff", np.abs(cg - ref_cg).max())
