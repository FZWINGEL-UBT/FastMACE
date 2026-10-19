"""2-T14: which steps of a production sequence fill cuequivariance's group-theory caches before the float32 conversion."""
import os, sys, tempfile, warnings
import fastmace  # noqa: F401
import torch
sys.path[:0] = ["tests"]
warnings.simplefilter("ignore")


def sizes():
    total = 0
    module = sys.modules.get("cuequivariance.group_theory.irreps_array.reduced_tensor_product")
    if module is not None:
        total += len(module._cache_reduce_basis_product)
    for name, m in list(sys.modules.items()):
        if m is not None and name.startswith("cuequivariance.group_theory"):
            for v in list(vars(m).values()):
                if callable(getattr(v, "cache_info", None)):
                    total += v.cache_info().currsize
    return total


from _support.tiny_models import FAMILIES, tiny_mace_model
path = os.path.join(tempfile.mkdtemp(dir=os.environ["TMPDIR"]), "m.model")
torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
from fastmace.model import loading
sys.modules["cuequivariance.group_theory.irreps_array.reduced_tensor_product"]._cache_reduce_basis_product.clear()
for n, m in list(sys.modules.items()):
    if m is not None and n.startswith("cuequivariance.group_theory"):
        for v in list(vars(m).values()):
            if callable(getattr(v, "cache_clear", None)):
                v.cache_clear()
print("after clearing:", sizes())
loaded = loading.load_pristine(path, trusted=True)
print("after load_pristine:", sizes())
import fastmace as fm
atoms = __import__("_support.structures", fromlist=["nacl"]).nacl(repeat=1, rattle=0.05, seed=3)
ff = fm.ForceField(loaded, atoms)
print("after ForceField:", sizes())
from fastmace.planner import minimal
plan = minimal.default_planner(ff)
print("after default_planner:", sizes())
