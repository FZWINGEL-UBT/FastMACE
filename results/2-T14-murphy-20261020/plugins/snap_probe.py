"""2-T14: module- and class-level containers that a float32 vs float64 conversion leaves different."""
import sys
import fastmace  # noqa: F401
import torch
sys.path[:0] = ["tests", "tests/cpu"]
from _support.tiny_models import FAMILIES, tiny_mace_model
from mace.cli.convert_e3nn_cueq import run
import cuequivariance_torch, e3nn, mace  # noqa: E401,F401

PREFIXES = ("mace", "cuequivariance", "cuequivariance_torch", "e3nn", "torch", "opt_einsum", "fastmace")


def snap():
    out = {}
    for name, module in list(sys.modules.items()):
        if module is None or name.split(".")[0] not in PREFIXES:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for attr, value in items:
            holders = [(attr, value)]
            if isinstance(value, type) and value.__module__ == name:
                holders += [(f"{attr}.{a}", v) for a, v in vars(value).items()]
            for key, v in holders:
                if isinstance(v, (dict, list, set)):
                    try:
                        out[f"{name}:{key}"] = len(v)
                    except Exception:  # noqa: BLE001
                        pass
                elif isinstance(v, (bool, int, float, str, torch.dtype)):
                    out[f"{name}:{key}"] = v
    return out


model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
dtype = torch.get_default_dtype()
run(model, device="cpu", return_model=True)  # a float64 conversion first: its caches are the baseline
before = snap()
m32 = tiny_mace_model([11, 17], **FAMILIES["mh1"]).float()
torch.set_default_dtype(torch.float32)
run(m32, device="cpu", return_model=True)
torch.set_default_dtype(dtype)
after = snap()
for key in sorted(set(before) | set(after)):
    if before.get(key) != after.get(key):
        print("DIFF", key, before.get(key), "->", after.get(key))
