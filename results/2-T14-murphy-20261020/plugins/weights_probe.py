"""2-T14: are the polar cuEq model's converted weights different after a float32 conversion?"""
import sys
import fastmace  # noqa: F401
import torch
sys.path[:0] = ["tests", "tests/cpu"]
import test_forces as tf
from _support.tiny_models import FAMILIES, tiny_mace_model
from mace.cli.convert_e3nn_cueq import run

if sys.argv[2] == "after32":
    dtype = torch.get_default_dtype()
    m32 = tiny_mace_model([11, 17], **FAMILIES["mh1"]).float()
    torch.set_default_dtype(torch.float32)
    run(m32, device="cpu", return_model=True)
    torch.set_default_dtype(dtype)
calc, cueq = tf.polar_calculators()
torch.save({k: v.clone() for k, v in cueq.models[0].state_dict().items()}, sys.argv[1])
