"""The linear family on CUDA (plan task 1a-T06): cuEquivariance's GPU Linear, float32 and float64.

On the GPU MACE's cuEq conversion builds its Linear layers with the GPU method, not the CPU
``naive`` one, so the weight extraction is checked there as well: the registry's self-checks at
the SPEC §6.4 thresholds and the energy and forces of the rewritten tiny models.
"""

import copy

import pytest
import torch

from fastmace.rewrites import registry as reg
from fastmace.rewrites.linear import modules as lm

pytest.importorskip("cuequivariance_torch")

Z_VALUES = [11, 17]


def tiny_cueq(family, dtype):
    from mace.cli.convert_e3nn_cueq import run

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_calculator, tiny_mace_model

    batch = tiny_mace_calculator(Z_VALUES, **FAMILIES[family])._atoms_to_batch(nacl(repeat=2, rattle=0.05, seed=1))
    # "cuda:0": MACE 0.3.16 fuses the convolution only for the device string "cuda"
    model = run(tiny_mace_model(Z_VALUES, **FAMILIES[family]).to("cuda", dtype), device="cuda:0", return_model=True)
    batch = batch.to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = {key: value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in batch.to_dict().items()}
    return model, data


def energy_and_forces(model, data):
    out = model(dict(data), training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def relative(got, expected):
    return float((got - expected).abs().max()) / float(expected.abs().max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("family", ["mh1", "density"])
def test_rewritten_tiny_models_agree_on_cuda(family, dtype):
    model, data = tiny_cueq(family, dtype)
    registry = reg.default_registry()
    rewritten = copy.deepcopy(model).to("cuda")  # a deep copy leaves cuEq fx constants on the CPU
    others_off = {name: "off" for name in registry.names()
                  if name not in ("block_linear", "grouped_linear", "merged_residual")}
    resolution = registry.resolve(rewritten, others_off | {"block_linear": "on", "grouped_linear": "on"},
                                  strict=False)
    applied = registry.apply(rewritten, resolution, original=model)
    threshold = reg.threshold_for(dtype)
    assert [entry.name for entry in applied] == ["block_linear", "grouped_linear"]
    assert all(entry.result.passed and entry.result.threshold == threshold for entry in applied)
    assert any(isinstance(m, lm.BlockLinear) and m.weight_0.is_cuda for m in rewritten.modules())
    energy, forces = energy_and_forces(model, data)
    new_energy, new_forces = energy_and_forces(rewritten, data)
    assert relative(new_energy, energy) <= threshold and relative(new_forces, forces) <= threshold


def test_default_original_is_a_copy_on_the_model_device():
    model, _ = tiny_cueq("mh1", torch.float64)
    registry = reg.default_registry()
    rewritten = copy.deepcopy(model).to("cuda")
    others_off = {name: "off" for name in registry.names() if name != "block_linear"}
    applied = registry.apply(rewritten, registry.resolve(rewritten, others_off | {"block_linear": "on"},
                                                         strict=False))
    assert applied[0].result.passed and applied[0].result.modules > 0
