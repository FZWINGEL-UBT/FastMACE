"""setup-across-compositions F2: the force call with MACE's gradient step after the model call (CPU, eager).

``ForceField._split_outputs`` calls a compiled OEq ScaleShiftMACE with forces and stress off, so the compiled frame
holds no ``torch.autograd.grad`` (no graph break, no Dynamo restart), and then runs MACE's ``get_outputs`` on the
interaction energy with respect to the same input leaves. Here, on the tiny MH-1-type model in float64, that order
gives bitwise MACE's own forces, virials and stress; the GPU gate covers the compiled hybrid path.
"""

import pytest
import torch

pytest.importorskip("mace")

from test_rewrites_mh1 import tiny  # noqa: E402

from fastmace.model import energy as energy64  # noqa: E402


def _leaves(batch, stress):
    data = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    data["positions"] = data["positions"].detach().clone().requires_grad_(True)
    if stress:
        data["displacement"] = torch.zeros((1, 3, 3), dtype=data["positions"].dtype, requires_grad=True)
    return data


@pytest.mark.parametrize("stress", [False, True])
def test_the_split_call_gives_mace_s_forces_and_stress_bitwise(stress):
    from mace.modules import utils

    model, batch = tiny("mh1")
    model = model.double()
    batch = {key: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
             for key, value in batch.items()}
    assert energy64.class_name(model) == "ScaleShiftMACE"

    data = _leaves(batch, stress)
    expected = model(dict(data), training=False, compute_force=True, compute_stress=stress)

    data = _leaves(batch, stress)
    out = model(dict(data), training=False, compute_force=False, compute_stress=False, compute_displacement=stress)
    assert out["forces"] is None
    forces, virials, stress_out, _, _ = utils.get_outputs(
        energy=out["interaction_energy"], positions=data["positions"],
        displacement=data["displacement"] if stress else None, cell=data["cell"], training=False,
        compute_force=True, compute_virials=False, compute_stress=stress)

    assert torch.equal(forces, expected["forces"])
    assert torch.equal(out["energy"], expected["energy"])
    if stress:
        assert torch.equal(stress_out, expected["stress"]) and torch.equal(virials, expected["virials"])
    else:
        assert stress_out is None and expected["stress"] is None

