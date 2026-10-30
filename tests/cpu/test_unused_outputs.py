"""Model outputs no FastMACE evaluation reads are not returned by the compiled model (capacity screen A).

``compile_model`` replaces its private copy's ``forward`` with one that leaves out ``UNUSED_OUTPUTS`` (MACE's
concatenated ``node_feats``), so Inductor does not materialise them and they do not stay alive through the force
backward. The outputs that are returned are the model's own tensors, unchanged.
"""

import torch

from fastmace.model import compile as compiling


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, data, training=False, compute_force=True):
        feats = data["x"] * self.weight
        return {"energy": feats.sum(), "node_feats": torch.cat([feats, feats]), "forces": None,
                "training": training, "compute_force": compute_force}


def test_unused_outputs_are_left_out():
    assert compiling.UNUSED_OUTPUTS == ("node_feats",)
    model = _Model()
    data = {"x": torch.tensor([1.0, 4.0])}
    expected = model(data, training=True, compute_force=False)
    lean = compiling.drop_unused_outputs(model)
    assert lean is model
    out = model(data, training=True, compute_force=False)
    assert "node_feats" not in out
    assert set(out) == set(expected) - {"node_feats"}
    assert torch.equal(out["energy"], expected["energy"])
    assert out["forces"] is None and out["training"] is True and out["compute_force"] is False
    assert type(model).forward is _Model.forward  # the class is not touched: only this private copy


def test_no_names_keep_the_forward():
    model = _Model()
    forward = model.forward
    compiling.drop_unused_outputs(model, ())
    assert "forward" not in vars(model) and model.forward == forward


def test_a_forward_without_a_dict_is_returned_unchanged():
    class Plain(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = compiling.drop_unused_outputs(Plain())
    assert torch.equal(model(torch.tensor([1.0])), torch.tensor([2.0]))
