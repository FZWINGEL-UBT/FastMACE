"""The tiny-model builder's cache (tests/_support/tiny_models.py, suite-speed): a copy equals a build."""

import numpy as np
import torch

from _support import tiny_models
from _support.tiny_models import FAMILIES, tiny_mace_model


def test_a_cached_model_is_a_private_copy_with_a_builds_weights_and_rng_state():
    first = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    after_first = torch.get_rng_state()
    torch.manual_seed(123)  # a later call must not depend on the state before it, as a build does not
    torch.set_default_dtype(torch.float32)
    second = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    assert torch.get_default_dtype() == torch.float64
    assert torch.equal(torch.get_rng_state(), after_first)
    built = tiny_models._build([11, 17], polar=False, **{"interaction_first": None, **FAMILIES["mh1"]})
    assert torch.equal(torch.get_rng_state(), after_first)
    assert second is not first
    for (name, a), (_, b), (_, c) in zip(first.state_dict().items(), second.state_dict().items(),
                                         built.state_dict().items(), strict=True):
        assert torch.equal(a, b) and torch.equal(b, c), name
        if a.numel():
            assert a.data_ptr() != b.data_ptr(), name
    with torch.no_grad():
        next(second.parameters()).add_(1.0)  # a test changing its copy changes no other model
    third = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    assert torch.equal(next(third.parameters()), next(first.parameters()))


def test_other_arguments_are_other_models_and_arrays_are_never_cached():
    a = tiny_mace_model([11, 17], **FAMILIES["residual"])
    b = tiny_mace_model([11, 17, 41], **FAMILIES["residual"])
    assert sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters())
    assert tiny_models._frozen({"atomic_energies": np.zeros(2)}) is None
    assert tiny_models._frozen({"heads": ["a", "b"], "x": {"y": 1}}) is not None
