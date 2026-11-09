"""The lab's frozen species snapshot must reject stale row maps before replay."""

import pytest
import torch

from tools.dmqe_species_lab import FrozenSpeciesPlan, build_plan


def onehot():
    return torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 2, 0, 1]), 3).float()


@pytest.mark.parametrize("counting", [False, True])
def test_frozen_plan_is_exact_and_validates_equal_replacement(counting):
    attrs = onehot()
    frozen = FrozenSpeciesPlan.bind(attrs, (1, 3), counting, "model-1")
    frozen.validate(attrs.clone(), "model-1")
    expected = build_plan(attrs, (1, 3), counting)
    for a, b in zip(frozen.tensors, expected):
        assert torch.equal(a, b)


@pytest.mark.parametrize("change", ["same-count-swap", "composition", "atom-count", "dtype", "model"])
def test_frozen_plan_rejects_stale_state(change):
    attrs = onehot()
    frozen = FrozenSpeciesPlan.bind(attrs, (1, 3), True, "model-1")
    model = "model-1"
    if change == "same-count-swap":
        attrs[[0, 1]] = attrs[[1, 0]]
    elif change == "composition":
        attrs[0] = attrs[1]
    elif change == "atom-count":
        attrs = attrs[:-1]
    elif change == "dtype":
        attrs = attrs.double()
    elif change == "model":
        model = "model-2"
    with pytest.raises(ValueError, match="stale"):
        frozen.validate(attrs, model)
    refreshed = FrozenSpeciesPlan.bind(attrs, (1, 3), True, model)
    refreshed.validate(attrs, model)


def test_same_count_swap_changes_plan_even_though_counts_match():
    attrs = onehot()
    old = build_plan(attrs, (1, 3), True)
    attrs[[0, 1]] = attrs[[1, 0]]
    new = build_plan(attrs, (1, 3), True)
    assert torch.equal(old[2], new[2])
    assert not torch.equal(old[1], new[1])
    assert any(not torch.equal(a, b) for a, b in zip(old[3:], new[3:]))
