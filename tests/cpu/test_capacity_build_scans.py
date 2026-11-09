"""The build's whole-structure scans without Python loops or sorts over the atoms (149M atoms on an A100: a set
over ``atoms.numbers`` and a ``torch.unique(dim=0)`` per checked module took most of a 2,000 s build)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_distinct_numbers_are_the_sorted_set():
    from fastmace.model.forcefield import distinct_numbers

    numbers = np.array([38, 8, 22, 8, 8, 38, 1])
    assert distinct_numbers(numbers) == (1, 8, 22, 38)
    assert all(type(z) is int for z in distinct_numbers(numbers))
    assert distinct_numbers([]) == ()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
def test_distinct_rows_of_a_one_hot_are_torch_unique(dtype):
    from fastmace.rewrites.species import _distinct_rows

    generator = torch.Generator().manual_seed(0)
    columns = torch.tensor([0, 3, 7, 3, 9])[torch.randint(0, 5, (500,), generator=generator)]
    attrs = torch.nn.functional.one_hot(columns, 12).to(dtype)
    got, expected = _distinct_rows(attrs), torch.unique(attrs, dim=0)
    assert got.dtype == expected.dtype and torch.equal(got, expected)


def test_distinct_rows_of_other_rows_fall_back_to_torch_unique():
    from fastmace.rewrites.species import _distinct_rows

    attrs = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    assert torch.equal(_distinct_rows(attrs), torch.unique(attrs, dim=0))
    two = torch.tensor([[2, 0], [0, 1]], dtype=torch.uint8)
    assert torch.equal(_distinct_rows(two), torch.unique(two, dim=0))
