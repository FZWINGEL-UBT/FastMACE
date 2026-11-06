"""``polar.SHARED_GEOMETRY`` (A/B constant, off): one reciprocal geometry per call for one graph.

POLAR-1's long-range part builds the [k-vectors, atoms] cosines and sines twice per call (the feature
projection's ``precompute_geometry`` and the Coulomb energy) and multiplies each by a [k, atoms] graph mask
that is the k-vectors' own column for one structure (A100 profile, results/off-polar-alex-20261106: about 1.9 ms
of POLAR-1-S's 11.3 ms NbCl5 step in these matrices). With the constant on, one graph masks by that column and the Coulomb energy reuses
the call's cosines and sines when it gets the same k-vectors and positions. Rounding only through the backward
(the two uses' gradients meet before the cosine's derivative instead of after).
"""

import copy

import pytest
import torch

pytest.importorskip("mace")

from _support.structures import nacl  # noqa: E402
from _support.tiny_models import polar_available, tiny_family_calculator  # noqa: E402
from fastmace.config import PolarConfig  # noqa: E402
from fastmace.longrange import polar  # noqa: E402

torch.serialization.add_safe_globals([slice])

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _evaluate(calc, model, atoms, stress):
    out = model(calc._atoms_to_batch(atoms).to_dict(), compute_stress=stress, training=False)
    return {key: out[key].detach() for key in ("energy", "forces", "stress") if out.get(key) is not None}


def test_one_geometry_helper_counts_its_builds(monkeypatch):
    """Two requests with the same k-vectors and positions in one call: one build with the constant on."""
    k = torch.randn(7, 3, dtype=torch.float64)
    positions = torch.randn(5, 3, dtype=torch.float64)
    batch = torch.zeros(5, dtype=torch.long)
    k_batch = torch.tensor([0, 0, 1, 0, 0, 1, 0])  # the fixed k-set marks k-vectors outside the cutoff
    k_norm2 = (k * k).sum(-1)
    k0 = torch.zeros(7, dtype=torch.float64)
    volume = torch.ones(1, dtype=torch.float64)
    call = polar._Call(None, {"ptr": torch.tensor([0, 5])})
    plain = polar._pbc_geometry(k, k_norm2, k_batch, k0, positions, batch, volume)
    monkeypatch.setattr(polar, "SHARED_GEOMETRY", True)
    first = polar._pbc_geometry(k, k_norm2, k_batch, k0, positions, batch, volume, call)
    second = polar._pbc_geometry(k, k_norm2, k_batch, k0, positions, batch, volume, call)
    for a, b, c in zip(plain, first, second):
        assert torch.equal(a, b) and torch.equal(b, c)
    assert second[0] is first[0] and second[1] is first[1]
    other = polar._pbc_geometry(k, k_norm2, k_batch, k0, positions.clone(), batch, volume, call)
    assert other[0] is not first[0]  # other positions: built again


def test_several_graphs_keep_the_mask(monkeypatch):
    monkeypatch.setattr(polar, "SHARED_GEOMETRY", True)
    k = torch.randn(4, 3, dtype=torch.float64)
    positions = torch.randn(6, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    k_batch = torch.tensor([0, 0, 1, 1])
    call = polar._Call(None, {"ptr": torch.tensor([0, 3, 6])})
    got = polar._pbc_geometry(k, (k * k).sum(-1), k_batch, torch.zeros(4, dtype=torch.float64), positions, batch,
                              torch.ones(2, dtype=torch.float64), call)
    expected = polar._pbc_geometry(k, (k * k).sum(-1), k_batch, torch.zeros(4, dtype=torch.float64), positions,
                                   batch, torch.ones(2, dtype=torch.float64))
    for a, b in zip(got, expected):
        assert torch.equal(a, b)
    assert getattr(call, "pbc_geometry", None) is None


@needs_polar
@pytest.mark.parametrize("stress", [False, True])
def test_enabled_path_with_shared_geometry(monkeypatch, stress):
    calc = tiny_family_calculator("polar", [11, 17])
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    model = copy.deepcopy(calc.models[0])
    polar.enable(model, atoms.cell.array, PolarConfig(), variable_cell=stress)
    expected = _evaluate(calc, model, atoms, stress)
    monkeypatch.setattr(polar, "SHARED_GEOMETRY", True)
    got = _evaluate(calc, model, atoms, stress)
    assert set(got) == set(expected)
    torch.testing.assert_close(got["energy"], expected["energy"], rtol=0, atol=0)  # the forward is the same
    for key in got:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-11, atol=1e-12)
