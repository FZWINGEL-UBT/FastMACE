"""``polar.PHASE_RECURRENCE`` (A/B constant, off): the dense [k, atoms] cosines and sines from per-axis phase tables.

The fixed k-set is integer combinations of the reciprocal basis, k = n1 b1 + n2 b2 + n3 b3, so exp(i k . r) is a
product of three per-atom phases exp(i n_j b_j . r). The tables need (2 M + 1) x 3 angles per atom instead of one
per k-vector (the NbCl5 profile: the cos/sin kernel 0.57 ms of POLAR-1-M's 41.5 ms step on the RTX 5060 Ti); the
backward is the dense formula from the kept cosines and sines. Float64 against the dense path.
"""

import copy

import numpy as np
import pytest
import torch

from _support.tiny_models import polar_available
from fastmace.longrange import polar

torch.serialization.add_safe_globals([slice])

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")


def _lattice(bounds=(3, 4, 5), atoms=11, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cell = np.diag([7.0, 8.5, 9.25]) + 0.6 * np.random.default_rng(seed).standard_normal((3, 3))
    basis = torch.tensor(2 * np.pi * np.linalg.inv(cell.T))
    integer = torch.tensor(polar._coefficients(bounds)).to(torch.int64)
    k = (integer.to(torch.float64) @ basis).requires_grad_()
    r = (torch.rand((atoms, 3), generator=generator, dtype=torch.float64) @ torch.tensor(cell)).requires_grad_()
    return k, r, basis, integer, max(bounds)


def test_the_constant_is_off():
    assert polar.PHASE_RECURRENCE is False


def test_tables_give_the_dense_phases():
    k, r, basis, integer, span = _lattice()
    cos, sin = polar.recurrence_phases(r.detach(), basis, integer, span)
    theta = k.detach() @ r.detach().t()
    torch.testing.assert_close(cos, torch.cos(theta), rtol=0, atol=1e-12)
    torch.testing.assert_close(sin, torch.sin(theta), rtol=0, atol=1e-12)


def test_the_backward_is_the_dense_one():
    k, r, basis, integer, span = _lattice(seed=1)
    generator = torch.Generator().manual_seed(2)
    wc, ws = (torch.randn((k.shape[0], r.shape[0]), generator=generator, dtype=torch.float64) for _ in range(2))
    cos, sin = polar.phase_factors().apply(k, r, basis, integer, span)
    got = torch.autograd.grad((wc * cos + ws * sin).sum(), (k, r))
    theta = k @ r.t()
    expected = torch.autograd.grad((wc * torch.cos(theta) + ws * torch.sin(theta)).sum(), (k, r))
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e, rtol=1e-10, atol=1e-10)
    only_cos = torch.autograd.grad(polar.phase_factors().apply(k, r, basis, integer, span)[0].sum(), r)[0]
    torch.testing.assert_close(only_cos, torch.autograd.grad(torch.cos(k @ r.t()).sum(), r)[0], rtol=1e-10,
                               atol=1e-10)


def test_other_k_vectors_stay_dense(monkeypatch):
    """The tables are used only for the k-vectors the call built from its lattice."""
    k, r, basis, integer, span = _lattice()
    monkeypatch.setattr(polar, "PHASE_RECURRENCE", True)
    call = polar._Call(None, {"ptr": torch.tensor([0, r.shape[0]])})
    call.k_lattice = (k, basis)
    calls = []
    monkeypatch.setattr(polar, "recurrence_phases", lambda *a: calls.append(1) or (torch.zeros(1), torch.zeros(1)))
    polar._phases(k.detach().clone(), r, call)  # another tensor
    polar._phases(k, r, call)  # no state: nothing to index
    assert calls == []


@needs_polar
@pytest.mark.parametrize("stress", [False, True])
def test_enabled_path_with_phase_tables(monkeypatch, stress):
    from _support.structures import nacl
    from _support.tiny_models import tiny_family_calculator
    from fastmace.config import PolarConfig

    calc = tiny_family_calculator("polar", [11, 17])
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    model = copy.deepcopy(calc.models[0])
    polar.enable(model, atoms.cell.array, PolarConfig(), variable_cell=stress)

    def evaluate():
        out = model(calc._atoms_to_batch(atoms).to_dict(), compute_stress=stress, training=False)
        return {key: out[key].detach() for key in ("energy", "forces", "stress") if out.get(key) is not None}

    expected = evaluate()
    monkeypatch.setattr(polar, "PHASE_RECURRENCE", True)
    used = []
    tables = polar.recurrence_phases
    monkeypatch.setattr(polar, "recurrence_phases", lambda *a: used.append(1) or tables(*a))
    got = evaluate()
    assert used, "the enabled path did not take the phase tables"
    assert set(got) == set(expected)
    for key in got:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-10, atol=1e-11)
