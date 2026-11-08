"""``polar.SPIN_BATCHED`` (default on since off-polar-alex): both spin channels' field features in one call.

PolarMACE's charge recursion calls the field-feature descriptor twice per step, once for the alpha and once for
the beta spin density, so each [k-vectors, atoms] product with the cosines and sines runs twice with a handful of
columns. With the constant on, the recursion stacks both densities as two channels and runs graph_longrange's
multichannel k-space sums once (twice the columns, half the products). Pure removal of launches; rounding only
where a wider product blocks its sums differently.
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


def test_the_edit_applies_once_to_the_qualified_forward():
    if not polar_available():
        pytest.skip("PolarMACE needs graph_longrange")
    import inspect

    import mace.modules.extensions as extensions

    source = inspect.getsource(polar._original("mace.modules.extensions:PolarMACE.forward"))
    for old, _ in polar.SPIN_BATCH_EDITS:
        assert source.count(old) == 1
    edited = polar.spin_batched_source(source)
    assert "_fastmace_spin_field_features(" in edited
    assert "electric_potential_descriptor.forward_dynamic" not in edited
    assert extensions  # imported for the qualified source


def test_other_sources_are_refused():
    with pytest.raises(polar.PolarUnavailableError):
        polar.spin_batched_source("def forward(self, data):\n    return data\n")


@needs_polar
@pytest.mark.parametrize("include_self_interaction", [False, True])
def test_one_call_equals_two(include_self_interaction):
    """The batched features equal the descriptor's two separate calls on a full-PBC cache."""
    from graph_longrange.features import GTOElectrostaticFeatures

    torch.manual_seed(5)
    descriptor = GTOElectrostaticFeatures(density_max_l=1, density_smearing_width=1.0, feature_max_l=1,
                                          feature_smearing_widths=[1.0, 2.0],
                                          include_self_interaction=include_self_interaction,
                                          kspace_cutoff=2.0).double()
    n, n_k = 6, 9
    positions = torch.randn(n, 3, dtype=torch.float64) * 2
    k_vectors = torch.randn(n_k, 3, dtype=torch.float64)
    k_norm2 = (k_vectors * k_vectors).sum(-1)
    k0 = torch.zeros(n_k, dtype=torch.float64)
    cache = descriptor.precompute_geometry(k_vectors=k_vectors, k_norm2=k_norm2,
                                           k_vector_batch=torch.zeros(n_k, dtype=torch.long), k0_mask=k0,
                                           node_positions=positions, batch=torch.zeros(n, dtype=torch.long),
                                           volume=torch.tensor([30.0], dtype=torch.float64),
                                           pbc=torch.ones(1, 3, dtype=torch.bool))
    alpha = torch.randn(n, 4, dtype=torch.float64)
    beta = torch.randn(n, 4, dtype=torch.float64)
    pbc = torch.ones(1, 3, dtype=torch.bool)
    got_a, got_b = polar.spin_field_features(descriptor, cache, alpha, beta, pbc)
    want_a = descriptor.forward_dynamic(cache=cache, source_feats=alpha.unsqueeze(-2), pbc=pbc)
    want_b = descriptor.forward_dynamic(cache=cache, source_feats=beta.unsqueeze(-2), pbc=pbc)
    torch.testing.assert_close(got_a, want_a, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(got_b, want_b, rtol=1e-13, atol=1e-14)


@needs_polar
def test_other_caches_take_two_calls():
    """Real space, non-periodic corrections and the fused Fourier sums keep the descriptor's own calls."""
    calls = []

    class Descriptor:
        def forward_dynamic(self, cache, source_feats, pbc):
            calls.append(source_feats.shape)
            return source_feats.squeeze(-2) * len(calls)

    alpha, beta = torch.ones(3, 4), torch.ones(3, 4)
    pbc_mode, mixed_mode = 0, 3  # graph_longrange's CORRECTION_MODE_PBC and CORRECTION_MODE_MIXED
    for cache in ({"mode": "realspace"}, {"mode": "pbc", "correction_mode": mixed_mode, "cosines": 1},
                  {"mode": "pbc", "cosines": 1},
                  {"mode": "pbc", "correction_mode": pbc_mode, "fourier_mask": torch.ones(2), "cosines": None}):
        calls.clear()
        a, b = polar.spin_field_features(Descriptor(), cache, alpha, beta, None)
        assert calls == [torch.Size([3, 1, 4])] * 2
        assert torch.equal(a, alpha) and torch.equal(b, 2 * beta)


@needs_polar
@pytest.mark.parametrize("stress", [False, True])
def test_enabled_path_with_spin_batching(monkeypatch, stress):
    calc = tiny_family_calculator("polar", [11, 17])
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    model = copy.deepcopy(calc.models[0])
    polar.enable(model, atoms.cell.array, PolarConfig(), variable_cell=stress)
    monkeypatch.setattr(polar, "SPIN_BATCHED", False)
    expected = _evaluate(calc, model, atoms, stress)
    batched = []
    real = polar.spin_field_features

    def recording(descriptor, cache, alpha, beta, pbc):
        batched.append(cache.get("cosines") is not None and cache.get("correction_mode") == 0)
        return real(descriptor, cache, alpha, beta, pbc)

    monkeypatch.setattr(polar, "spin_field_features", recording)
    monkeypatch.setattr(polar, "SPIN_BATCHED", True)
    got = _evaluate(calc, model, atoms, stress)
    assert batched and all(batched)  # every recursion step took the one-call path
    assert set(got) == set(expected)
    for key in got:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-11, atol=1e-12)


def test_spin_batching_is_the_default():
    # Pure removal of launches, faster on the A100 and the RTX 5060 Ti, gate passed (off-polar-alex).
    assert polar.SPIN_BATCHED is True
