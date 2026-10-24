"""The GPU-only PolarMACE path and the polar_uvu rewrite on the CPU (plan task 1a-T10, SPEC §11.1, §6.2).

Ported from the research engine's ``tests/core/test_polar_gpu.py`` (tag
research-engine-baseline-20261017), with its expected values, and extended:

- the fixed-size k-vector list includes exactly graph_longrange's k-vectors, for the cell it was
  built for and for strained cells inside the headroom, and flags cells outside it;
- the reassociated SparseUvuTensorProduct (MACE PR 1711) matches MACE's at the SPEC §6.4
  thresholds, and its registry record's self-check passes;
- the PR 1762 port, now with explicit options: off by default, r = 0 is the released
  normalisation, the cap is the identity below the onset;
- on a tiny PolarMACE: the enabled path gives MACE's energy and forces, its stress agrees with
  finite differences of the energy (MACE issue 1642), and MACE 0.3.16's own stress does not;
- the compiled plain cuEquivariance route is refused with a reason.

Everything that needs graph_longrange skips without it (``fastmace_cpu`` has none; run these
with ``mace_cueq_latest``).
"""

import copy
import types

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import polar_available, tiny_family_calculator
from fastmace.config import PolarConfig
from fastmace.longrange import polar
from fastmace.rewrites.polar import RECORDS, uvu

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4's constants under torch's weights_only load

needs_polar = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")
Z_VALUES = [11, 17]


def _have(module):
    try:
        __import__(module)
    except ImportError:
        return False
    return True


needs_mace = pytest.mark.skipif(not _have("mace"), reason="MACE is not installed")


def _model(cutoff, dtype=torch.float64):
    return types.SimpleNamespace(kspace_cutoff=torch.tensor(cutoff, dtype=dtype))


def _cells(rng):
    yield np.identity(3) * 16.76  # NaCl 3 x 3 x 3
    yield np.array([[11.6, 0, 0], [0.4, 10.9, 0], [-1.1, 0.7, 12.8]])  # triclinic
    for _ in range(3):
        yield np.diag(rng.uniform(8, 20, 3)) + rng.uniform(-1.5, 1.5, (3, 3))


# --------------------------------------------------------------------------- fixed k-vectors


def _check_fixed(state, cutoff, cell, dtype=torch.float64):
    from graph_longrange.kspace import compute_k_vectors_flat

    cell_t = torch.tensor(cell, dtype=dtype).view(1, 3, 3)
    rcell_t = (2 * torch.pi * torch.linalg.inv(torch.tensor(cell).mT)).to(dtype).view(1, 3, 3)
    cutoff_t = torch.tensor(cutoff, dtype=dtype)
    k_ref, n2_ref, _, k0_ref = compute_k_vectors_flat(cutoff_t, cell_t, rcell_t)
    k, n2, batch, k0 = polar.compute_k_vectors_fixed(polar._Call(state), cutoff_t, cell_t, rcell_t)
    included = batch == 0
    assert int(included.sum()) == len(k_ref)
    assert bool(included[0]) and float(k0[0]) == 1.0 and float(k0[1:].sum()) == 0.0
    torch.testing.assert_close(k[included], k_ref, rtol=0, atol=0)
    torch.testing.assert_close(n2[included], n2_ref, rtol=0, atol=0)
    assert bool((n2[~included] == cutoff_t * cutoff_t).all())  # finite basis values
    return bool(state.overflow)


@needs_polar
def test_fixed_k_vectors_are_graph_longranges_set():
    rng = np.random.default_rng(3)
    for cell in _cells(rng):
        for cutoff in (1.2, 1.85):
            state = polar.PolarState(_model(cutoff), cell, headroom=0.0)
            assert not _check_fixed(state, cutoff, cell)


@needs_polar
def test_strained_cells_inside_the_headroom():
    rng = np.random.default_rng(5)
    cell = np.identity(3) * 16.76
    state = polar.PolarState(_model(1.85), cell, headroom=0.1)
    for _ in range(6):
        e = rng.uniform(-0.03, 0.03, (3, 3))
        strained = cell @ (np.identity(3) + 0.5 * (e + e.T))
        assert not _check_fixed(state, 1.85, strained)


@needs_polar
def test_overflow_is_flagged():
    cell = np.identity(3) * 16.76
    state = polar.PolarState(_model(1.85), cell, headroom=0.0)
    cell_t = torch.tensor(cell * 1.3).view(1, 3, 3)
    rcell_t = (2 * torch.pi * torch.linalg.inv(cell_t[0].mT)).view(1, 3, 3)
    polar.compute_k_vectors_fixed(polar._Call(state), torch.tensor(1.85, dtype=torch.float64), cell_t, rcell_t)
    assert bool(state.overflow)


def test_one_graph_per_batch():
    state = types.SimpleNamespace()
    with pytest.raises(ValueError, match="one graph"):
        polar.compute_k_vectors_fixed(polar._Call(state), torch.tensor(1.0), torch.zeros(2, 3, 3),
                                      torch.zeros(2, 3, 3))


# --------------------------------------------------------------------------- polar_uvu (MACE PR 1711)


def _uvu_products(dtype):
    pytest.importorskip("mace.modules.field_blocks")
    from e3nn import o3
    from mace.modules.field_blocks import SparseUvuTensorProduct

    torch.manual_seed(0)
    irreps1, irreps2 = o3.Irreps("8x0e + 4x1o"), o3.Irreps("6x0e + 3x1o")
    out = o3.Irreps("8x0e + 4x0e + 4x1o")  # l x l -> 0 (both l = 0 and l = 1) and l x 0 -> l
    instructions = [(0, 0, 0, "uvu", True), (1, 1, 1, "uvu", True), (1, 0, 2, "uvu", True)]
    for layout in ("mul_ir", "ir_mul"):
        tp = SparseUvuTensorProduct(irreps1, irreps2, out, instructions, layout=layout).to(dtype)
        x1 = torch.randn(5, irreps1.dim, dtype=dtype)
        x2 = torch.randn(5, irreps2.dim, dtype=dtype)
        yield tp, x1, x2


def _relative(a, b):
    return float((a - b).abs().max() / b.abs().max())


@needs_mace
def test_uvu_matches_mace():
    for tp, x1, x2 in _uvu_products(torch.float64):
        torch.testing.assert_close(uvu.uvu_forward(tp, x1, x2), tp(x1, x2), rtol=1e-12, atol=1e-12)


@needs_mace
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_uvu_within_the_self_check_thresholds(dtype):
    threshold = uvu.THRESHOLDS[str(dtype).removeprefix("torch.")]
    assert uvu.THRESHOLDS == {"float64": 1e-10, "float32": 1e-4}  # SPEC §6.4
    for tp, x1, x2 in _uvu_products(dtype):
        with torch.no_grad():
            assert _relative(uvu.uvu_forward(tp, x1, x2), tp(x1, x2)) <= threshold


def test_record_has_the_rewrite_fields():
    assert set(RECORDS) == {"polar_uvu", "polar_graph_sums"}  # polar_graph_sums: test_polar_graph_sums.py
    record = RECORDS["polar_uvu"]
    assert {"name", "applies_to", "requires", "conflicts", "mace_versions", "source_hashes", "exactness",
            "default", "workload_constraints", "apply", "selfcheck"} <= set(record)  # SPEC §6.1
    assert record["name"] == "polar_uvu" and record["exactness"] == "rounding" and record["default"] == "on"
    assert record["mace_versions"] == ">=0.3.16,<0.3.17"
    assert record["config_field"] == "RewriteConfig.polar_uvu"
    from fastmace.config import RewriteConfig
    assert RewriteConfig().polar_uvu is True


@needs_mace
def test_uvu_source_hashes_match_the_installed_mace():
    pytest.importorskip("mace.modules.field_blocks")
    for qualified, expected in uvu.SOURCE_HASHES.items():
        assert polar.source_digest(polar._resolve(qualified)) == expected, qualified


@needs_polar
def test_uvu_on_a_tiny_polar_model():
    model = tiny_family_calculator("polar", Z_VALUES).models[0]
    assert uvu.applies_to(model) == (True, "ok")
    rewritten = copy.deepcopy(model)
    names = uvu.apply(rewritten)
    assert names and uvu.is_applied(rewritten) and not uvu.is_applied(model)
    assert uvu.is_applied(copy.deepcopy(rewritten))  # a deep copy stays rewritten
    result = uvu.selfcheck(model, rewritten)
    assert result["passed"] and result["modules"] == len(names) and result["max_relative_error"] <= 1e-10
    assert uvu.revert(rewritten) == names and not uvu.is_applied(rewritten)


@needs_polar
def test_uvu_self_check_catches_a_wrong_rewrite():
    model = tiny_family_calculator("polar", Z_VALUES).models[0]
    broken = copy.deepcopy(model)
    for _, module in uvu._targets(broken):
        module.forward = types.MethodType(lambda self, x1, x2: 1.001 * uvu.uvu_forward(self, x1, x2), module)
    result = uvu.selfcheck(model, broken)
    assert not result["passed"] and "relative error" in result["detail"]


@needs_mace
def test_uvu_does_not_apply_to_other_models():
    model = tiny_family_calculator("mh1", Z_VALUES).models[0]
    supported, reason = uvu.applies_to(model)
    assert not supported and "not PolarMACE" in reason


# --------------------------------------------------------------------------- MACE PR 1762


def _fukui(sources, batch, r):
    pytest.importorskip("mace.tools.scatter")
    return polar.normalise_fukui(sources, batch, int(batch.max()) + 1, r)


def test_pr1762_is_off_by_default():
    config = PolarConfig()
    assert config.fukui_regularisation == 0.0 and config.dipole_cap is None
    assert not polar.stabilised(config) and not polar.stabilised(None)
    assert polar.stabilised(PolarConfig(fukui_regularisation=0.1))
    assert polar.stabilised(PolarConfig(dipole_cap=(2.0, 3.0)))


def test_r0_is_the_released_normalisation():
    torch.manual_seed(1)
    f = torch.randn(7, 2, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    sums = torch.stack([f[:3].sum(0), f[3:].sum(0)])[batch]
    torch.testing.assert_close(_fukui(f, batch, 0.0), f / sums, rtol=0, atol=0)


def test_weights_sum_to_one_and_stay_bounded_when_the_sum_cancels():
    batch = torch.zeros(4, dtype=torch.long)
    f = torch.tensor([[1.0, 2.0], [-1.0, -2.0], [0.5, 1e-9], [-0.5, 0.0]], dtype=torch.float64)  # s ~ 0
    for r in (0.05, 0.1, 0.3):
        w = _fukui(f, batch, r)
        torch.testing.assert_close(w.sum(0), torch.ones(2, dtype=torch.float64))
        assert bool((w.abs() <= 1 / (2 * r) + 1).all())


def test_close_to_f_over_s_without_cancellation():
    torch.manual_seed(2)
    f = torch.rand(6, 2, dtype=torch.float64) + 0.5  # all positive: |s| = sum|f|
    batch = torch.zeros(6, dtype=torch.long)
    r = 0.1
    exact = f / f.sum(0)
    w = _fukui(f, batch, r)
    assert float((w - exact).abs().max() / exact.abs().max()) < 2 * r * r


def test_all_zero_sources_give_uniform_weights():
    w = _fukui(torch.zeros(5, 2, dtype=torch.float64), torch.zeros(5, dtype=torch.long), 0.1)
    torch.testing.assert_close(w, torch.full((5, 2), 0.2, dtype=torch.float64))


def test_dipole_cap():
    torch.manual_seed(3)
    density = torch.randn(6, 2, 8, dtype=torch.float64) * 0.2  # [atoms, spin, (q, dipole, q, dipole)]
    density[0, :, 1:4] = 5.0  # charge dipole |d| = 2 * 5 * sqrt(3), far above the ceiling
    capped = polar.cap_charge_dipole(density, (2.0, 3.0))
    small = (density[:, 0, 1:4] + density[:, 1, 1:4]).norm(dim=-1) <= 2.0
    assert bool(small[1:].all())
    torch.testing.assert_close(capped[1:], density[1:], rtol=0, atol=0)  # identity below the onset
    charge = capped[0, 0] + capped[0, 1]
    assert 2.0 < float(charge[1:4].norm()) < 3.0
    torch.testing.assert_close(charge[0], density[0, 0, 0] + density[0, 1, 0])  # monopole unchanged
    torch.testing.assert_close(capped[0, 0] - capped[0, 1], density[0, 0] - density[0, 1])  # spin unchanged
    assert polar.cap_charge_dipole(density, None) is density
    assert polar.cap_charge_dipole(density) is density  # off by default


def test_spin_restriction_is_not_carried_over():
    # SPEC §6.2 "Never enable"; switch map: polar_gpu.SPIN_RESTRICTED not carried over.
    assert not hasattr(polar, "restrict_spin")
    assert "spin" not in {f.name for f in __import__("dataclasses").fields(PolarConfig)}
    assert all("restrict_spin" not in new for _, new in polar.PR1762_EDITS)


@needs_polar
def test_source_rewrite_applies_to_the_installed_mace():
    import mace.modules.extensions as extensions

    forward = polar.pr1762_forward(polar._original("mace.modules.extensions:PolarMACE.forward"))
    del extensions
    names = forward.__code__.co_names
    assert "_fastmace_normalise_fukui" in names and "_fastmace_cap_charge_dipole" in names


def test_source_rewrite_refuses_other_source():
    pytest.importorskip("mace.modules.extensions")

    def forward(self, data):  # not MACE 0.3.16's PolarMACE.forward
        return data

    with pytest.raises(polar.PolarUnavailableError, match="PR 1762"):
        polar.pr1762_forward(forward)


# --------------------------------------------------------------------------- routes and options


def test_compiled_plain_cueq_is_refused():
    supported, reason = polar.route_support("cueq", "default")
    assert not supported and reason == polar.ZERO_FORCE_REASON and "zero forces" in reason
    assert polar.route_support("cueq", "none") == (True, "ok")
    assert polar.route_support("cueq", None) == (True, "ok")
    assert polar.route_support("hybrid", "default") == (True, "ok")
    assert polar.route_support("hybrid", "none") == (True, "ok")
    assert not polar.route_support("e3nn", "none")[0]
    with pytest.raises(polar.PolarUnavailableError, match="zero forces"):
        polar.require_route("cueq", "default")
    polar.require_route("hybrid", "default")


def test_kspace_headroom_resolution():
    assert polar.resolve_kspace_headroom(PolarConfig(), variable_cell=False) == 0.0
    assert polar.resolve_kspace_headroom(PolarConfig(), variable_cell=True) == 0.1
    assert polar.resolve_kspace_headroom(PolarConfig(kspace_headroom=0.25), variable_cell=False) == 0.25


def test_dynamo_reset_is_a_documented_process_constraint():
    assert any(c.startswith("dynamo_reset_per_polar_engine") for c in polar.PROCESS_CONSTRAINTS)
    assert "torch._dynamo.reset()" in polar.__doc__ and "torch._dynamo.reset()" in polar.enable.__doc__
    import inspect
    assert "_dynamo.reset(" not in inspect.getsource(polar).replace("torch._dynamo.reset()", "")


def test_charge_and_spin_validation():
    nacl_numbers = [11, 17] * 4  # 224 electrons
    assert polar.validate_charge_spin(nacl_numbers) == (0, 1)
    assert polar.validate_charge_spin(nacl_numbers, charge=1, spin_multiplicity=2) == (1, 2)
    assert polar.validate_charge_spin(nacl_numbers, 0, 3) == (0, 3)
    with pytest.raises(ValueError, match="impossible"):
        polar.validate_charge_spin(nacl_numbers, 0, 2)
    with pytest.raises(ValueError, match="at least 1"):
        polar.validate_charge_spin(nacl_numbers, 0, 0)
    with pytest.raises(ValueError, match="integer"):
        polar.validate_charge_spin(nacl_numbers, 0.5, 1)
    with pytest.raises(ValueError, match="integer"):
        polar.validate_charge_spin(nacl_numbers, True, 1)


# --------------------------------------------------------------------------- the path on a tiny PolarMACE


def _evaluate(calc, model, atoms, stress=True):
    """MACE's own data preparation, then the model's forward (energy, forces, stress) as float64 numpy."""
    batch = calc._atoms_to_batch(atoms)
    out = model(batch.to_dict(), compute_stress=stress, training=False)
    energy = float(out["energy"].detach().reshape(-1)[0])
    forces = out["forces"].detach().numpy()
    return energy, forces, (out["stress"].detach().numpy()[0] if stress else None)


def _deformed(atoms, deformation):
    deformed = atoms.copy()
    deformed.set_cell(atoms.cell.array @ deformation, scale_atoms=False)
    deformed.positions = atoms.positions @ deformation
    return deformed


def _fd_stress(calc, model, atoms, h=1e-4):
    """Central differences sigma_ij = (E(+h) - E(-h)) / (2 h V) for a symmetric strain (polar_stress_probe.py)."""
    volume = abs(np.linalg.det(atoms.cell.array))
    fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            eps = np.zeros((3, 3))
            if i == j:
                eps[i, i] = h
            else:
                eps[i, j] = eps[j, i] = h / 2
            plus = _evaluate(calc, model, _deformed(atoms, np.identity(3) + eps), stress=False)[0]
            minus = _evaluate(calc, model, _deformed(atoms, np.identity(3) - eps), stress=False)[0]
            fd[i, j] = fd[j, i] = (plus - minus) / (2 * h * volume)
    return fd


@pytest.fixture
def tiny_polar():
    if not polar_available():
        pytest.skip("PolarMACE needs graph_longrange")
    calc = tiny_family_calculator("polar", Z_VALUES)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    e = np.random.default_rng(20261016).uniform(-0.02, 0.02, (3, 3))
    atoms.set_cell(atoms.cell.array @ (np.identity(3) + 0.5 * (e + e.T)), scale_atoms=True)  # sheared
    return calc, calc.models[0], atoms


def test_install_is_idempotent_and_qualified(tiny_polar):
    assert polar.source_problems() == []
    record = polar.install()
    assert record == polar.install() == polar.installed()
    assert record["name"] == polar.PATCH_NAME
    assert "mace.modules.extensions:PolarMACE.forward" in record["targets"]
    assert "e3nn.o3._irreps:Irrep.__len__" in record["targets"]


def test_models_that_are_not_enabled_keep_maces_code(tiny_polar):
    calc, model, atoms = tiny_polar
    polar.install()
    released = polar._original("mace.modules.extensions:PolarMACE.forward")
    _evaluate(calc, model, atoms)  # throwaway: the first model of a process rounds differently (docs/development.md)
    batch = calc._atoms_to_batch(atoms)
    reference = released(model, batch.to_dict(), compute_stress=True, training=False)
    out = model(calc._atoms_to_batch(atoms).to_dict(), compute_stress=True, training=False)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(out[key], reference[key], rtol=0, atol=0)
    assert polar.get_state(model) is None and polar.get_config(model) == PolarConfig()


def test_enabled_path_gives_maces_energy_and_forces(tiny_polar):
    calc, model, atoms = tiny_polar
    e0, f0, _ = _evaluate(calc, model, atoms)
    enabled = copy.deepcopy(model)
    info = polar.enable(enabled, atoms.cell.array)
    assert info["headroom"] == 0.0 and 0 < info["k_vectors"] <= info["k_superset"]
    e1, f1, _ = _evaluate(calc, enabled, atoms)
    np.testing.assert_allclose(e1, e0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f1, f0, rtol=1e-12, atol=1e-12)
    assert not polar.overflowed(enabled)


def test_stress_fix_against_finite_differences(tiny_polar):
    calc, model, atoms = tiny_polar
    enabled = copy.deepcopy(model)
    polar.enable(enabled, atoms.cell.array, variable_cell=True)  # headroom 0.1 for the strained cells
    _, _, stress = _evaluate(calc, enabled, atoms)
    _, _, released = _evaluate(calc, model, atoms)
    fd = _fd_stress(calc, enabled, atoms)
    scale = np.abs(fd).max()
    gpu_error, released_error = np.abs(stress - fd).max(), np.abs(released - fd).max()
    # CPU float64, 17 October 2026: 2.8e-7 (h = 1e-4 truncation) against 9.0e-2 for MACE 0.3.16.
    assert gpu_error < 1e-5 * scale, (gpu_error, scale)
    assert released_error > 1e-2 * scale, (released_error, scale)
    assert not polar.overflowed(enabled)
    # The finite differences of the released model's energy are the same function (energy unchanged).
    np.testing.assert_allclose(_fd_stress(calc, model, atoms), fd, rtol=0, atol=1e-6 * scale)


def test_state_is_per_instance(tiny_polar):
    calc, model, atoms = tiny_polar
    first = copy.deepcopy(model)
    polar.enable(first, atoms.cell.array)
    copied = copy.deepcopy(first)
    assert polar.get_state(copied) is not polar.get_state(first)
    assert polar.get_state(copied).bounds == polar.get_state(first).bounds
    # Enabling the copy for another structure leaves the first instance alone.
    other = nacl(repeat=2)
    polar.enable(copied, other.cell.array)
    assert polar.get_state(first).bounds != polar.get_state(copied).bounds
    np.testing.assert_array_equal(polar.get_state(first).cell, atoms.cell.array)
    # A stale state evaluated on another cell is caught by the deformation test.
    stale = copy.deepcopy(first)
    _evaluate(calc, stale, other, stress=False)
    assert polar.overflowed(stale)
    polar.reset_overflow(stale)
    assert not polar.overflowed(stale)
    polar.disable(stale)
    assert polar.get_state(stale) is None


def test_enable_refuses_other_models_and_open_boundaries(tiny_polar):
    calc, model, atoms = tiny_polar
    with pytest.raises(ValueError, match="periodic"):
        polar.enable(copy.deepcopy(model), atoms.cell.array, pbc=(True, True, False))
    with pytest.raises(TypeError, match="not PolarMACE"):
        polar.enable(tiny_family_calculator("mh1", Z_VALUES).models[0], atoms.cell.array)
    with pytest.raises(TypeError, match="PolarConfig"):
        polar.configure(copy.deepcopy(model), {"fukui_regularisation": 0.1})


def test_pr1762_forward_with_inactive_options_is_the_released_model(tiny_polar):
    calc, model, atoms = tiny_polar
    _evaluate(calc, model, atoms)  # throwaway: the first model of a process rounds differently (docs/development.md)
    e0, f0, s0 = _evaluate(calc, model, atoms)
    # Off by default: an enabled model with the default config runs the released forward.
    default = copy.deepcopy(model)
    polar.configure(default, PolarConfig())
    e1, f1, _ = _evaluate(calc, default, atoms)
    assert (e1, f1.tobytes()) == (e0, f0.tobytes())
    # The rewritten forward (cap far above any dipole, r = 0) gives the released numbers.
    rewritten = copy.deepcopy(model)
    polar.configure(rewritten, PolarConfig(dipole_cap=(1e3, 2e3)))
    e2, f2, s2 = _evaluate(calc, rewritten, atoms)
    np.testing.assert_allclose(e2, e0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f2, f0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s2, s0, rtol=1e-12, atol=1e-12)
    # A strong regularisation changes the model (opt-in only, SPEC §6.2).
    regularised = copy.deepcopy(model)
    polar.configure(regularised, PolarConfig(fukui_regularisation=0.5))
    e3, f3, _ = _evaluate(calc, regularised, atoms)
    assert np.isfinite(e3) and np.isfinite(f3).all() and e3 != e0


def test_enabled_path_with_pr1762_options(tiny_polar):
    calc, model, atoms = tiny_polar
    config = PolarConfig(fukui_regularisation=0.3, dipole_cap=(2.0, 3.0))
    reference = copy.deepcopy(model)
    polar.configure(reference, config)  # MACE's long-range code with the PR 1762 forward
    enabled = copy.deepcopy(model)
    polar.enable(enabled, atoms.cell.array, config)
    e0, f0, _ = _evaluate(calc, reference, atoms)
    e1, f1, _ = _evaluate(calc, enabled, atoms)
    np.testing.assert_allclose(e1, e0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f1, f0, rtol=1e-12, atol=1e-12)
    assert polar.get_config(enabled) is config
