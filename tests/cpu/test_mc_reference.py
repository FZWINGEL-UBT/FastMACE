"""The exact MC reference of the enumeration tests (plan task 3-T12; SPEC §12.2 "MC exact enumeration").

``_support.mc_reference`` holds the exact chain that the driver tests compare with: the 70 swap and 256
transmutation configurations, their Boltzmann populations from float64 reference energies of the tiny
model, the transition matrix with its second eigenvalue (the thinning) and the χ² population test with
its power checks. These tests check the reference itself, with samples drawn from the exact chain, so a
failure of the driver tests later points at the driver.
"""

import math

import numpy as np
import pytest
from scipy import stats

from _support import mc_reference as ref
from _support.structures import nacl
from _support.tiny_models import tiny_mace_calculator

Z_NA, Z_CL, Z_K = 11, 17, 19


@pytest.fixture(scope="module")
def calculator():
    return tiny_mace_calculator([Z_NA, Z_CL, Z_K])


@pytest.fixture(scope="module")
def swap_case(calculator):
    """NaCl's conventional cell: 8 sites, 4 Na + 4 Cl, every arrangement (70)."""
    return ref.swap_case(nacl(repeat=1), (Z_NA, Z_CL), calculator)


@pytest.fixture(scope="module")
def transmute_case(calculator):
    """NaCl 2×1×1: the 8 cation sites Na or K, Cl fixed (256)."""
    atoms = nacl(repeat=1).repeat((2, 1, 1))
    sites = np.flatnonzero(atoms.numbers == Z_NA)
    return ref.transmute_case(atoms, sites, (Z_NA, Z_K), calculator)


def test_configurations_are_complete_and_ordered():
    swaps = ref.swap_configurations()
    assert swaps.shape == (70, 8)
    assert set(swaps.sum(axis=1)) == {4}
    assert len({row.tobytes() for row in swaps}) == 70
    assert [tuple(row) for row in swaps] == sorted(tuple(row) for row in swaps)

    flips = ref.transmute_configurations()
    assert flips.shape == (256, 8)
    assert len({row.tobytes() for row in flips}) == 256
    assert ref.index_of(flips, flips[137]) == 137
    with pytest.raises(KeyError):
        ref.index_of(swaps, np.zeros(8, dtype=np.int8))


@pytest.mark.parametrize("move", ["swap", "transmute"])
def test_proposals_are_single_moves_and_symmetric(move):
    configs = ref.swap_configurations() if move == "swap" else ref.transmute_configurations()
    targets = ref.proposals(configs, move)
    assert targets.shape == (len(configs), 16 if move == "swap" else 8)
    for c, row in enumerate(targets):
        for d in row:
            changed = np.flatnonzero(configs[c] != configs[d])
            if move == "swap":
                assert len(changed) == 2 and configs[c][changed].sum() == 1
            else:
                assert len(changed) == 1
            # The reverse move is proposed with the same probability.
            assert np.count_nonzero(targets[d] == c) == np.count_nonzero(row == d) == 1


def test_swap_proposal_order_is_a_sites_by_b_sites():
    configs = ref.swap_configurations()
    c = ref.index_of(configs, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    expected = []
    for i in range(4):
        for j in range(4, 8):
            labels = configs[c].copy()
            labels[[i, j]] = labels[[j, i]]
            expected.append(ref.index_of(configs, labels))
    assert ref.proposals(configs, "swap")[c].tolist() == expected


@pytest.mark.parametrize("move", ["swap", "transmute"])
def test_transition_matrix_is_reversible_with_boltzmann_stationary(move):
    configs = ref.swap_configurations() if move == "swap" else ref.transmute_configurations()
    x = np.random.default_rng(7).normal(scale=0.05, size=len(configs))
    kT = ref.rule_kT(x)
    targets = ref.proposals(configs, move)
    matrix = ref.transition_matrix(x, kT, targets)
    pi = ref.boltzmann(x, kT)
    assert np.all(matrix >= 0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    flux = pi[:, None] * matrix
    np.testing.assert_allclose(flux, flux.T, rtol=0, atol=1e-15)
    np.testing.assert_allclose(pi @ matrix, pi, rtol=0, atol=1e-14)
    # Each off-diagonal entry is the proposal probability times the Metropolis acceptance.
    c = 3
    d = targets[c][0]
    assert matrix[c, d] == pytest.approx(min(1.0, math.exp(-(x[d] - x[c]) / kT)) / targets.shape[1], rel=1e-14)


def test_second_eigenvalue_and_thinning():
    configs = ref.swap_configurations()
    x = np.random.default_rng(8).normal(scale=0.05, size=len(configs))
    kT = ref.rule_kT(x)
    matrix = ref.transition_matrix(x, kT, ref.proposals(configs, "swap"))
    lambda2 = ref.second_eigenvalue(matrix, ref.boltzmann(x, kT))
    moduli = np.sort(np.abs(np.linalg.eigvals(matrix)))
    assert moduli[-1] == pytest.approx(1.0, abs=1e-12)
    assert lambda2 == pytest.approx(moduli[-2], abs=1e-10)
    assert 0 < lambda2 < 1
    k = ref.thinning(lambda2)
    assert lambda2 ** k <= ref.MAX_LAG_CORRELATION < lambda2 ** (k - 1)
    assert ref.thinning(0.0) == 1


def test_rule_kT_spans_e_cubed():
    x = np.array([0.3, -0.2, 0.1, 0.25])
    kT = ref.rule_kT(x)
    assert kT == pytest.approx(0.5 / ref.SPAN_KT)
    pi = ref.boltzmann(x, kT)
    assert pi.max() / pi.min() == pytest.approx(math.exp(ref.SPAN_KT), rel=1e-12)
    assert pi.sum() == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError, match="span"):
        ref.rule_kT(np.zeros(3))


def test_metropolis_decision_and_near_ties():
    x = np.array([0.0, 0.1])
    kT = 0.05
    log_u_tie = -(0.1 / 0.05)
    assert ref.metropolis_decision(x, kT, 1, 0, log_u=-0.01) == (True, False)  # downhill: always
    assert ref.metropolis_decision(x, kT, 0, 1, log_u=log_u_tie - 0.1) == (True, False)
    assert ref.metropolis_decision(x, kT, 0, 1, log_u=log_u_tie + 0.1) == (False, False)
    assert ref.metropolis_decision(x, kT, 0, 1, log_u=log_u_tie + 1e-9, margin=1e-6) == (False, True)
    assert ref.metropolis_decision(x, kT, 0, 1, log_u=log_u_tie - 1e-9, margin=1e-6) == (True, True)


def test_chi_squared():
    p = np.array([0.5, 0.25, 0.25])
    exact = ref.chi_squared(np.array([50, 25, 25]), p)
    assert (exact.statistic, exact.dof, exact.p_value) == (0.0, 2, 1.0)
    off = ref.chi_squared(np.array([60, 20, 20]), p)
    assert off.statistic == pytest.approx(100 / 50 + 25 / 25 + 25 / 25)
    assert off.p_value == pytest.approx(stats.chi2.sf(4.0, 2))
    with pytest.raises(ValueError, match="bins"):
        ref.chi_squared(np.array([1, 2]), p)


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("case", ["swap_case", "transmute_case"])
def test_sample_size_gives_the_power(case, pool, request):
    plan = request.getfixturevalue(case).plan
    if pool:
        plan = ref.pooled_plan(plan)
    pi = ref.pooled(plan.pi, plan.groups)
    assert plan.n_samples * pi.min() >= ref.MIN_EXPECTED
    assert ref.sample_size(pi, [ref.pooled(q, plan.groups) for q in ref.alternatives(plan.x, plan.kT).values()],
                           min_expected=ref.MIN_EXPECTED) == plan.n_samples
    dof = len(pi) - 1
    critical = stats.chi2.isf(ref.P_FAIL, dof)
    for alternative in ref.alternatives(plan.x, plan.kT).values():
        q = ref.pooled(alternative, plan.groups)
        noncentrality = plan.n_samples * np.sum((pi - q) ** 2 / q)
        assert stats.ncx2.sf(critical, dof, noncentrality) >= ref.POWER
    assert plan.attempts == plan.thinning * plan.n_samples


def test_levels_group_equal_energies():
    x = np.array([0.2, 0.0, 0.2 + 1e-12, 0.1, 0.0])
    groups = ref.levels(x, kT=0.05)
    assert groups.tolist() == [2, 0, 2, 1, 0]
    assert ref.pooled(np.array([1, 2, 3, 4, 5]), groups).tolist() == [7, 4, 4]
    assert ref.pooled(np.array([1, 2]), None).tolist() == [1, 2]


def test_tiny_model_levels_are_symmetry_classes(swap_case, transmute_case):
    # NaCl's symmetry leaves few distinct swap energies; the levels must not split them by noise.
    swap_groups = ref.pooled_plan(swap_case.plan).groups
    assert 2 <= swap_groups.max() + 1 < 70
    transmute_groups = ref.pooled_plan(transmute_case.plan).groups
    assert 9 <= transmute_groups.max() + 1 < 256  # at least one level per K count


def test_tiny_model_swap_energies(swap_case, calculator):
    energies = swap_case.energies
    assert energies.dtype == np.float64 and energies.shape == (70,)
    assert np.all(np.isfinite(energies))
    assert swap_case.plan.kT > 0
    # MACE's calculator repeats an energy to about 1e-11 eV, not bitwise.
    again = ref.reference_energies(nacl(repeat=1), np.arange(8), (Z_NA, Z_CL), swap_case.configs[:5], calculator)
    np.testing.assert_allclose(again, energies[:5], rtol=0, atol=1e-10)
    # The crystal's own arrangement is one of the configurations.
    labels = (nacl(repeat=1).numbers == Z_CL).astype(np.int8)
    assert ref.index_of(swap_case.configs, labels) in range(70)


def test_tiny_model_transmute_energies_and_mu(transmute_case):
    case = transmute_case
    assert case.energies.shape == (256,) and np.all(np.isfinite(case.energies))
    n_b = case.configs.sum(axis=1)
    np.testing.assert_allclose(case.plan.x, case.energies - case.delta_mu * n_b, rtol=0, atol=1e-12)
    # Δμ is the least-squares slope of E against the K count, so the populations centre on the
    # middle compositions and every composition is visited.
    slope = np.polyfit(n_b, case.energies, 1)[0]
    assert case.delta_mu == pytest.approx(slope, rel=1e-10)


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize(("case", "seed"), [("swap_case", 20261022), ("transmute_case", 20261023)])
def test_population_test_on_exact_chain_samples(case, seed, pool, request):
    plan = request.getfixturevalue(case).plan
    if pool:
        plan = ref.pooled_plan(plan)
    rng = np.random.default_rng(seed)
    samples = ref.sample_thinned_chain(plan.matrix, plan.thinning, plan.n_samples, start=0, rng=rng)
    result = ref.population_test(ref.counts_of(samples, len(plan.pi)), plan.x, plan.kT, plan.groups)
    assert result.target.p_value >= ref.P_PASS, result
    assert result.uniform.p_value <= ref.P_FAIL, result
    assert result.wrong_beta.p_value <= ref.P_FAIL, result
    assert result.passes


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize(("case", "seed"), [("swap_case", 20261024), ("transmute_case", 20261025)])
def test_population_test_rejects_a_chain_at_the_wrong_temperature(case, seed, pool, request):
    plan = request.getfixturevalue(case).plan
    if pool:
        plan = ref.pooled_plan(plan)
    wrong = ref.transition_matrix(plan.x, plan.kT / ref.WRONG_BETA_FACTOR, plan.targets)
    rng = np.random.default_rng(seed)
    samples = ref.sample_thinned_chain(wrong, plan.thinning, plan.n_samples, start=0, rng=rng)
    result = ref.population_test(ref.counts_of(samples, len(plan.pi)), plan.x, plan.kT, plan.groups)
    assert result.target.p_value <= ref.P_FAIL, result
    assert not result.passes


def test_thinned_and_counts():
    sequence = np.arange(10)
    assert ref.thinned(sequence, 3).tolist() == [2, 5, 8]
    assert ref.counts_of(np.array([0, 2, 2]), 4).tolist() == [1, 0, 2, 0]


def test_batch_mean_and_agree():
    values = np.repeat([1.0, 3.0], 5)
    mean, error = ref.batch_mean(values, n_batches=2)
    assert mean == 2.0 and error == pytest.approx(1.0)
    # Independent normal values: the standard error matches sigma / sqrt(n).
    rng = np.random.default_rng(3)
    mean, error = ref.batch_mean(rng.normal(size=100_000))
    assert abs(mean) < 4 * error and error == pytest.approx(1 / math.sqrt(100_000), rel=0.5)
    assert ref.agree((1.0, 0.1), (1.3, 0.0)) and not ref.agree((1.0, 0.1), (1.5, 0.0))
    assert ref.agree((1.0, 0.3), (2.0, 0.4)) and not ref.agree((1.0, 0.03), (2.0, 0.04))
    with pytest.raises(ValueError, match="batches"):
        ref.batch_mean([1.0], n_batches=2)


def test_cases_from_given_energies(swap_case, transmute_case):
    again = ref.swap_case(nacl(repeat=1), (Z_NA, Z_CL), None, energies=swap_case.energies.tolist())
    np.testing.assert_array_equal(again.plan.matrix, swap_case.plan.matrix)
    atoms = nacl(repeat=1).repeat((2, 1, 1))
    other = ref.transmute_case(atoms, transmute_case.sites, (Z_NA, Z_K), None, energies=transmute_case.energies)
    assert other.delta_mu == transmute_case.delta_mu
    np.testing.assert_array_equal(other.plan.x, transmute_case.plan.x)
