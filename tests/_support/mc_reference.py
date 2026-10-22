"""The exact chain of the MC enumeration tests (plan task 3-T12; SPEC §12.2 "MC exact enumeration").

A small cell at fixed positions has few enough configurations to evaluate them all with the float64
reference: 8 sites with 4 + 4 atoms (70 swap configurations), or 8 sites each of two species (256
transmutation configurations). From those energies this module builds what the driver is compared
with:

- the Boltzmann populations, with x_c = E_c − Δμ N_B(c) (Δμ configurational, §10.4);
- the transition matrix of the Metropolis chain with the driver's symmetric proposals, and its second
  eigenvalue, which sets how far apart samples must be;
- the χ² test of sampled populations, with its power checks: the same counts must fail against
  uniform populations and against 1.5 β.

A configuration is a vector of labels per site, 0 for species A and 1 for species B (the
transmutation's second species). The thresholds below are fixed in advance (3-T12 prestart):
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

# kT is the span of x divided by SPAN_KT, so the Boltzmann weights span e^SPAN_KT and every
# configuration is visited, without tuning the temperature per model.
SPAN_KT = 3.0
# The populations pass at p ≥ P_PASS; the power checks must give p ≤ P_FAIL.
P_PASS = 1e-3
P_FAIL = 1e-6
# Samples are taken every k attempts, with λ2^k ≤ MAX_LAG_CORRELATION, so the χ² test, which
# assumes independent samples, applies.
MAX_LAG_CORRELATION = 0.01
# Sample size: every configuration expects at least MIN_EXPECTED samples, and each power check fails
# at p ≤ P_FAIL with probability at least POWER.
MIN_EXPECTED = 10
POWER = 0.999
# The wrong-temperature alternative: β' = WRONG_BETA_FACTOR β.
WRONG_BETA_FACTOR = 1.5
# Configurations within LEVEL_TOLERANCE kT of each other form one energy level for the pooled test
# (far above the reference's run-to-run noise of about 1e-11 eV, far below real splittings).
LEVEL_TOLERANCE = 1e-6

MOVES = ("swap", "transmute")


def swap_configurations(n_sites: int = 8, n_b: int = 4) -> np.ndarray:
    """All arrangements of n_b B atoms on n_sites sites, ``(C, n_sites)`` int8, in lexicographic order."""
    rows = []
    for b_sites in itertools.combinations(range(n_sites), n_b):
        labels = np.zeros(n_sites, dtype=np.int8)
        labels[list(b_sites)] = 1
        rows.append(labels)
    return np.array(sorted(rows, key=tuple), dtype=np.int8)


def transmute_configurations(n_sites: int = 8) -> np.ndarray:
    """All 2^n_sites labelings, ``(C, n_sites)`` int8, in lexicographic order."""
    return np.array(list(itertools.product((0, 1), repeat=n_sites)), dtype=np.int8)


def index_of(configs: np.ndarray, labels) -> int:
    """The row of ``configs`` equal to ``labels``; KeyError if there is none."""
    hits = np.flatnonzero(np.all(configs == np.asarray(labels, dtype=np.int8), axis=1))
    if len(hits) != 1:
        raise KeyError(f"no configuration {np.asarray(labels).tolist()}")
    return int(hits[0])


def proposals(configs: np.ndarray, move: str) -> np.ndarray:
    """``(C, n_proposals)``: the configuration each equally likely proposal leads to.

    - swap: atom i among the A sites and atom j among the B sites, each uniform, in the order
      (i ascending, then j ascending);
    - transmute: one site uniform, ascending, to the other species (S has two).
    """
    if move not in MOVES:
        raise ValueError(f"move must be one of {MOVES}, got {move!r}")
    lookup = {row.tobytes(): c for c, row in enumerate(configs)}
    targets = []
    for labels in configs:
        row = []
        if move == "swap":
            for i in np.flatnonzero(labels == 0):
                for j in np.flatnonzero(labels == 1):
                    new = labels.copy()
                    new[[i, j]] = new[[j, i]]
                    row.append(lookup[new.tobytes()])
        else:
            for i in range(len(labels)):
                new = labels.copy()
                new[i] = 1 - new[i]
                row.append(lookup[new.tobytes()])
        targets.append(row)
    return np.array(targets, dtype=np.int64)


def reference_energies(atoms, sites, species, configs: np.ndarray, calculator) -> np.ndarray:
    """float64 energies (eV) of ``atoms`` with ``sites`` set to ``species[label]`` per configuration."""
    sites = np.asarray(sites)
    species = np.asarray(species)
    atoms = atoms.copy()
    atoms.calc = calculator
    energies = np.empty(len(configs), dtype=np.float64)
    for c, labels in enumerate(configs):
        numbers = atoms.numbers.copy()
        numbers[sites] = species[labels]
        atoms.numbers = numbers
        energies[c] = atoms.get_potential_energy()
    return energies


def effective_energies(energies: np.ndarray, configs: np.ndarray, delta_mu: float = 0.0) -> np.ndarray:
    """x_c = E_c − Δμ N_B(c), the exponent of the semi-grand-canonical weights (swaps: Δμ = 0)."""
    return np.asarray(energies, dtype=np.float64) - delta_mu * configs.sum(axis=1)


def fitted_delta_mu(energies: np.ndarray, configs: np.ndarray) -> float:
    """The least-squares slope of E against N_B: a Δμ that centres the populations on the middle
    compositions, so that every composition is visited."""
    slope, _ = np.polyfit(configs.sum(axis=1), energies, 1)
    return float(slope)


def rule_kT(x: np.ndarray, span: float = SPAN_KT) -> float:
    """kT (eV) such that the Boltzmann weights of ``x`` span e^span."""
    width = float(np.max(x) - np.min(x))
    if not width > 0:
        raise ValueError("rule_kT: the energies have no span")
    return width / span


def boltzmann(x: np.ndarray, kT: float) -> np.ndarray:
    """Normalised exp(−x/kT)."""
    weights = np.exp(-(x - np.min(x)) / kT)
    return weights / weights.sum()


def transition_matrix(x: np.ndarray, kT: float, targets: np.ndarray) -> np.ndarray:
    """P[c, d]: the proposal probability times min(1, exp(−(x_d − x_c)/kT)); rejections on the diagonal."""
    n_configs, n_proposals = targets.shape
    matrix = np.zeros((n_configs, n_configs))
    for c in range(n_configs):
        for d in targets[c]:
            matrix[c, d] += min(1.0, math.exp(-(x[d] - x[c]) / kT)) / n_proposals
        matrix[c, c] += 1.0 - matrix[c].sum()
    return matrix


def second_eigenvalue(matrix: np.ndarray, pi: np.ndarray) -> float:
    """The largest |λ| below the eigenvalue 1 of a reversible chain with stationary ``pi``."""
    root = np.sqrt(pi)
    symmetric = root[:, None] * matrix / root[None, :]
    eigenvalues = np.linalg.eigvalsh((symmetric + symmetric.T) / 2)
    return float(max(abs(eigenvalues[-2]), abs(eigenvalues[0])))


def thinning(lambda2: float, max_correlation: float = MAX_LAG_CORRELATION) -> int:
    """The smallest k ≥ 1 with λ2^k ≤ max_correlation."""
    if lambda2 <= max_correlation:
        return 1
    k = math.ceil(math.log(max_correlation) / math.log(lambda2))
    while lambda2 ** k > max_correlation:  # rounding of the logarithms
        k += 1
    while k > 1 and lambda2 ** (k - 1) <= max_correlation:
        k -= 1
    return k


def alternatives(x: np.ndarray, kT: float) -> dict[str, np.ndarray]:
    """The populations the samples must be told apart from."""
    return {"uniform": np.full(len(x), 1.0 / len(x)),
            "wrong_beta": boltzmann(x, kT / WRONG_BETA_FACTOR)}


def sample_size(pi: np.ndarray, alternative_populations, *, min_expected: int = MIN_EXPECTED,
                p_fail: float = P_FAIL, power: float = POWER) -> int:
    """Samples needed for MIN_EXPECTED per configuration and the power of every check.

    Counts drawn from ``pi`` and tested against an alternative q give a noncentral χ² with
    λ = N Σ (pi − q)² / q; N is the smallest with P(χ² > critical(p_fail)) ≥ power for each q.
    """
    dof = len(pi) - 1
    critical = stats.chi2.isf(p_fail, dof)
    n = math.ceil(min_expected / float(np.min(pi)))
    for q in alternative_populations:
        distance = float(np.sum((pi - q) ** 2 / q))
        needed = optimize.brentq(lambda lam: stats.ncx2.sf(critical, dof, lam) - power, 1e-9, 1e7)
        n = max(n, math.ceil(needed / distance))
    return n


@dataclass(frozen=True)
class ChiSquared:
    statistic: float
    dof: int
    p_value: float


def chi_squared(counts: np.ndarray, p: np.ndarray) -> ChiSquared:
    """Pearson's χ² of ``counts`` against populations ``p`` (len(p) − 1 degrees of freedom)."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != p.shape:
        raise ValueError(f"chi_squared: {len(counts)} bins of counts, {len(p)} populations")
    expected = counts.sum() * p
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    dof = len(p) - 1
    return ChiSquared(statistic, dof, float(stats.chi2.sf(statistic, dof)))


@dataclass(frozen=True)
class PopulationTest:
    target: ChiSquared
    uniform: ChiSquared
    wrong_beta: ChiSquared

    @property
    def passes(self) -> bool:
        """The target fits, and both power checks reject their alternative."""
        return (self.target.p_value >= P_PASS and self.uniform.p_value <= P_FAIL
                and self.wrong_beta.p_value <= P_FAIL)


def population_test(counts: np.ndarray, x: np.ndarray, kT: float,
                    groups: np.ndarray | None = None) -> PopulationTest:
    """χ² of ``counts`` (per configuration) against exp(−x/kT), uniform populations and 1.5 β;
    with ``groups``, counts and populations are pooled per group first."""
    others = alternatives(x, kT)
    counts = pooled(counts, groups)
    return PopulationTest(chi_squared(counts, pooled(boltzmann(x, kT), groups)),
                          chi_squared(counts, pooled(others["uniform"], groups)),
                          chi_squared(counts, pooled(others["wrong_beta"], groups)))


def metropolis_decision(x: np.ndarray, kT: float, c: int, d: int, log_u: float,
                        margin: float = 0.0) -> tuple[bool, bool]:
    """(accepted, near tie) of the move c → d with acceptance number ln u: accepted when
    ln u < −(x_d − x_c)/kT. A near tie, |ln u + (x_d − x_c)/kT| < margin (in units of kT), is a
    decision that energy noise of that size could flip."""
    exponent = (x[d] - x[c]) / kT
    return bool(log_u < -exponent), bool(abs(log_u + exponent) < margin)


def levels(x: np.ndarray, kT: float, tolerance: float = LEVEL_TOLERANCE) -> np.ndarray:
    """A group id per configuration: configurations whose x differ by at most ``tolerance`` kT from
    their neighbour in sorted order share an energy level (symmetry-equivalent arrangements)."""
    order = np.argsort(x, kind="stable")
    breaks = np.diff(x[order]) > tolerance * kT
    groups = np.empty(len(x), dtype=np.int64)
    groups[order] = np.concatenate(([0], np.cumsum(breaks)))
    return groups


def pooled(values: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Counts or populations summed per group (unchanged without groups)."""
    if groups is None:
        return np.asarray(values)
    return np.bincount(groups, weights=np.asarray(values, dtype=np.float64), minlength=int(groups.max()) + 1)


def counts_of(indices: np.ndarray, n_configs: int) -> np.ndarray:
    """Samples per configuration."""
    return np.bincount(np.asarray(indices, dtype=np.int64), minlength=n_configs)


def thinned(sequence: np.ndarray, k: int) -> np.ndarray:
    """Every k-th element, the first after k attempts."""
    return np.asarray(sequence)[k - 1::k]


def sample_thinned_chain(matrix: np.ndarray, k: int, n: int, *, start: int, rng) -> np.ndarray:
    """n states of the exact chain, k steps apart (drawn from P^k), after starting at ``start``."""
    step = np.linalg.matrix_power(matrix, k)
    cumulative = np.cumsum(step, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(n)
    states = np.empty(n, dtype=np.int64)
    state = start
    for t in range(n):
        state = int(np.searchsorted(cumulative[state], uniforms[t], side="right"))
        states[t] = state
    return states


@dataclass(frozen=True)
class Plan:
    """The exact chain of one enumeration and the sampling it needs."""

    x: np.ndarray
    kT: float
    pi: np.ndarray
    targets: np.ndarray
    matrix: np.ndarray
    lambda2: float
    thinning: int
    n_samples: int
    groups: np.ndarray | None = None

    @property
    def attempts(self) -> int:
        """Attempts of the driver for n_samples thinned samples."""
        return self.thinning * self.n_samples


def plan(x: np.ndarray, targets: np.ndarray, kT: float | None = None, *, pool: bool = False) -> Plan:
    """The exact chain for effective energies ``x`` (kT by ``rule_kT`` unless given).

    ``pool=True`` sizes the sample for the test pooled per energy level (``levels``), which needs far
    fewer samples than the test per configuration; the plan's ``groups`` then go to
    ``population_test``.
    """
    kT = rule_kT(x) if kT is None else kT
    pi = boltzmann(x, kT)
    matrix = transition_matrix(x, kT, targets)
    lambda2 = second_eigenvalue(matrix, pi)
    groups = levels(x, kT) if pool else None
    n = sample_size(pooled(pi, groups), [pooled(q, groups) for q in alternatives(x, kT).values()])
    return Plan(x, kT, pi, targets, matrix, lambda2, thinning(lambda2), n, groups)


def pooled_plan(full: Plan) -> Plan:
    """``full`` with the sample sized for the test pooled per energy level."""
    return plan(full.x, full.targets, full.kT, pool=True)


@dataclass(frozen=True)
class Case:
    """One enumeration: the configurations, their reference energies, Δμ and the exact chain."""

    move: str
    sites: np.ndarray
    species: tuple
    configs: np.ndarray
    energies: np.ndarray
    delta_mu: float
    plan: Plan


def swap_case(atoms, species, calculator, sites=None) -> Case:
    """Swaps of species (A, B) over ``sites`` (default all atoms) with the atoms' own A and B counts."""
    sites = np.arange(len(atoms)) if sites is None else np.asarray(sites)
    n_b = int(np.count_nonzero(atoms.numbers[sites] == species[1]))
    configs = swap_configurations(len(sites), n_b)
    energies = reference_energies(atoms, sites, species, configs, calculator)
    return Case("swap", sites, tuple(species), configs, energies, 0.0,
                plan(effective_energies(energies, configs), proposals(configs, "swap")))


def transmute_case(atoms, sites, species, calculator, delta_mu: float | None = None) -> Case:
    """Transmutations between species (A, B) on ``sites``; Δμ_B (configurational, relative to A) by
    ``fitted_delta_mu`` unless given."""
    sites = np.asarray(sites)
    configs = transmute_configurations(len(sites))
    energies = reference_energies(atoms, sites, species, configs, calculator)
    delta_mu = fitted_delta_mu(energies, configs) if delta_mu is None else float(delta_mu)
    return Case("transmute", sites, tuple(species), configs, energies, delta_mu,
                plan(effective_energies(energies, configs, delta_mu), proposals(configs, "transmute")))
