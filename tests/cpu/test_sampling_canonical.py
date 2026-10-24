"""The canonical sampling checks of plan phase 4, choice 8 (4-T13): CSVR's and NHC's kinetic energy against the
canonical gamma distribution, CPU, fixed seeds. Acceptance tier: they run at the phase acceptance, not at merges.

The integrators are driven in the engine's eager order with ASE's Lennard-Jones on a host copy of the state, as
``test_nose_hoover_mtk.py`` does: 108 argon atoms at 150 K (dense, anharmonic), 2 fs steps, the production
conventions (``fixcm=True``, ``dof="auto"``: N_f = 3N - 3, decision 45). In the canonical ensemble the kinetic
energy is Gamma(N_f / 2, k_B T). After 2 ps of equilibration, one sample every 100 fs (five time constants) for
40 ps: 400 samples, compared by a one-sample Kolmogorov-Smirnov test. The statistic and p-value are recorded
(``FASTMACE_RESULTS``); a p-value below 0.001 fails, which the runbook sends to the conductor before the report.
NVT Berendsen, which suppresses the fluctuations, is the negative control: the same test must reject it.
"""

import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.acceptance]

TEMPERATURE_K, DT_FS, TAU_FS = 150.0, 2.0, 20.0
EQUILIBRATION, EVERY, SAMPLES = 1000, 50, 400
P_LIMIT = 1e-3


def _harness():
    spec = importlib.util.spec_from_file_location("_nhc_cpu_harness", Path(__file__).parent / "test_nose_hoover_mtk.py")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


def record(name):
    from fastmace import integrators

    common = dict(timestep_fs=DT_FS, temperature_K=TEMPERATURE_K)
    if name == "csvr":
        return integrators.CSVR(tau_fs=TAU_FS, **common)
    if name == "nhc":
        return integrators.NoseHooverChain(tdamp_fs=TAU_FS, **common)
    return integrators.Berendsen(taut_fs=TAU_FS, **common)


def kinetic_samples(name, seed=20261023):
    from fastmace.integrators.registry import build_integrator

    harness = _harness()
    atoms = harness.argon(seed=seed % 1000, temperature_K=TEMPERATURE_K)
    state = harness.new_state(atoms)
    field = harness.ASEForces(atoms)
    integrator = build_integrator(state, record(name).to_config(seed=seed))
    harness.start(state, integrator, field)
    samples = []
    for step in range(1, EQUILIBRATION + EVERY * SAMPLES + 1):
        harness.md_step(state, integrator, field)
        if step > EQUILIBRATION and (step - EQUILIBRATION) % EVERY == 0:
            samples.append(float(0.5 * (state.masses * state.velocities ** 2).sum()))
    return np.asarray(samples), integrator.degrees_of_freedom(state), len(atoms)


def ks(samples, dof):
    from ase import units
    from scipy import stats

    return stats.kstest(samples, "gamma", args=(0.5 * dof, 0.0, units.kB * TEMPERATURE_K))


def _report(name, summary):
    directory = os.environ.get("FASTMACE_RESULTS")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"4-T13-sampling-{name}.json").write_text(json.dumps(summary, indent=2) + "\n")


@pytest.mark.parametrize("name", ["csvr", "nhc"])
def test_kinetic_energy_follows_the_canonical_gamma_distribution(name):
    samples, dof, n = kinetic_samples(name)
    assert dof == 3 * n - 3  # fixcm=True, dof="auto" (decision 45)
    result = ks(samples, dof)
    from ase import units

    expected_mean, expected_sd = 0.5 * dof * units.kB * TEMPERATURE_K, np.sqrt(0.5 * dof) * units.kB * TEMPERATURE_K
    summary = {"scheme": name, "samples": len(samples), "degrees_of_freedom": int(dof),
               "ks_statistic": float(result.statistic), "p_value": float(result.pvalue),
               "mean_eV": float(samples.mean()), "expected_mean_eV": expected_mean,
               "sd_eV": float(samples.std(ddof=1)), "expected_sd_eV": expected_sd,
               "setting": {"atoms": n, "temperature_K": TEMPERATURE_K, "timestep_fs": DT_FS, "tau_fs": TAU_FS,
                           "equilibration_steps": EQUILIBRATION, "every": EVERY}}
    _report(name, summary)
    print(summary)
    assert result.pvalue >= P_LIMIT, summary


def test_berendsens_suppressed_fluctuations_fail_the_same_test():
    samples, dof, _ = kinetic_samples("berendsen")
    result = ks(samples, dof)
    _report("berendsen-negative-control", {"ks_statistic": float(result.statistic), "p_value": float(result.pvalue),
                                           "sd_eV": float(samples.std(ddof=1))})
    assert result.pvalue < P_LIMIT, (result.statistic, result.pvalue)
