"""``fm.ForceField(polar=PolarConfig(...))``: MACE PR 1762's options for PolarMACE, flagged as not tested."""

import json
import types

import pytest

from fastmace.config import PolarConfig
from fastmace.model.forcefield import ForceField

PR1762 = PolarConfig(fukui_regularisation=0.1, dipole_cap=(0.5, 2.0))


class PolarMACE:  # polar.is_polar goes by the class name
    pass


class ScaleShiftMACE:
    pass


def _loaded(model):
    return types.SimpleNamespace(calculator=None, sha256="0" * 64, model=model, functional=None, label="polar1-s",
                                 head=None, charge=0, spin_multiplicity=1, path="polar.model")


def test_the_default_is_polar_config_and_adds_no_warning():
    ff = ForceField(_loaded(PolarMACE()))
    assert ff.polar == PolarConfig() and ff.polar_warnings == ()


def test_pr1762_options_are_kept_and_flagged_as_not_tested():
    ff = ForceField(_loaded(PolarMACE()), polar=PR1762)
    assert ff.polar == PR1762
    (warning,) = ff.polar_warnings
    assert "not tested" in warning and "PR 1762" in warning


def test_a_non_default_polar_config_needs_a_polarmace_model():
    with pytest.raises(ValueError, match="PolarMACE"):
        ForceField(_loaded(ScaleShiftMACE()), polar=PR1762)
    assert ForceField(_loaded(ScaleShiftMACE()), polar=PolarConfig()).polar == PolarConfig()
    with pytest.raises(TypeError, match="PolarConfig"):
        ForceField(_loaded(PolarMACE()), polar={"fukui_regularisation": 0.1})


def test_the_options_are_part_of_the_identity_only_when_set(monkeypatch):
    plain, stabilised = ForceField(_loaded(PolarMACE())), ForceField(_loaded(PolarMACE()), polar=PR1762)
    for ff in (plain, stabilised):
        ff.species = (11, 17)
        monkeypatch.setattr(ff, "effective_precision", lambda float64_energy=True: ("float32", None))
    assert "polar" not in json.loads(plain.identity.physics)  # unchanged digests for runs without the options
    assert json.loads(stabilised.identity.physics)["polar"] == {"fukui_regularisation": 0.1, "dipole_cap": [0.5, 2.0]}


def test_the_planner_and_the_simulation_record_carry_it():
    from fastmace.planner import minimal
    from fastmace.simulation import Simulation, _record_polar

    ff = ForceField(_loaded(PolarMACE()), polar=PR1762)
    assert minimal.default_planner(ff).polar == PR1762
    ff.species = (11, 17)
    record = Simulation._force_field_record(types.SimpleNamespace(force_field=ff))
    assert record["polar"] == {"fukui_regularisation": 0.1, "dipole_cap": [0.5, 2.0], "kspace_headroom": "auto"}
    assert _record_polar(record) == PR1762
    plain = ForceField(_loaded(PolarMACE()))
    plain.species = (11, 17)
    # absent at the default, so older records are unchanged
    assert "polar" not in Simulation._force_field_record(types.SimpleNamespace(force_field=plain))
    assert _record_polar({"precision": "float32"}) is None
