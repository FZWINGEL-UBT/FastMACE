"""The exception hierarchy and the message shape (SPEC §7.6 "Exceptions", §7.7 "Errors"; plan task 1b-T02)."""

import pickle
import warnings

import pytest

from fastmace import errors
from fastmace.errors import FastMACEError, PlanRefused, Reason, SpeciesNotInSuperset

ENVIRONMENT_ERRORS = (errors.HybridUnavailableError, errors.BackendError, errors.NaiveCueqError,
                      errors.ProcessConstraintError, errors.CacheLocationError)
SPEC_ERRORS = (errors.PlanRefused, errors.SpeciesNotInSuperset, errors.GateFailed, errors.GateUnavailable,
               errors.ModelElementError, errors.RestartMismatch, errors.ProcessConstraintError,
               errors.CheckpointTrustError, errors.NonFiniteError)
FROM_1A = (errors.HybridUnavailableError, errors.BackendError, errors.NaiveCueqError, errors.RewriteError,
           errors.RewriteRefusedError, errors.SelfCheckError, errors.PolarUnavailableError,
           errors.CacheLocationError, errors.BaselineExportError, errors.OEqRuntimeError)
ALL_ERRORS = tuple(dict.fromkeys(SPEC_ERRORS + FROM_1A))


def build(cls, text="something went wrong"):
    """One instance of ``cls`` built the way its call sites build it."""
    if issubclass(cls, PlanRefused):
        return cls([text])
    return cls(text)


# --------------------------------------------------------------------------- hierarchy

@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_every_exception_derives_from_fastmace_error(cls):
    assert issubclass(cls, FastMACEError)


def test_species_not_in_superset_is_caught_as_plan_refused():
    with pytest.raises(PlanRefused) as caught:
        raise SpeciesNotInSuperset([Reason("Xe is not in the superset", rule=2, code="species-not-in-superset")])
    assert type(caught.value) is SpeciesNotInSuperset


def test_builtin_bases_of_1a_are_kept():
    assert issubclass(PlanRefused, ValueError)
    assert issubclass(errors.ProcessConstraintError, RuntimeError)
    assert issubclass(errors.OEqRuntimeError, RuntimeError)
    assert issubclass(errors.RewriteRefusedError, errors.RewriteError)
    assert issubclass(errors.SelfCheckError, errors.RewriteError)
    assert issubclass(errors.ModelLicenceWarning, UserWarning)
    assert not issubclass(errors.ModelLicenceWarning, FastMACEError)


def test_old_import_paths_give_the_same_class():
    from fastmace import _cache
    from fastmace.bench import ab
    from fastmace.engine import constraints
    from fastmace.kernels import _oeq_runtime
    from fastmace.longrange import polar
    from fastmace.model import loading
    from fastmace.planner import fixed
    from fastmace.rewrites import registry

    pairs = [(loading.CheckpointTrustError, errors.CheckpointTrustError),
             (loading.HybridUnavailableError, errors.HybridUnavailableError),
             (loading.BackendError, errors.BackendError),
             (loading.NaiveCueqError, errors.NaiveCueqError),
             (loading.ModelLicenceWarning, errors.ModelLicenceWarning),
             (fixed.PlanRefused, errors.PlanRefused),
             (constraints.ProcessConstraintError, errors.ProcessConstraintError),
             (registry.RewriteError, errors.RewriteError),
             (registry.RewriteRefusedError, errors.RewriteRefusedError),
             (registry.SelfCheckError, errors.SelfCheckError),
             (polar.PolarUnavailableError, errors.PolarUnavailableError),
             (_cache.CacheLocationError, errors.CacheLocationError),
             (ab.BaselineExportError, errors.BaselineExportError),
             (_oeq_runtime.OEqRuntimeError, errors.OEqRuntimeError)]
    for old, new in pairs:
        assert old is new, old.__qualname__


def test_errors_module_imports_neither_torch_nor_mace():
    import subprocess
    import sys

    code = ("import sys, fastmace.errors; "
            "print(sorted(m for m in ('torch', 'mace', 'numpy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"


# --------------------------------------------------------------------------- reasons

def test_reason_is_a_string_with_rule_and_code():
    reason = Reason("constraints are not supported", rule=1, code="constraints")
    assert reason == "constraints are not supported"
    assert isinstance(reason, str)
    assert (reason.rule, reason.code) == (1, "constraints")
    assert Reason("a run request", code="schedule-above-max-temperature").rule is None


def test_reason_survives_pickling():
    reason = pickle.loads(pickle.dumps(Reason("x", rule=2, code="species-not-in-superset")))
    assert (str(reason), reason.rule, reason.code) == ("x", 2, "species-not-in-superset")


def test_one_refusal_with_three_reasons_is_one_exception_with_three_reasons():
    reasons = [Reason("the structure is not periodic in z", rule=1, code="non-3d-pbc"),
               Reason("PolarMACE needs the hybrid or eager cuEq", rule=3, code="polar-plain-compiled-cueq"),
               Reason("reproducible=True: no family qualified (phase 2)", rule=5, code="reproducible")]
    with pytest.raises(PlanRefused) as caught:
        raise PlanRefused.from_reasons(reasons)
    error = caught.value
    assert type(error) is PlanRefused
    assert error.reasons == tuple(reasons)
    assert error.codes == ("non-3d-pbc", "polar-plain-compiled-cueq", "reproducible")
    assert error.rules == (1, 3, 5)
    for reason in reasons:
        assert reason in str(error)


def test_rule_2_among_the_reasons_gives_the_subclass_with_every_reason():
    reasons = [Reason("constraints are not supported", rule=1, code="constraints"),
               Reason("Xe is not in the ForceField's superset", rule=2, code="species-not-in-superset")]
    error = PlanRefused.from_reasons(reasons)
    assert type(error) is SpeciesNotInSuperset
    assert error.reasons == tuple(reasons)


def test_plain_string_reasons_of_1a_still_work():
    error = PlanRefused(["streaming_conv needs sort_receivers", "zero forces"])
    assert error.reasons == ("streaming_conv needs sort_receivers", "zero forces")
    assert error.codes == (None, None)
    assert str(error).splitlines()[0] == "plan refused: streaming_conv needs sort_receivers; zero forces"
    assert type(PlanRefused.from_reasons(["x"])) is PlanRefused


# --------------------------------------------------------------------------- message shape

def test_message_shape_has_every_part_in_order():
    error = FastMACEError("the run stopped at step 120", why="The potential energy is NaN.",
                          next_step="Lower the timestep.", record="metadata.json", environment=True)
    lines = str(error).splitlines()
    assert lines == ["the run stopped at step 120",
                     "Why: The potential energy is NaN.",
                     "Next: Lower the timestep.",
                     "Record: metadata.json",
                     "Environment: run `fastmace doctor` to check this installation."]
    assert (error.what, error.why, error.next_step, error.record, error.environment) == (
        "the run stopped at step 120", "The potential energy is NaN.", "Lower the timestep.", "metadata.json", True)


def test_one_argument_construction_still_works():
    error = FastMACEError("plain message")
    assert str(error) == "plain message"
    assert error.args == ("plain message",)
    assert (error.why, error.next_step, error.record, error.environment) == (None, None, None, False)


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_message_shape_for_each_class(cls):
    error = build(cls)
    lines = str(error).splitlines()
    assert "something went wrong" in lines[0]
    assert lines[0] == error.what
    for line in lines[1:]:
        assert line.split(":", 1)[0] in ("Why", "Next", "Record", "Environment")
    assert error.next_step, f"{cls.__name__} says what to do next"
    assert ("fastmace doctor" in str(error)) is (cls in ENVIRONMENT_ERRORS)


@pytest.mark.parametrize("cls", ENVIRONMENT_ERRORS, ids=lambda c: c.__name__)
def test_doctor_line_can_be_switched_off(cls):
    assert "fastmace doctor" not in str(cls("x", environment=False))
    assert "fastmace doctor" in str(errors.ModelElementError("x", environment=True))


def test_explicit_parts_override_class_defaults():
    error = errors.CheckpointTrustError("bad file", why="It is a pickle.", next_step="Pass sha256=.", record="r")
    assert str(error).splitlines() == ["bad file", "Why: It is a pickle.", "Next: Pass sha256=.", "Record: r"]


@pytest.mark.parametrize("cls", (errors.GateFailed, errors.GateUnavailable), ids=lambda c: c.__name__)
def test_gate_errors_carry_the_gate_record_and_its_location(cls):
    gate_record = {"status": "failed", "energy_ratio": 3.2}
    error = cls("the gate did not pass", gate_record=gate_record)
    assert error.gate_record is gate_record
    assert error.record == "run.gate"
    assert "Record: run.gate" in str(error)
    assert cls("x", gate_record=gate_record, record="metadata.json").record == "metadata.json"
    assert cls("x").gate_record is None


def test_reason_attribute_of_1a_is_kept():
    assert errors.HybridUnavailableError("mace was imported before").reason == "mace was imported before"
    assert "hybrid is unavailable: mace was imported before" in str(
        errors.HybridUnavailableError("mace was imported before"))
    assert errors.PolarUnavailableError("no graph_longrange").reason == "no graph_longrange"


@pytest.mark.parametrize("cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_exceptions_survive_pickling(cls):
    error = cls("x", why="w", next_step="n", record="r", environment=True) if not issubclass(cls, PlanRefused) \
        else cls.from_reasons([Reason("x", rule=2, code="c")], why="w", next_step="n", record="r", environment=True)
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert str(copy) == str(error)
    assert copy.__dict__ == error.__dict__


def test_gate_record_survives_pickling():
    copy = pickle.loads(pickle.dumps(errors.GateFailed("x", gate_record={"status": "failed"})))
    assert copy.gate_record == {"status": "failed"}


def test_licence_warning_is_a_plain_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("CC BY-NC", errors.ModelLicenceWarning)
    assert caught[0].category is errors.ModelLicenceWarning
