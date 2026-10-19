"""FastMACE exceptions (SPEC §7.6 "Exceptions") and their message shape (§7.7 "Errors").

Every exception FastMACE raises derives from ``FastMACEError``. Its message has
one shape, so that no traceback needs reading: what happened; why, in one or
two sentences; what to do next; where the record is (``run.plan``,
``run.gate``, ``metadata.json``); and ``fastmace doctor`` for environment
problems. The CLI prints this message alone (§7.5).

This module imports nothing heavy: ``import fastmace`` imports neither torch
nor MACE, and the 1a modules that defined these classes re-export them.
"""

DOCTOR_LINE = "Environment: run `fastmace doctor` to check this installation."


def _restore(cls, args, state):
    """Unpickle an exception without calling its ``__init__`` (whose signature differs per class)."""
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


class FastMACEError(Exception):
    """Base class of every error FastMACE raises.

    ``FastMACEError(what, why=None, next_step=None, record=None, environment=None)``;
    a plain one-argument construction gives just ``what``. Each class may set a
    default next step, record location and environment flag; ``None`` takes the
    class default.
    """

    default_next_step = None
    default_record = None
    default_environment = False

    def __init__(self, what, why=None, next_step=None, record=None, environment=None):
        super().__init__(str(what))
        self.what = str(what)
        self.why = why
        self.next_step = self.default_next_step if next_step is None else next_step
        self.record = self.default_record if record is None else record
        self.environment = self.default_environment if environment is None else bool(environment)

    def __str__(self):
        lines = [self.what]
        if self.why:
            lines.append(f"Why: {self.why}")
        if self.next_step:
            lines.append(f"Next: {self.next_step}")
        if self.record:
            lines.append(f"Record: {self.record}")
        if self.environment:
            lines.append(DOCTOR_LINE)
        return "\n".join(lines)

    def __reduce__(self):
        return _restore, (type(self), self.args, dict(self.__dict__))


# --------------------------------------------------------------------------- planning

class Reason(str):
    """One refusal reason: the text, the §9.3 hard rule (``None`` for a ``run`` request) and a stable code.

    Reports and tests match ``code``, never the text.
    """

    def __new__(cls, text, *, rule=None, code):
        reason = super().__new__(cls, text)
        reason.rule = rule
        reason.code = code
        return reason

    def __reduce__(self):
        return _new_reason, (str(self), self.rule, self.code)


def _new_reason(text, rule, code):
    return Reason(text, rule=rule, code=code)


class PlanRefused(FastMACEError, ValueError):
    """The planner cannot plan this workload, or a ``run`` request the plan does not cover.

    ``reasons`` lists every hard-constraint reason of the refusal at once (SPEC
    §9.3): plain strings (1a) or ``Reason`` records. Raise through
    ``PlanRefused.from_reasons``, which picks ``SpeciesNotInSuperset`` when rule
    2 is among them.
    """

    default_next_step = "Change the inputs each reason names; the plan is never relaxed silently (SPEC §9.3)."

    def __init__(self, reasons, why=None, next_step=None, record=None, environment=None):
        self.reasons = tuple(reasons)
        super().__init__("plan refused: " + "; ".join(self.reasons), why, next_step, record, environment)

    @classmethod
    def from_reasons(cls, reasons, **parts):
        """One exception carrying every reason: ``SpeciesNotInSuperset`` if a reason has ``rule == 2``."""
        reasons = tuple(reasons)
        target = cls
        if any(getattr(reason, "rule", None) == 2 for reason in reasons) and not issubclass(cls, SpeciesNotInSuperset):
            target = SpeciesNotInSuperset
        return target(reasons, **parts)

    @property
    def codes(self):
        """The stable code of each reason (``None`` for a plain-string reason)."""
        return tuple(getattr(reason, "code", None) for reason in self.reasons)

    @property
    def rules(self):
        """The §9.3 rule of each reason (``None`` where none applies)."""
        return tuple(getattr(reason, "rule", None) for reason in self.reasons)


class SpeciesNotInSuperset(PlanRefused):
    """Rule 2: a species outside the ``ForceField``'s declared superset (SPEC §8.5), with every reason."""

    default_next_step = ("Declare every species the run can meet in fm.ForceField(species=...), "
                         "and fix the other reasons listed.")


class ProcessConstraintError(FastMACEError, RuntimeError):
    """An engine would break a process-wide constraint (SPEC §8.7)."""

    default_next_step = "Run this plan in a separate process (a fresh Python process or the subprocess runner)."
    default_environment = True


class FallbackExhausted(PlanRefused):
    """No fallback candidate passed validation (SPEC §9.4 step 5); ``revisions`` holds every tried plan.

    ``revisions`` is the revision chain of the fallback, oldest first: the failed plan, then each
    candidate tried, each with its reason. ``reasons`` are the refusal's, as for ``PlanRefused``.
    """

    default_next_step = ("Read the tried candidates and their reasons in the record; pass fallback=FallbackConfig("
                         "fallback='refuse') to see the first failure alone, and report the model and structure.")
    default_record = "run.plan (its fallbacks and revision chain) and metadata.json"

    def __init__(self, reasons, why=None, next_step=None, record=None, environment=None, *, revisions=()):
        super().__init__(reasons, why, next_step, record, environment)
        self.revisions = tuple(revisions)


class SubprocessRunnerError(FastMACEError, RuntimeError):
    """The subprocess runner's child failed, was killed or stalled (SPEC §8.7); never a hang.

    ``exit_code`` is the child's (negative: killed by that signal; ``None``: still running when it
    was stopped), ``child_message`` its error message, and ``directory`` the hand-off directory,
    kept after an error.
    """

    default_next_step = ("Read the child's message; the hand-off directory holds its restart and log. Resume from "
                         "the last restart point, in a fresh process if the constraint allows.")
    default_record = "the hand-off directory and metadata.json"

    def __init__(self, what, why=None, next_step=None, record=None, environment=None, *, exit_code=None,
                 child_message=None, directory=None):
        super().__init__(what, why, next_step, record, environment)
        self.exit_code = exit_code
        self.child_message = child_message
        self.directory = None if directory is None else str(directory)


class IsolationError(FastMACEError, RuntimeError):
    """An instance whose lease on a shared component was taken by another instance (SPEC §8.7).

    ``component`` names the shared component, ``holder`` the instance that holds its lease now.
    """

    default_next_step = "Call rebind() on this force field or engine before using it again (SPEC §8.7)."

    def __init__(self, what, why=None, next_step=None, record=None, environment=None, *, component=None,
                 holder=None):
        super().__init__(what, why, next_step, record, environment)
        self.component = component
        self.holder = holder


class ReproducibilityError(FastMACEError):
    """A (backend, model family) failed qualification for ``reproducible=True`` (SPEC §9.3 rule 5).

    ``blocking`` names the kernels or operators without a deterministic implementation. A plan that
    asks for ``reproducible=True`` without a qualified family is refused by ``PlanRefused`` (rule 5),
    not by this error.
    """

    default_next_step = ("Run with reproducible=False (results compared statistically), or report the blocking "
                         "kernels.")
    default_record = "the qualification record (planner.reproducible) and metadata.json"

    def __init__(self, what, why=None, next_step=None, record=None, environment=None, *, blocking=()):
        super().__init__(what, why, next_step, record, environment)
        self.blocking = tuple(str(name) for name in blocking)


# --------------------------------------------------------------------------- the gate

class _GateError(FastMACEError):
    """A gate result that stops the run; ``gate_record`` is the record, ``record`` where it is."""

    default_record = "run.gate"

    def __init__(self, what, why=None, next_step=None, record=None, environment=None, gate_record=None):
        super().__init__(what, why, next_step, record, environment)
        self.gate_record = gate_record


class GateFailed(_GateError):
    """The gate on the user's structure failed (SPEC §12.3); ``gate_record`` holds the comparison."""

    default_next_step = "Read the failed quantities in the gate record; the plan's fallback is SPEC §9.4."


class GateUnavailable(_GateError):
    """The gate's reference could not run on any allowed device (SPEC §12.3)."""

    default_next_step = ("Free GPU memory or allow CPU float64, or pass check='allow_unavailable' "
                         "to run without the gate, recorded.")


# --------------------------------------------------------------------------- model, restart, runtime

class ModelElementError(FastMACEError):
    """A declared superset species has no embedding in the model (SPEC §8.5)."""

    default_next_step = "Remove that species from species=, or use a model trained on it."


class RestartMismatch(FastMACEError):
    """A restart's force-field identity differs, or its plan changed without ``allow_plan_change`` (§10.6)."""

    default_next_step = ("Resume with the force field the restart was written with, or pass allow_plan_change=True "
                         "for a plan change.")
    default_record = "the restart directory's identity and plan records"


class CheckpointTrustError(FastMACEError):
    """A checkpoint would be unpickled without a matching SHA-256 or ``trusted=True`` (SPEC §7.6)."""

    default_next_step = ("Pass sha256= with the file's published hash, or trusted=True for a file you trust; "
                         "a checkpoint is a pickle and runs code when loaded.")


class NonFiniteError(FastMACEError):
    """A non-finite energy or observable stopped the run (SPEC §9.6); it is never "fixed"."""

    default_next_step = "Check the structure and the timestep, and resume from the last restart point."
    default_record = "metadata.json"


# --------------------------------------------------------------------------- backends (1a)

class HybridUnavailableError(FastMACEError):
    """The cuEq/OEq hybrid cannot be built in this process; ``reason`` says why (SPEC §9.4 falls back)."""

    default_next_step = "Import fastmace before mace, or install OpenEquivariance for this torch and CUDA."
    default_environment = True

    def __init__(self, reason, why=None, next_step=None, record=None, environment=None):
        super().__init__(f"the cuEquivariance/OpenEquivariance hybrid is unavailable: {reason}",
                         why, next_step, record, environment)
        self.reason = reason


class BackendError(FastMACEError):
    """The requested backend was not built as asked (missing package, failed conversion)."""

    default_next_step = "Install the backend's packages, or choose another backend."
    default_environment = True


class NaiveCueqError(FastMACEError):
    """cuEquivariance fell back to its naive kernels (MACE PR 1614; SPEC §9.2, §9.3 rule 6)."""

    default_next_step = ("Install cuEquivariance's fused kernels for this GPU, or pass allow_naive_cueq=True "
                         "to run slowly, recorded.")
    default_environment = True


class PolarUnavailableError(FastMACEError):
    """The GPU-only PolarMACE path cannot be used here; ``reason`` says why."""

    default_next_step = "Use the qualified MACE and graph_longrange versions, or a non-polar model."

    def __init__(self, reason, why=None, next_step=None, record=None, environment=None):
        super().__init__(f"the GPU-only PolarMACE path is unavailable: {reason}", why, next_step, record, environment)
        self.reason = reason


class OEqRuntimeError(FastMACEError, RuntimeError):
    """The OEq host-extension runtime cannot be prepared."""

    default_next_step = "Prepare the OEq runtime before importing openequivariance or mace (import fastmace first)."


class CacheLocationError(FastMACEError):
    """The cache root is not usable (inside sys.prefix)."""

    default_next_step = "Set FASTMACE_CACHE to a writable directory outside the Python environment."
    default_environment = True


# --------------------------------------------------------------------------- rewrites (1a)

class RewriteError(FastMACEError):
    """A rewrite could not be resolved or applied."""

    default_next_step = "Turn that rewrite off (rewrites=...) and report the model; the others stay exact."


class RewriteRefusedError(RewriteError):
    """An explicitly requested rewrite (``"on"``) cannot be enabled; the message gives the reasons."""

    default_next_step = "Set the refused rewrites to \"auto\" or \"off\"."


class SelfCheckError(RewriteError):
    """A rewrite's self-check failed at setup (SPEC §6.4)."""


# --------------------------------------------------------------------------- benchmarks

class BaselineExportError(FastMACEError):
    """A baseline directory is not a usable export of the research engine."""

    default_next_step = ("Export ResearchModule's tag research-engine-baseline-20261017 with git archive "
                         "(AGENTS.md, \"Baseline for A/B\").")


# --------------------------------------------------------------------------- warnings

class ModelLicenceWarning(UserWarning):
    """The licence of a loaded model (SPEC §4.3: shown at load time)."""


__all__ = ["FastMACEError", "Reason", "PlanRefused", "SpeciesNotInSuperset", "ProcessConstraintError",
           "FallbackExhausted", "SubprocessRunnerError", "IsolationError", "ReproducibilityError",
           "GateFailed", "GateUnavailable", "ModelElementError", "RestartMismatch", "CheckpointTrustError",
           "NonFiniteError", "HybridUnavailableError", "BackendError", "NaiveCueqError", "PolarUnavailableError",
           "OEqRuntimeError", "CacheLocationError", "RewriteError", "RewriteRefusedError", "SelfCheckError",
           "BaselineExportError", "ModelLicenceWarning"]
