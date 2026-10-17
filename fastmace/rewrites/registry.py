"""The rewrite registry: records, order, qualification, resolution and self-checks (SPEC §6.1, §6.3, §6.4).

Every model rewrite is one ``Rewrite`` record (exactly SPEC §6.1). A ``Registry`` holds records
together with the ``RewriteConfig`` field that switches each one, and provides:

- **order** (``Registry.order``): registration order, except that every record comes after the
  records it ``requires``. ``RECORD_SOURCES`` fixes the registration order of the default
  registry, so the families register in the order the research engine applies them;
- **qualification** (``Registry.qualification``, SPEC §6.3): the record's ``source_hashes`` are
  checked against the installed source (sha256 of ``inspect.getsource``, as the research engine
  pins them). A mismatch, or a source that cannot be found, disables the record, and the reason
  is logged on the ``fastmace.rewrites`` logger. Outside ``mace_versions`` the hashes decide; the
  version only appears in the reason;
- **candidates** (``Registry.candidates``): per record, in order, everything the planner (1a-T15)
  needs to resolve ``auto``: the config field, default, exactness, the model verdict
  (``applies_to``), the qualification, ``requires``, ``conflicts`` and workload constraints;
- **resolution** (``Registry.resolve``): turns per-record requests (``"on"``, ``"off"``,
  ``"auto"``, from a ``RewriteConfig`` or a mapping) into enabled or disabled decisions with
  reasons. ``"auto"`` means the record's default; an ``approximate`` record is never enabled by
  default (SPEC §6.1), and cannot even be built with a default other than ``"off"``. A record is
  disabled when a record it requires is not enabled or a conflicting one is. An explicit ``"on"``
  that cannot be honoured is refused (``RewriteRefusedError``), never silently downgraded;
- **application and self-checks** (``Registry.apply``, ``Registry.selfcheck``, SPEC §6.4): the
  enabled records are applied in order, each followed by its self-check against the original
  model; the runner holds every result to the §6.4 threshold of its dtype (``THRESHOLDS``)
  whatever the record claims, and a failure raises ``SelfCheckError``.

Adding a family (plan tasks 1a-T07, 1a-T08): put a module or package whose ``RECORDS`` maps
names to ``Rewrite`` records (or to dicts with the §6.1 fields plus an optional
``config_field``) into ``RECORD_SOURCES``, at the position where the research engine applies it.
A record's config field defaults to its name; a record whose name differs from its field (a
``GENERIC_*`` variant) names it with ``config_field``.

The module holds no state: ``default_registry()`` builds a fresh registry on every call.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import inspect
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from fastmace.config import RewriteConfig
from fastmace.errors import FastMACEError
from fastmace.model.capabilities import RewriteSupport

Support = RewriteSupport
Exactness = Literal["exact", "rounding", "approximate"]
Default = Literal["on", "off", "auto"]
EXACTNESS = ("exact", "rounding", "approximate")
DEFAULTS = ("on", "off", "auto")

# SPEC §6.4 per-module thresholds: relative error max |rewritten - original| / max |original|.
THRESHOLDS = MappingProxyType({"float64": 1e-10, "float32": 1e-4})

# Families registered by default_registry(), in application order (the research engine applies
# the MH-1 rewrites, then the GENERIC_* ones, then the block linears, then the merge).
RECORD_SOURCES = (
    # The research engine's order (compile_model): MH-1 package, then GENERIC_*, then the linear family.
    "fastmace.rewrites.mh1",
    "fastmace.rewrites.generic",
    "fastmace.rewrites.linear",
    "fastmace.rewrites.polar",
    # Process-wide stress-path patch (MACE PR 1498), as a record so the planner gets a decision and reason.
    "fastmace.rewrites.engine",
)

LOGGER = logging.getLogger("fastmace.rewrites")

_CONFIG_FIELDS = tuple(f.name for f in dataclasses.fields(RewriteConfig))


class RewriteError(FastMACEError):
    """A rewrite could not be resolved or applied."""


class RewriteRefusedError(RewriteError):
    """An explicitly requested rewrite (``"on"``) cannot be enabled; the message gives the reasons."""


class SelfCheckError(RewriteError):
    """A rewrite's self-check failed at setup (SPEC §6.4)."""


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one record's self-check (SPEC §6.4)."""

    passed: bool
    max_relative_error: float
    threshold: float
    modules: int  # how many modules were compared
    detail: str = "ok"
    dtype: str | None = None  # "float64" or "float32"; the runner then holds it to THRESHOLDS[dtype]

    @classmethod
    def coerce(cls, value):
        """A CheckResult from a CheckResult or a dict with its fields (as ``rewrites.polar`` returns)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            names = {f.name for f in dataclasses.fields(cls)}
            return cls(**{key: item for key, item in value.items() if key in names})
        raise TypeError(f"a self-check returned {type(value).__name__}, not a CheckResult or a dict")


@dataclass(frozen=True)
class Context:
    """What a record's ``apply`` and ``selfcheck`` may read besides the model.

    ``inputs`` carries structure-dependent arguments (1a-T07's reassociation takes the
    structure's node attributes); ``applied`` lists the records already applied, in order.
    """

    config: RewriteConfig = field(default_factory=RewriteConfig)
    applied: tuple[str, ...] = ()
    inputs: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0  # self-check inputs come from a torch.Generator with this seed


def _specifier_set(value):
    from packaging.specifiers import SpecifierSet

    return value if isinstance(value, SpecifierSet) else SpecifierSet(str(value))


@dataclass(frozen=True)
class Rewrite:
    """One model rewrite (SPEC §6.1)."""

    name: str
    applies_to: Callable[[Any], Support]  # (supported, reason)
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]
    mace_versions: Any  # packaging SpecifierSet; a string is converted
    source_hashes: dict[str, str]  # "module:Qualified.name" -> sha256 of inspect.getsource
    exactness: Exactness
    default: Default
    workload_constraints: tuple[str, ...]
    apply: Callable[[Any, Context], Any]
    selfcheck: Callable[[Any, Any, Context], Any]  # (original, rewritten, context) -> CheckResult

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Rewrite.name must be a non-empty string, got {self.name!r}")
        for name in ("requires", "conflicts", "workload_constraints"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        object.__setattr__(self, "source_hashes", {str(k): str(v) for k, v in dict(self.source_hashes).items()})
        object.__setattr__(self, "mace_versions", _specifier_set(self.mace_versions))
        if self.exactness not in EXACTNESS:
            raise ValueError(f"Rewrite {self.name}: exactness {self.exactness!r} is not one of {EXACTNESS}")
        if self.default not in DEFAULTS:
            raise ValueError(f"Rewrite {self.name}: default {self.default!r} is not one of {DEFAULTS}")
        if self.exactness == "approximate" and self.default != "off":
            raise ValueError(f"Rewrite {self.name}: approximate rewrites are never default (SPEC §6.1); "
                             f"default must be 'off', not {self.default!r}")
        if self.name in self.requires or self.name in self.conflicts:
            raise ValueError(f"Rewrite {self.name} requires or conflicts with itself")
        both = sorted(set(self.requires) & set(self.conflicts))
        if both:
            raise ValueError(f"Rewrite {self.name} requires and conflicts with {both}")
        for name in ("applies_to", "apply", "selfcheck"):
            if not callable(getattr(self, name)):
                raise ValueError(f"Rewrite {self.name}: {name} is not callable")

    @classmethod
    def from_dict(cls, mapping):
        """(Rewrite, config field or None) from a dict of the §6.1 fields plus an optional ``config_field``."""
        names = [f.name for f in dataclasses.fields(cls)]
        unknown = set(mapping) - set(names) - {"config_field"}
        if unknown:
            raise ValueError(f"unknown Rewrite fields {sorted(unknown)}")
        missing = set(names) - set(mapping)
        if missing:
            raise ValueError(f"Rewrite {mapping.get('name')!r} lacks the fields {sorted(missing)}")
        config_field = mapping.get("config_field")
        if config_field is not None:
            config_field = str(config_field).removeprefix("RewriteConfig.")
        return cls(**{name: mapping[name] for name in names}), config_field


# --------------------------------------------------------------------------- qualification


def _resolve(qualified):
    module_name, _, attribute = qualified.partition(":")
    if not attribute:
        raise ValueError(f"{qualified!r} is not 'module:Qualified.name'")
    obj = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def source_sha256(qualified):
    """sha256 of ``inspect.getsource`` of ``"module:Qualified.name"`` (the research engine's pins).

    When FastMACE's own recorded patch replaced that object (``fastmace._patches``), MACE's preserved
    original is hashed, so a record stays qualified after its patch is applied. Any other
    replacement is hashed as found and therefore fails its pin.
    """
    from fastmace import _patches

    obj = _resolve(qualified)
    original = _patches.original_if_patched(qualified, obj)
    return hashlib.sha256(inspect.getsource(obj if original is None else original).encode()).hexdigest()


def installed_mace_version():
    """The installed mace-torch version, or None."""
    import importlib.metadata

    try:
        return importlib.metadata.version("mace-torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def qualify(rewrite, mace_version=None):
    """(qualified, reason) of one record against the installed source (SPEC §6.3).

    Every pinned source must hash to its pin; the MACE version only shapes the reason.
    """
    version = installed_mace_version() if mace_version is None else str(mace_version)
    inside = version is not None and rewrite.mace_versions.contains(version, prereleases=True)
    for qualified, expected in rewrite.source_hashes.items():
        try:
            digest = source_sha256(qualified)
        except Exception as error:  # ImportError, AttributeError, OSError (no source), ...
            return Support(False, f"source {qualified} cannot be read on MACE {version} "
                                  f"({type(error).__name__}: {error})")
        if digest != expected:
            return Support(False, f"source {qualified} differs on MACE {version} "
                                  f"(sha256 {digest[:12]}..., pinned {expected[:12]}...)")
    if inside:
        return Support(True, "ok")
    if rewrite.source_hashes:
        return Support(True, f"MACE {version} is outside {rewrite.mace_versions}, but every pinned source matches")
    return Support(True, f"MACE {version} is outside {rewrite.mace_versions}; no source is pinned")


# --------------------------------------------------------------------------- self-check helpers


def threshold_for(dtype):
    """The SPEC §6.4 per-module threshold for a torch dtype or its name."""
    name = str(dtype).removeprefix("torch.")
    if name not in THRESHOLDS:
        raise ValueError(f"no SPEC §6.4 self-check threshold for {name} (only {sorted(THRESHOLDS)})")
    return THRESHOLDS[name]


def relative_error(got, expected):
    """max |got - expected| / max |expected| (1 when the reference is all zero); inf for a shape change."""
    if tuple(got.shape) != tuple(expected.shape):
        return math.inf
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    error = float((got - expected).abs().max()) if expected.numel() else 0.0
    return error / (scale if scale > 0 else 1.0)


def check_tensors(cases, dtype=None):
    """CheckResult over ``(name, expected, got)`` triples at the threshold of their dtype.

    ``dtype`` defaults to the dtype of the first expected tensor. Each name counts as one module.
    """
    cases = list(cases)
    if not cases:
        return CheckResult(True, 0.0, threshold_for(dtype or "float64"), 0, "nothing to check",
                           str(dtype or "float64").removeprefix("torch."))
    dtype = str(dtype or cases[0][1].dtype).removeprefix("torch.")
    threshold = threshold_for(dtype)
    worst, problems, names = 0.0, [], []
    for name, expected, got in cases:
        error = relative_error(got, expected)
        worst = max(worst, error)
        if name not in names:
            names.append(name)
        if not error <= threshold:
            problems.append(f"{name}: relative error {error:.3g} > {threshold:g}")
    return CheckResult(not problems, worst, threshold, len(names), "; ".join(problems) or "ok", dtype)


# --------------------------------------------------------------------------- planner-facing records


@dataclass(frozen=True)
class Candidate:
    """One record as the planner sees it for a model (the input of 1a-T15's ``auto`` resolution)."""

    name: str
    config_field: str
    default: Default
    exactness: Exactness
    support: Support  # applies_to(model)
    qualified: Support  # source hashes and version (SPEC §6.3)
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]
    workload_constraints: tuple[str, ...]

    @property
    def available(self):
        """Supported by the model and qualified by the installed source."""
        return self.support.supported and self.qualified.supported


@dataclass(frozen=True)
class Decision:
    """The resolution of one record: requested value, outcome and reason."""

    name: str
    requested: str  # "on", "off" or "auto"
    enabled: bool
    reason: str
    refused: bool = False  # an explicit "on" that could not be honoured (strict=False only)


@dataclass(frozen=True)
class Resolution:
    """Decisions for every registered record, in application order."""

    decisions: tuple[Decision, ...]

    @property
    def enabled(self):
        return tuple(d.name for d in self.decisions if d.enabled)

    def decision(self, name):
        for d in self.decisions:
            if d.name == name:
                return d
        raise KeyError(name)

    def reasons(self):
        """{name: reason} for every record."""
        return {d.name: d.reason for d in self.decisions}


@dataclass(frozen=True)
class AppliedRewrite:
    """One applied record: its self-check and whatever its ``apply`` returned."""

    name: str
    result: CheckResult | None  # None with check=False
    detail: Any = None


def _toggle(name, value):
    if isinstance(value, bool):  # RewriteConfig's bool fields (polar_uvu): True asks for the default
        return "auto" if value else "off"
    if value not in DEFAULTS:
        raise ValueError(f"rewrite {name}: {value!r} is not one of {DEFAULTS}")
    return value


# --------------------------------------------------------------------------- the registry


class Registry:
    """Rewrite records with their config fields; see the module docstring."""

    def __init__(self, records=()):
        self._records = {}  # name -> Rewrite, in registration order
        self._fields = {}  # name -> RewriteConfig field
        for record in records:
            self.register(record)

    # ---- contents

    def register(self, rewrite, config_field=None):
        """Add a record (a Rewrite or a dict of its fields); returns the Rewrite.

        ``config_field`` is the ``RewriteConfig`` field that switches it (default: a dict's
        ``config_field``, else the record's name).
        """
        if isinstance(rewrite, Mapping):
            rewrite, from_dict = Rewrite.from_dict(rewrite)
            config_field = config_field or from_dict
        if not isinstance(rewrite, Rewrite):
            raise TypeError(f"cannot register {type(rewrite).__name__}; expected a Rewrite or a dict")
        if rewrite.name in self._records:
            raise ValueError(f"rewrite {rewrite.name} is already registered")
        config_field = str(config_field or rewrite.name).removeprefix("RewriteConfig.")
        if config_field not in _CONFIG_FIELDS:
            raise ValueError(f"rewrite {rewrite.name}: {config_field!r} is not a RewriteConfig field")
        self._records[rewrite.name] = rewrite
        self._fields[rewrite.name] = config_field
        return rewrite

    def names(self):
        """Registered names in registration order."""
        return tuple(self._records)

    def config_field(self, name):
        return self._fields[name]

    def __getitem__(self, name):
        return self._records[name]

    def __contains__(self, name):
        return name in self._records

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        """The records in application order."""
        return iter([self._records[name] for name in self.order()])

    def order(self):
        """Application order: registration order, each record after the registered records it requires."""
        done, visiting, order = set(), [], []

        def visit(name):
            if name in done:
                return
            if name in visiting:
                cycle = visiting[visiting.index(name):] + [name]
                raise ValueError(f"rewrite requirements form a cycle: {' -> '.join(cycle)}")
            visiting.append(name)
            for required in self._records[name].requires:
                if required in self._records:
                    visit(required)
            visiting.pop()
            done.add(name)
            order.append(name)

        for name in self._records:
            visit(name)
        return tuple(order)

    # ---- planner input

    def qualification(self, name, mace_version=None):
        """(qualified, reason) of a record against the installed source (SPEC §6.3)."""
        return qualify(self._records[name], mace_version)

    def candidates(self, model, mace_version=None):
        """Every record as a Candidate for ``model``, in application order; disabled sources are logged."""
        out = []
        for rewrite in self:
            qualified = qualify(rewrite, mace_version)
            if not qualified.supported:
                LOGGER.warning("rewrite %s disabled: %s", rewrite.name, qualified.reason)
            support = Support(*rewrite.applies_to(model))
            out.append(Candidate(rewrite.name, self._fields[rewrite.name], rewrite.default, rewrite.exactness,
                                 support, qualified, rewrite.requires, rewrite.conflicts,
                                 rewrite.workload_constraints))
        return tuple(out)

    def requested(self, config):
        """{name: "on" | "off" | "auto"} from a RewriteConfig (a bool field: True -> "auto", False -> "off")."""
        return {name: _toggle(name, getattr(config, self._fields[name])) for name in self._records}

    # ---- resolution

    def resolve(self, model, requested=None, mace_version=None, strict=True):
        """Decide every record for ``model``; returns a Resolution.

        ``requested``: a RewriteConfig, or {name: "on" | "off" | "auto" | bool} (missing names are
        "auto"). ``strict``: raise RewriteRefusedError when an explicit "on" cannot be honoured;
        otherwise such a record is disabled with ``refused=True``.

        Several records can share one config field, as the MH-1 and ``GENERIC_*`` variants of
        ``reassociate_first``, ``streaming_conv`` and ``fused_folded_backward`` do. An explicit "on"
        for such a field is honoured when at least one of its records is enabled; the records of
        that field that do not fit the model are then disabled with their reason, not refused.
        """
        if isinstance(requested, RewriteConfig):
            requested = self.requested(requested)
        requested = dict(requested or {})
        unknown = set(requested) - set(self._records)
        if unknown:
            raise ValueError(f"unknown rewrite(s) {sorted(unknown)}; registered: {list(self._records)}")
        requested = {name: _toggle(name, value) for name, value in requested.items()}
        enabled, decisions, misfits = [], [], set()
        for candidate in self.candidates(model, mace_version):
            name = candidate.name
            want = requested.get(name, "auto")
            reason = self._blocker(candidate, want, enabled)
            if reason is None:
                enabled.append(name)
                decisions.append(Decision(name, want, True, "on" if want == "on" else f"default {candidate.default}"))
                continue
            if want == "on" and candidate.qualified.supported and not candidate.support.supported:
                misfits.add(name)  # the model is not this variant's kind; another variant may take the field
            decisions.append(Decision(name, want, False, reason, want == "on" and reason != "off"))
        # A field shared by several variants is honoured when one of them is enabled: the variants
        # that do not fit the model are then off, not refused. Any other blocker still refuses.
        decisions = [self._excuse_misfit(d, enabled) if d.name in misfits else d for d in decisions]
        refusals = [f"{d.name}: {d.reason}" for d in decisions if d.refused]
        if refusals and strict:
            raise RewriteRefusedError("requested rewrites cannot be enabled: " + "; ".join(refusals))
        return Resolution(tuple(decisions))

    def _excuse_misfit(self, decision, enabled):
        """``decision`` (a variant that does not fit the model) unrefused if another variant of its field is on."""
        field = self._fields[decision.name]
        honoured_by = [name for name in enabled if name != decision.name and self._fields[name] == field]
        if not honoured_by:
            return decision
        return dataclasses.replace(
            decision, refused=False, reason=f"{decision.reason} (field {field} honoured by {', '.join(honoured_by)})")

    def _blocker(self, candidate, want, enabled):
        """Why ``candidate`` stays off given the records enabled so far, or None."""
        if want == "off":
            return "off"
        if not candidate.qualified.supported:
            return candidate.qualified.reason
        if not candidate.support.supported:
            return candidate.support.reason
        if want == "auto":
            if candidate.exactness == "approximate":
                return "approximate rewrites are never default (SPEC §6.1)"
            if candidate.default == "off":
                return "default off; only an explicit 'on' enables it"
        for required in candidate.requires:
            if required not in self._records:
                return f"requires {required}, which is not registered"
            if required not in enabled:
                return f"requires {required}, which is off"
        for other in enabled:
            if other in candidate.conflicts or candidate.name in self._records[other].conflicts:
                return f"conflicts with {other}"
        return None

    # ---- application and self-checks

    def selfcheck(self, name, original, rewritten, context=None):
        """Run one record's self-check and hold it to SPEC §6.4 (THRESHOLDS), whatever it claims."""
        context = context or Context()
        result = CheckResult.coerce(self._records[name].selfcheck(original, rewritten, context))
        # A result without a known dtype is held to the strictest threshold (SPEC §6.4), never the loosest.
        limit = THRESHOLDS[result.dtype] if result.dtype in THRESHOLDS else min(THRESHOLDS.values())
        error = result.max_relative_error
        passed = bool(result.passed) and error <= limit
        detail = result.detail
        if result.passed and not passed:
            detail = f"relative error {error:.3g} > {limit:g} (SPEC §6.4); record reported: {detail}"
        return dataclasses.replace(result, passed=passed, threshold=limit, detail=detail)

    def apply(self, model, resolution, context=None, original=None, check=True):
        """Apply the enabled records to ``model`` in place, in order, each followed by its self-check.

        ``original`` is the unrewritten model the self-checks compare against (the force field
        passes the model it copied); by default a deep copy of ``model`` is taken first. Raises
        SelfCheckError on a failed check. Returns a tuple of AppliedRewrite.
        """
        context = context or Context()
        if check and original is None:
            original = _copy(model)
        applied = []
        for name in resolution.enabled:
            detail = self._records[name].apply(model, context)
            result = None
            if check:
                result = self.selfcheck(name, original, model, context)
                if not result.passed:
                    raise SelfCheckError(f"rewrite {name} failed its self-check: {result.detail}")
            applied.append(AppliedRewrite(name, result, detail))
            context = dataclasses.replace(context, applied=context.applied + (name,))
        return tuple(applied)


def _copy(model):
    """A deep copy of ``model`` on its device.

    A deep copy of a CUDA cuEquivariance model leaves constants of its naive fx graphs on the CPU;
    moving the copy back to the model's device restores them (the research engine does the same).
    """
    duplicate = copy.deepcopy(model)
    tensors = getattr(model, "parameters", None)
    first = next(iter(tensors()), None) if tensors is not None else None
    if first is None and hasattr(model, "buffers"):
        first = next(iter(model.buffers()), None)
    return duplicate.to(first.device) if first is not None and hasattr(duplicate, "to") else duplicate


def default_registry():
    """A fresh Registry of every family in RECORD_SOURCES."""
    registry = Registry()
    for source in RECORD_SOURCES:
        for record in importlib.import_module(source).RECORDS.values():
            registry.register(record)
    return registry


__all__ = [
    "AppliedRewrite", "Candidate", "CheckResult", "Context", "DEFAULTS", "Decision", "EXACTNESS", "RECORD_SOURCES",
    "Registry", "Resolution", "Rewrite", "RewriteError", "RewriteRefusedError", "SelfCheckError", "Support",
    "THRESHOLDS", "check_tensors", "default_registry", "installed_mace_version", "qualify", "relative_error",
    "source_sha256", "threshold_for",
]
