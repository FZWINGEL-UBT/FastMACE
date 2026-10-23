"""The scheme registry: SPEC §4.4's support table as data, and the planner's dispatch (plan task 4-T01).

Every integrator and optimiser scheme declares one record: what it is (the class that implements it, as a
``"module:Class"`` path imported only when the planner builds it), which ensembles and cell modes it runs,
which constraints it supports, whether it samples a canonical ensemble, and how it counts degrees of
freedom. Rule 1 (SPEC §9.3) reads the records (``support_refusals``) and the planner builds through them
(``build_integrator``), so a new scheme is a new record and its implementation, never a planner branch.

Records live one module per scheme in ``fastmace.integrators.schemes`` and ``fastmace.optimize.schemes``,
each with a module-level ``RECORDS`` tuple; ``registry()`` discovers them. Those modules import neither
torch nor the implementation, so importing the registry stays cheap. A plan task adds its scheme by adding
its module there, and changes no shared file.

Owner decisions this module encodes (phase 4 plan, "Owner decisions"; recommendations until decided):

- D1: ``FIX_ATOMS_WITH_FIXCM``: ``fixcm=True`` with ``FixAtoms`` is refused. Which schemes take
  ``FixAtoms`` is each record's ``constraints``.
- D2: ``degrees_of_freedom``: ``dof="fastmace"`` counts 3N - 3 under ``fixcm``, 3(N - n_fixed) with
  ``FixAtoms``, 3N otherwise; ``dof="ase"`` the count of the scheme's ASE class (``ase_dof``);
  ``dof="auto"`` resolves to ``DOF_AUTO`` (``"fastmace"``; ``"ase"`` under alternative (b)).
- Decision 10 (17 October 2026): MC moves with constraints are refused, also in the reference tier.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fastmace.errors import Reason

if TYPE_CHECKING:
    from fastmace.config import IntegratorConfig, OptimizerConfig
    from fastmace.state import SimulationState

KINDS = ("integrator", "optimizer", "cell_filter")
ENSEMBLES = ("nve", "nvt", "npt")
# The cell modes a record runs: "fixed" (the cell never changes) and the variable-cell modes of
# IntegratorConfig.cell_mode ("hydrostatic": shape fixed, "full": fully flexible). Optimisers are
# fixed cell; their variable cell is a cell filter's (OptimizerConfig.cell_filter).
CELL_MODES = ("fixed", "hydrostatic", "full")
# The §4.4 constraint rows a record can declare.
CONSTRAINTS = ("fixcm", "fix_atoms")
DOF_MODES = ("fastmace", "ase")
# D2 (recommended (a)): what dof="auto" resolves to in a plan; "ase" under alternative (b).
DOF_AUTO = "fastmace"
# ASE's degree-of-freedom counts, by scheme: "constraints" is Atoms.get_number_of_degrees_of_freedom
# (3N minus the constraints' removed degrees; ASE Bussi, Atoms.get_temperature for Berendsen); "3n"
# ignores constraints (ASE NoseHooverChainNVT, MTKNPT, IsotropicMTKNPT).
ASE_DOF = ("constraints", "3n")
# D1 (recommended (a)): FixAtoms pins the frame already, and ASE's fixcm displacement moves fixed atoms
# before the constraint resets them. Changing D1 here lifts the refusal for every scheme.
FIX_ATOMS_WITH_FIXCM = "refused"

_PACKAGES = {"integrator": "fastmace.integrators.schemes", "optimizer": "fastmace.optimize.schemes",
             "cell_filter": "fastmace.optimize.schemes"}


def _import(path: str) -> Any:
    """The object at ``"module:attribute"``."""
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(f"an implementation path is 'module:attribute', got {path!r}")
    return getattr(importlib.import_module(module), name)


def _check(values: Iterable[str], allowed: tuple[str, ...], what: str, scheme: str) -> tuple[str, ...]:
    values = tuple(values)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"scheme {scheme!r}: {what} {unknown} not among {allowed}")
    return values


@dataclass(frozen=True, kw_only=True)
class SchemeRecord:
    """One row of the support table: a scheme and what it supports (the base of every record).

    - ``scheme``: the name in ``IntegratorConfig.scheme`` / ``OptimizerConfig.scheme`` /
      ``OptimizerConfig.cell_filter``, unique per kind;
    - ``implementation``: ``"module:Class"``, imported only when the planner builds the scheme;
    - ``constraints``: the §4.4 rows it supports (``CONSTRAINTS``);
    - ``refusals``: ``{constraint: reason}`` for the rows it does not support, quoted in the refusal
      (for example "ASE's NoseHooverChainNVT refuses constraints, so there is no reference");
    - ``ase_class``: the ASE 3.27 class it ports, for reports and the equivalence tests;
    - ``state_keys``: the integrator or optimiser ``state_dict`` keys a restart holds (format 4);
    - ``buffers``: the runtime-state policies of its own persistent buffers (SPEC §8.1;
      ``fastmace.engine.registry``), ``(name, restart, recompute)`` with ``name`` relative to its owner
      (``"eta"`` is ``integrator.eta``; optimisers ``optimizer.``, cell filters ``dof.``), ``restart`` in
      ``RESTART_POLICIES``; names Langevin and BAOAB share are in ``engine.registry.POLICIES`` already;
    - ``report``: a line for the setup report ("non-canonical: equilibration only",
      "experimental: not yet qualified"), or ``None``.
    """

    kind: Literal["integrator", "optimizer", "cell_filter"]
    scheme: str
    implementation: str
    constraints: tuple[str, ...] = ()
    refusals: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
    ase_class: str | None = None
    state_keys: tuple[str, ...] = ()
    buffers: tuple[tuple[str, str, str | None], ...] = ()
    report: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"scheme {self.scheme!r}: kind {self.kind!r} not among {KINDS}")
        if not isinstance(self.scheme, str) or not self.scheme or self.scheme != self.scheme.strip():
            raise ValueError(f"a scheme name is a non-empty string, got {self.scheme!r}")
        object.__setattr__(self, "constraints", _check(self.constraints, CONSTRAINTS, "constraints", self.scheme))
        refusals = tuple(sorted(dict(self.refusals).items()))  # hashable, as the record is
        _check((name for name, _ in refusals), CONSTRAINTS, "refusals", self.scheme)
        object.__setattr__(self, "refusals", refusals)
        object.__setattr__(self, "state_keys", tuple(self.state_keys))
        buffers = tuple(tuple(entry) for entry in self.buffers)
        for entry in buffers:
            if len(entry) != 3 or not isinstance(entry[0], str) or entry[1] not in ("save", "reset", "derived"):
                raise ValueError(f"scheme {self.scheme!r}: a buffer policy is (name, 'save'|'reset'|'derived', "
                                 f"recompute), got {entry!r}")
        object.__setattr__(self, "buffers", buffers)

    def load(self) -> type:
        """The implementing class (imports it)."""
        return _import(self.implementation)

    def refusal(self, constraint: str) -> str:
        """Why ``constraint`` is not supported: the record's reason, or the generic one."""
        return dict(self.refusals).get(constraint, f"{self.kind} scheme {self.scheme!r} does not declare {constraint} "
                                             "support (SPEC §4.4)")


@dataclass(frozen=True, kw_only=True)
class IntegratorRecord(SchemeRecord):
    """An integrator scheme (SPEC §10.2).

    - ``ensembles``: the ``IntegratorConfig.ensemble`` values it runs;
    - ``cell_modes``: ``"fixed"`` for fixed-cell schemes, else the ``IntegratorConfig.cell_mode`` values
      it runs (an ``"npt"`` config is variable cell, any other fixed);
    - ``non_canonical``: Berendsen; the plan warns and the report says "equilibration only";
    - ``dof_modes``: the ``IntegratorConfig.dof`` values it implements;
    - ``ase_dof``: ASE's count for ``dof="ase"`` (``ASE_DOF``);
    - ``dof_rule``: ``"d2"``, D2's conventions (``degrees_of_freedom``), or ``"langevin"``: Langevin's and
      BAOAB's count since 3-T22c (3N - 3 only under NVT ``fixcm`` with the ``"mass_weighted"`` projection),
      which phase 4 keeps bitwise;
    - ``keywords``: ``"module:function"`` giving extra constructor keywords from the config
      (BAOAB's ``externalstress``), or ``None``;
    - ``conserved``: what ``output_values()["conserved"]`` is, or ``None`` for a scheme without one
      (Langevin, BAOAB).
    """

    kind: Literal["integrator"] = "integrator"
    ensembles: tuple[str, ...] = ("nvt",)
    cell_modes: tuple[str, ...] = ("fixed",)
    non_canonical: bool = False
    dof_modes: tuple[str, ...] = ("fastmace", "ase")
    ase_dof: str = "constraints"
    dof_rule: Literal["d2", "langevin"] = "d2"
    keywords: str | None = None
    conserved: str | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "ensembles", _check(self.ensembles, ENSEMBLES, "ensembles", self.scheme))
        object.__setattr__(self, "cell_modes", _check(self.cell_modes, CELL_MODES, "cell modes", self.scheme))
        object.__setattr__(self, "dof_modes", _check(self.dof_modes, DOF_MODES, "dof modes", self.scheme))
        _check((self.ase_dof,), ASE_DOF, "ase_dof", self.scheme)
        if self.dof_rule not in ("d2", "langevin"):
            raise ValueError(f"scheme {self.scheme!r}: dof_rule {self.dof_rule!r} is 'd2' or 'langevin'")
        if ("npt" in self.ensembles) != any(mode != "fixed" for mode in self.cell_modes):
            raise ValueError(f"scheme {self.scheme!r}: an 'npt' scheme has variable cell modes and only an npt "
                             f"one, got ensembles={self.ensembles} cell_modes={self.cell_modes}")

    @property
    def variable_cell(self) -> bool:
        """Whether the scheme can change the cell."""
        return "npt" in self.ensembles

    def cell_mode(self, config: IntegratorConfig) -> str:
        """The config's cell mode: ``"fixed"`` unless the ensemble is ``"npt"``."""
        return config.cell_mode if config.ensemble == "npt" else "fixed"

    def degrees_of_freedom(self, config: Any, natoms: int, n_fixed: int = 0) -> int:
        """The kinetic degrees of freedom of ``config`` on ``natoms`` atoms with ``n_fixed`` fixed ones."""
        if self.dof_rule == "langevin":
            if n_fixed:
                return 3 * (natoms - n_fixed)
            if (getattr(config, "ensemble", "nvt") == "nvt" and getattr(config, "fixcm", False)
                    and getattr(config, "fixcm_projection", "ase") == "mass_weighted"):
                return 3 * natoms - 3
            return 3 * natoms
        return degrees_of_freedom(natoms, n_fixed=n_fixed, fixcm=bool(getattr(config, "fixcm", False)),
                                  dof=resolve_dof(getattr(config, "dof", "auto")), ase_dof=self.ase_dof)

    def constructor_keywords(self, config: IntegratorConfig) -> dict[str, Any]:
        """The extra constructor keywords (``keywords``), ``{}`` without."""
        return dict(_import(self.keywords)(config)) if self.keywords else {}


@dataclass(frozen=True, kw_only=True)
class OptimizerRecord(SchemeRecord):
    """An optimiser scheme (SPEC §10.3): ``cell_filters`` are the ``OptimizerConfig.cell_filter`` values it
    runs (``None`` = fixed cell). The implementation takes a ``fastmace.optimize.base.DOFView``."""

    kind: Literal["optimizer"] = "optimizer"
    cell_filters: tuple[str | None, ...] = (None,)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "cell_filters", tuple(self.cell_filters))


@dataclass(frozen=True, kw_only=True)
class CellFilterRecord(SchemeRecord):
    """A cell filter (4-T07's ``FrechetCellFilter``): the implementation is a ``DOFView`` over positions and
    cell, so both optimisers run unchanged with it."""

    kind: Literal["cell_filter"] = "cell_filter"


def resolve_dof(dof: str) -> str:
    """``dof`` with ``"auto"`` resolved (``DOF_AUTO``, owner decision D2); the planner records it."""
    return DOF_AUTO if dof == "auto" else dof


def degrees_of_freedom(natoms: int, *, n_fixed: int = 0, fixcm: bool, dof: str = "auto",
                       ase_dof: str = "constraints") -> int:
    """D2's count (recommended (a)) for the new integrators.

    ``dof="fastmace"``: 3(N - n_fixed) with fixed atoms (no centre-of-mass projection then, D1), else
    3N - 3 under ``fixcm``, else 3N. ``dof="ase"``: the scheme's ASE class, ``ase_dof`` ``"constraints"``
    (3(N - n_fixed)) or ``"3n"``. ``"auto"`` is ``DOF_AUTO``.
    """
    dof = resolve_dof(dof)
    if dof not in DOF_MODES:
        raise ValueError(f"dof must be one of {DOF_MODES}, got {dof!r}")
    if dof == "ase":
        return 3 * natoms if ase_dof == "3n" else 3 * (natoms - n_fixed)
    if n_fixed:
        return 3 * (natoms - n_fixed)
    return 3 * natoms - 3 if fixcm else 3 * natoms


class Registry:
    """The records of one process, by kind and scheme name."""

    def __init__(self, records: Iterable[SchemeRecord] = ()):
        self._records: dict[tuple[str, str], SchemeRecord] = {}
        for record in records:
            self.register(record)

    def register(self, record: SchemeRecord) -> SchemeRecord:
        """Add ``record``; a second record of the same kind and scheme raises."""
        if not isinstance(record, SchemeRecord):
            raise TypeError(f"a scheme record is a SchemeRecord, got {type(record).__name__}")
        key = (record.kind, record.scheme)
        if key in self._records and self._records[key] != record:
            raise ValueError(f"{record.kind} scheme {record.scheme!r} is registered twice")
        self._records[key] = record
        return record

    def get(self, kind: str, scheme: str | None) -> SchemeRecord | None:
        return self._records.get((kind, scheme))

    def integrator(self, scheme: str) -> IntegratorRecord:
        """The integrator record of ``scheme``; ``KeyError`` naming the registered ones."""
        return self._lookup("integrator", scheme)

    def optimizer(self, scheme: str) -> OptimizerRecord:
        return self._lookup("optimizer", scheme)

    def cell_filter(self, scheme: str) -> CellFilterRecord:
        return self._lookup("cell_filter", scheme)

    def _lookup(self, kind, scheme):
        record = self._records.get((kind, scheme))
        if record is None:
            raise KeyError(f"no {kind} scheme {scheme!r} in this FastMACE; registered: "
                           f"{', '.join(self.schemes(kind)) or 'none'}")
        return record

    def schemes(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(scheme for k, scheme in self._records if k == kind))

    def records(self, kind: str | None = None) -> tuple[SchemeRecord, ...]:
        return tuple(record for (k, _), record in sorted(self._records.items()) if kind in (None, k))


def discover(packages: Iterable[str] = tuple(dict.fromkeys(_PACKAGES.values()))) -> Registry:
    """A registry of every ``RECORDS`` entry in the modules of ``packages``."""
    found = Registry()
    for package in packages:
        module = importlib.import_module(package)
        for info in sorted(pkgutil.iter_modules(module.__path__), key=lambda info: info.name):
            for record in getattr(importlib.import_module(f"{package}.{info.name}"), "RECORDS", ()):
                found.register(record)
    return found


@functools.lru_cache(maxsize=1)
def registry() -> Registry:
    """The process's registry, discovered once (records are immutable, so it is never invalidated)."""
    return discover()


# --------------------------------------------------------------------------- rule 1 (SPEC §4.4, §9.3)


def _reason(text: str, code: str) -> Reason:
    return Reason(text, rule=1, code=code)


def constraint_names(atoms: Any) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``atoms``' ``FixAtoms`` indices (sorted, unique, all ``FixAtoms`` together) and the class names of every
    other constraint."""
    from ase.constraints import FixAtoms

    fixed, other = set(), []
    for constraint in getattr(atoms, "constraints", ()):
        if isinstance(constraint, FixAtoms):
            fixed.update(int(index) for index in constraint.get_indices())
        else:
            other.append(type(constraint).__name__)
    return tuple(sorted(fixed)), tuple(sorted(set(other)))


def _constraint_refusals(record: SchemeRecord, *, fix_atoms: bool, other: tuple[str, ...], fixcm: bool,
                         mc: bool, what: str) -> list[Reason]:
    refusals = []
    if other:
        refusals.append(_reason(f"the constraints {', '.join(other)} are refused in 1.0; only FixAtoms is "
                                "supported (SPEC §4.4)", "constraints"))
    if fix_atoms:
        if "fix_atoms" not in record.constraints:
            refusals.append(_reason(f"FixAtoms constraints are refused with {what}: {record.refusal('fix_atoms')} "
                                    "(SPEC §4.4)", "constraints"))
        elif fixcm and FIX_ATOMS_WITH_FIXCM == "refused":
            refusals.append(_reason("fixcm=True with FixAtoms is refused: the fixed atoms already pin the frame, "
                                    "and ASE's fixcm displacement moves fixed atoms before the constraint resets "
                                    "them; set fixcm=False (owner decision D1, SPEC §4.4)", "fix-atoms-with-fixcm"))
    if mc and (fix_atoms or other):
        refusals.append(_reason("MC moves with constraints are refused in 1.0, also in the reference tier (owner "
                                "decision 10, SPEC §4.4)", "mc-with-constraints"))
    return refusals


def integrator_refusals(config: IntegratorConfig, *, fix_atoms: bool = False, other_constraints: tuple[str, ...] = (),
                        mc: bool = False, variable_cell: bool = False,
                        records: Registry | None = None) -> list[Reason]:
    """Rule 1 for an integrator: every reason ``config`` cannot run here, ``[]`` if it can.

    The scheme is registered and runs the config's ensemble and cell mode (``variable_cell``: the workload
    asks for a variable cell); its constraint rows cover ``fixcm`` and ``FixAtoms``; no other constraint;
    no MC with constraints. Codes: ``scheme-unknown``, ``scheme-ensemble``, ``scheme-cell-mode``,
    ``scheme-dof``, ``variable-cell-without-barostat``, ``constraints``, ``fix-atoms-with-fixcm``,
    ``mc-with-constraints``.
    """
    records = records or registry()
    record = records.get("integrator", config.scheme)
    if record is None:
        return [_reason(f"the integrator scheme {config.scheme!r} is not available in this FastMACE; registered: "
                        f"{', '.join(records.schemes('integrator'))} (SPEC §10.2)", "scheme-unknown")]
    refusals = []
    if config.ensemble not in record.ensembles:
        refusals.append(_reason(f"the {config.scheme} integrator runs ensembles {', '.join(record.ensembles)}, not "
                                f"{config.ensemble!r} (SPEC §10.2)", "scheme-ensemble"))
    elif record.cell_mode(config) not in record.cell_modes:
        refusals.append(_reason(f"the {config.scheme} integrator runs cell modes {', '.join(record.cell_modes)}, "
                                f"not {config.cell_mode!r} (SPEC §10.2)", "scheme-cell-mode"))
    if variable_cell and config.ensemble != "npt":
        refusals.append(_reason(f"a variable cell needs a barostat integrator, got ensemble={config.ensemble!r} "
                                "(SPEC §4.4, §9.3 rule 1)", "variable-cell-without-barostat"))
    if resolve_dof(config.dof) not in record.dof_modes:
        refusals.append(_reason(f"the {config.scheme} integrator implements dof {', '.join(record.dof_modes)}, "
                                f"not {config.dof!r} (owner decision D2)", "scheme-dof"))
    if config.fixcm and "fixcm" not in record.constraints:
        refusals.append(_reason(f"fixcm=True is refused with the {config.scheme} integrator: "
                                f"{record.refusal('fixcm')}", "constraints"))
    refusals += _constraint_refusals(record, fix_atoms=fix_atoms, other=tuple(other_constraints),
                                     fixcm=config.fixcm, mc=mc, what=f"the {config.scheme} integrator")
    return refusals


def optimizer_refusals(config: OptimizerConfig, *, fix_atoms: bool = False, other_constraints: tuple[str, ...] = (),
                       records: Registry | None = None) -> list[Reason]:
    """Rule 1 for an optimiser and its cell filter. Codes: ``scheme-unknown``, ``scheme-cell-filter``,
    ``constraints``."""
    records = records or registry()
    record = records.get("optimizer", config.scheme)
    if record is None:
        return [_reason(f"the optimizer scheme {config.scheme!r} is not available in this FastMACE; registered: "
                        f"{', '.join(records.schemes('optimizer')) or 'none'} (SPEC §10.3)", "scheme-unknown")]
    refusals = []
    if config.cell_filter not in record.cell_filters:
        refusals.append(_reason(f"the {config.scheme} optimizer runs cell filters "
                                f"{', '.join(str(name) for name in record.cell_filters)}, not {config.cell_filter!r} "
                                "(SPEC §10.3)", "scheme-cell-filter"))
    holders = [(record, f"the {config.scheme} optimizer")]
    if config.cell_filter is not None:
        cell = records.get("cell_filter", config.cell_filter)
        if cell is None:
            refusals.append(_reason(f"the cell filter {config.cell_filter!r} is not available in this FastMACE; "
                                    f"registered: {', '.join(records.schemes('cell_filter')) or 'none'} (SPEC §10.3)",
                                    "scheme-unknown"))
        else:
            holders.append((cell, f"the {config.cell_filter} cell filter"))
    seen = set()
    for holder, what in holders:
        for reason in _constraint_refusals(holder, fix_atoms=fix_atoms, other=tuple(other_constraints), fixcm=False,
                                           mc=False, what=what):
            if str(reason) not in seen:
                seen.add(str(reason))
                refusals.append(reason)
    return refusals


# --------------------------------------------------------------------------- dispatch


def build_integrator(state: SimulationState, config: IntegratorConfig, *, temperature_K: float | None = None,
                     records: Registry | None = None) -> Any:
    """The integrator of ``config`` on ``state``: ``Class(state, config, **keywords)``, with ``temperature_K``
    among the keywords unless it is ``None``.

    Only the planner calls it (SPEC §5). Raises ``KeyError`` for an unregistered scheme; rule 1 has
    refused it before.
    """
    record = (records or registry()).integrator(config.scheme)
    keywords = record.constructor_keywords(config)
    if temperature_K is not None:
        keywords["temperature_K"] = temperature_K
    return record.load()(state, config, **keywords)


def build_optimizer(dof: Any, config: OptimizerConfig, *, records: Registry | None = None) -> Any:
    """The optimiser of ``config`` over the DOF view ``dof`` (``fastmace.optimize.base.DOFView``)."""
    return (records or registry()).optimizer(config.scheme).load()(dof, config)


def build_dof(state: SimulationState, config: OptimizerConfig, *, records: Registry | None = None) -> Any:
    """The DOF view an optimiser of ``config`` runs on: positions (``PositionsDOF``) without a cell filter, the
    filter's view with one."""
    if config.cell_filter is None:
        from fastmace.optimize.base import PositionsDOF

        return PositionsDOF(state)
    return (records or registry()).cell_filter(config.cell_filter).load()(state, config)


def callable_path(function: Callable) -> str:
    """``"module:qualname"`` of a module-level function, for ``keywords``."""
    return f"{function.__module__}:{function.__qualname__}"


__all__ = ("ASE_DOF", "CELL_MODES", "CONSTRAINTS", "CellFilterRecord", "DOF_AUTO", "DOF_MODES", "ENSEMBLES",
           "FIX_ATOMS_WITH_FIXCM", "IntegratorRecord", "OptimizerRecord", "Registry", "SchemeRecord",
           "build_dof", "build_integrator", "build_optimizer", "constraint_names", "degrees_of_freedom", "discover",
           "integrator_refusals", "optimizer_refusals", "registry", "resolve_dof")
