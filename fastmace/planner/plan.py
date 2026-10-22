"""The Plan record and the Planner protocol (SPEC §9.1, §9.5).

The record and protocol are plan task 1a-T02's; the fixed-plan planner (``fixed.py``, 1a-T15)
reproduces what the research engine's ``main.py`` runs today, and the minimal planner
(``minimal.py``, 1b-T12) decides from the rules of §9.3 that phase 1b has. The planner is the only
code that builds force fields, integrators and engines from a plan (SPEC §5).

Phase 1b adds revisions (``Plan.revise``), the software traits the plan was made with, and the
restart record (``Plan.to_restart``, ``Plan.from_restart``; the format is
docs/design/plan-1b.md). Phase 2 records the fallback's tried candidates in ``fallbacks``
(``planner.fallback.TriedCandidate``, 2-T06), in restart record version 2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastmace.config import (
        EngineConfig,
        IntegratorConfig,
        NeighbourConfig,
        PolarConfig,
        PrecisionConfig,
        RewriteConfig,
    )
    from fastmace.engine.base import Engine
    from fastmace.planner.fallback import TriedCandidate
    from fastmace.planner.workload import Workload

# The schema of Plan.to_restart; a change of its keys or meaning increases it. Version 2 (2-T06) adds
# "fallbacks"; version 1 records still read.
PLAN_RECORD_VERSION = 2
PLAN_RECORD_VERSIONS = (1, 2)
# The resolved config parts of a plan, by attribute name, and their classes in fastmace.config.
# Where a plan runs (SPEC §8.7): in this process, or in a child through the subprocess runner (2-T07).
RUNNERS = ("in-process", "subprocess")
_PARTS = (("rewrites", "RewriteConfig"), ("engine", "EngineConfig"), ("precision", "PrecisionConfig"),
          ("neighbours", "NeighbourConfig"), ("polar", "PolarConfig"), ("integrator", "IntegratorConfig"))


@dataclass(frozen=True)
class Hardware:
    """What the planner knows about the GPU: name, compute capability, memory (bytes)."""

    name: str
    compute_capability: tuple[int, int]
    memory_bytes: int


@dataclass(frozen=True)
class SoftwareTraits:
    """The software traits of SPEC §9.2 the plan was made with: each ``(available, reason)``.

    - ``oeq``: the OpenEquivariance build path (installed; built when the hybrid model is loaded);
    - ``triton``: Triton imports;
    - ``nvrtc``: NVRTC loads (the streaming kernels compile with it);
    - ``fused_cueq``: cuEquivariance's fused kernels load (``cuequivariance_ops_torch`` imports);
    - ``mace_imported_first``: modules imported before FastMACE could prepare the OEq runtime
      (``loading.hybrid_blockers``), empty when it is prepared or still possible.
    """

    oeq: tuple[bool, str]
    triton: tuple[bool, str]
    nvrtc: tuple[bool, str]
    fused_cueq: tuple[bool, str]
    mace_imported_first: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("oeq", "triton", "nvrtc", "fused_cueq"):
            available, reason = getattr(self, name)
            object.__setattr__(self, name, (bool(available), str(reason)))
        object.__setattr__(self, "mace_imported_first", tuple(str(m) for m in self.mace_imported_first))

    def to_dict(self) -> dict[str, Any]:
        return {"oeq": list(self.oeq), "triton": list(self.triton), "nvrtc": list(self.nvrtc),
                "fused_cueq": list(self.fused_cueq), "mace_imported_first": list(self.mace_imported_first)}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> SoftwareTraits:
        return cls(oeq=tuple(record["oeq"]), triton=tuple(record["triton"]), nvrtc=tuple(record["nvrtc"]),
                   fused_cueq=tuple(record["fused_cueq"]), mace_imported_first=tuple(record["mace_imported_first"]))


@dataclass(frozen=True)
class SelfCheck:
    """One rewrite record's self-check at build time (SPEC §6.4; plan task 2-T04).

    ``passed`` and the numbers are ``None`` when the force field was built with ``check=False``.
    """

    name: str
    passed: bool | None
    max_relative_error: float | None
    threshold: float | None
    modules: int
    dtype: str | None
    detail: str

    @classmethod
    def from_mapping(cls, record: Any) -> SelfCheck:
        """From one entry of ``fastmace.rewrites.registry.selfcheck_records``."""
        return cls(**{f.name: record[f.name] for f in dataclasses.fields(cls)})


@dataclass(frozen=True)
class Plan:
    """A frozen execution plan (SPEC §9.5).

    - ``tier``: ``"graph"`` (MD), ``"compiled"`` (the one-off evaluation of ``fm.ForceField.evaluate``
      and ``fm.calc``; later also MD), ``"eager"``, ``"reference"``;
    - the resolved config objects (no ``auto`` left): ``rewrites``, ``engine``, ``precision``,
      ``neighbours``, ``polar`` (``None`` for non-polar models), ``integrator`` (``None`` for an
      evaluation plan);
    - ``decisions``: ``(key, choice, reason)`` for every decision, enabled and disabled rewrites
      included. ``key`` is stable and unique within a plan (``plan.decision(key)``); a reason that
      reports and tests match on is a ``fastmace.errors.Reason`` with a ``code``;
    - ``requested``: ``(item, value)`` of every option the caller set to something other than
      ``"auto"`` (an explicit ``"on"`` is refused, never downgraded, when it cannot be honoured);
    - ``auto_options_off``: ``(option, reason)`` for every option that ``auto`` switched off because
      the model does not support it (the research engine's ``model.auto_options_off``);
    - ``fallbacks``: every candidate the fallback tried (``planner.fallback.TriedCandidate``), in
      order, carried from revision to revision (§9.4 step 4); empty when no fallback ran;
    - ``predicted_memory_bytes``: ``None`` until the memory model exists;
    - ``superset``: the species superset, atomic numbers sorted (minimal plans; ``()`` in 1a's);
    - ``workload``: the ``Workload`` planned for (``None`` for evaluations and 1a's fixed plans);
    - ``software``: the ``SoftwareTraits`` the plan was made with (``None`` in 1a's fixed plans);
    - ``allowances``: the ``allow_*`` flags the plan used (``"allow_naive_cueq"``), recorded;
    - ``revision``, ``predecessor``, ``reason``, ``step``: a plan never changes in place; a runtime
      switch or fallback makes a new revision (``revise``) that points at the one before, with the
      step it takes effect at;
    - ``runner``: ``"in-process"``, or ``"subprocess"`` where a process constraint of §8.7 forbids the
      plan in this process and the subprocess runner runs it in a child (2-T07). Not in the restart
      record: the child resumes the plan in its own process;
    - ``selfchecks``: the per-record self-checks (``SelfCheck``) of the built plan, in application
      order; ``()`` before the build (``MinimalPlanner.build`` fills them, 2-T04). Not in the
      restart record: a resume runs them again.
    """

    tier: str
    rewrites: RewriteConfig | None = None
    engine: EngineConfig | None = None
    precision: PrecisionConfig | None = None
    neighbours: NeighbourConfig | None = None
    polar: PolarConfig | None = None
    integrator: IntegratorConfig | None = None
    decisions: tuple[tuple[str, str, str], ...] = ()
    requested: tuple[tuple[str, str], ...] = ()
    auto_options_off: tuple[tuple[str, str], ...] = ()
    fallbacks: tuple[TriedCandidate, ...] = ()
    predicted_memory_bytes: int | None = None
    superset: tuple[int, ...] = ()
    workload: Workload | None = None
    software: SoftwareTraits | None = None
    allowances: tuple[str, ...] = ()
    revision: int = 0
    predecessor: Plan | None = None
    reason: str = "initial plan"
    step: int | None = None
    runner: Literal["in-process", "subprocess"] = "in-process"
    selfchecks: tuple[SelfCheck, ...] = ()

    def __post_init__(self):
        if self.runner not in RUNNERS:
            raise ValueError(f"Plan.runner {self.runner!r} is not one of {RUNNERS}")

    def decision(self, key: str) -> tuple[str, str, str]:
        """The decision with ``key``; ``KeyError`` if the plan has none."""
        for entry in self.decisions:
            if entry[0] == key:
                return entry
        raise KeyError(f"the plan has no decision {key!r}")

    def revise(self, reason: str, step: int | None, **changes: Any) -> Plan:
        """A new revision: ``changes`` applied, ``predecessor`` this plan, at ``step`` (SPEC §9.5).

        This plan stays as it is. ``revision``, ``predecessor`` and ``step`` are set here, not changed.
        """
        for name in ("revision", "predecessor", "reason", "step"):
            if name in changes:
                raise ValueError(f"Plan.revise sets {name}; pass the other fields to change")
        return dataclasses.replace(self, revision=self.revision + 1, predecessor=self, reason=str(reason),
                                   step=step, **changes)

    def chain(self) -> tuple[Plan, ...]:
        """The predecessors, oldest first (provenance keeps the whole chain, §9.5)."""
        out, plan = [], self.predecessor
        while plan is not None:
            out.append(plan)
            plan = plan.predecessor
        return tuple(reversed(out))

    def describe(self) -> str:
        """The human-readable plan (``str(plan)``), written into provenance (SPEC §13)."""
        lines = [f"Plan revision {self.revision}, tier {self.tier}: {self.reason}"
                 + ("" if self.step is None else f" (from step {self.step})")]
        lines += [f"  {item} = {choice}  ({reason})" for item, choice, reason in self.decisions]
        if self.requested:
            lines.append("  requested: " + ", ".join(f"{item}={value}" for item, value in self.requested))
        if self.allowances:
            lines.append("  allowed: " + ", ".join(self.allowances))
        if self.runner != "in-process":
            lines.append(f"  runner: {self.runner} (SPEC §8.7)")
        if self.selfchecks:
            failed = [check.name for check in self.selfchecks if check.passed is False]
            lines.append(f"  self-checks: {len(self.selfchecks)} records"
                         + (f", failed: {', '.join(failed)}" if failed else ""))
        for tried in self.fallbacks:
            outcome = (f"activated as revision {tried.revision}" if tried.outcome == "activated"
                       else tried.outcome + (f" at {tried.stage}" if tried.stage else ""))
            lines.append(f"  fallback tried: {tried.change}: {outcome} ({tried.reason})")
        if self.predecessor is not None:
            lines.append(f"  predecessor: revision {self.predecessor.revision}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    # ------------------------------------------------------------------ restart record

    def to_restart(self, *, edge_bucket: int, neighbour_capacity: int) -> dict[str, Any]:
        """The active revision as a JSON-safe mapping for the restart file (SPEC §9.5, §10.6).

        ``edge_bucket`` and ``neighbour_capacity`` are the engine's at the restart point (§10.6,
        "edge and capacity buckets"); the predecessors are summarised in ``chain``.
        """
        from fastmace.errors import Reason

        def config(part):
            return None if part is None else _json_safe(dataclasses.asdict(part))

        def decision(entry):
            key, choice, reason = entry
            return [key, choice, str(reason), reason.code if isinstance(reason, Reason) else None,
                    reason.rule if isinstance(reason, Reason) else None]

        record = {
            "plan_record_version": PLAN_RECORD_VERSION,
            "edge_bucket": int(edge_bucket),
            "neighbour_capacity": int(neighbour_capacity),
            "tier": self.tier,
            "revision": self.revision,
            "reason": str(self.reason),
            "step": self.step,
            "chain": [{"revision": p.revision, "step": p.step, "reason": str(p.reason)} for p in self.chain()],
            **{name: config(getattr(self, name)) for name, _ in _PARTS},
            "decisions": [decision(entry) for entry in self.decisions],
            "requested": [list(entry) for entry in self.requested],
            "auto_options_off": [list(entry) for entry in self.auto_options_off],
            "predicted_memory_bytes": self.predicted_memory_bytes,
            "superset": list(self.superset),
            "workload": None if self.workload is None else _json_safe(dataclasses.asdict(self.workload)),
            "software": None if self.software is None else self.software.to_dict(),
            "allowances": list(self.allowances),
            "fallbacks": [dataclasses.asdict(tried) for tried in self.fallbacks],
        }
        return record

    @classmethod
    def from_restart(cls, record: dict[str, Any]) -> Plan:
        """The plan of a ``to_restart`` record; the predecessors come back as summaries (tier and reason)."""
        from fastmace import config as configs
        from fastmace.errors import Reason
        from fastmace.planner.fallback import TriedCandidate
        from fastmace.planner.workload import Workload

        if record.get("plan_record_version") not in PLAN_RECORD_VERSIONS:
            raise ValueError(f"plan record version {record.get('plan_record_version')!r} is not one of "
                             f"{PLAN_RECORD_VERSIONS}")

        def part(name, class_name):
            values = record[name]
            if values is None:
                return None
            cls_ = getattr(configs, class_name)
            fields = {f.name: f for f in dataclasses.fields(cls_)}
            values = {key: _from_json(value) for key, value in values.items() if key in fields}
            if class_name == "IntegratorConfig":  # a plan from before the option ran ASE's projection (3-T22c)
                values.setdefault("fixcm_projection", "ase")
            return cls_(**values)

        def decision(entry):
            key, choice, reason, code, rule = entry
            return key, choice, Reason(reason, rule=rule, code=code) if code is not None else reason

        predecessor = None
        for summary in record["chain"]:
            predecessor = cls(tier=record["tier"], revision=summary["revision"], step=summary["step"],
                              reason=summary["reason"], predecessor=predecessor)
        workload = record["workload"]
        if workload is not None:
            workload = Workload(**{key: _from_json(value) for key, value in workload.items()})
        return cls(
            tier=record["tier"],
            **{name: part(name, class_name) for name, class_name in _PARTS},
            decisions=tuple(decision(entry) for entry in record["decisions"]),
            requested=tuple(tuple(entry) for entry in record["requested"]),
            auto_options_off=tuple(tuple(entry) for entry in record["auto_options_off"]),
            predicted_memory_bytes=record["predicted_memory_bytes"],
            superset=tuple(record["superset"]),
            workload=workload,
            software=None if record["software"] is None else SoftwareTraits.from_dict(record["software"]),
            allowances=tuple(record["allowances"]),
            fallbacks=tuple(TriedCandidate(**tried) for tried in record.get("fallbacks", ())),
            revision=record["revision"],
            predecessor=predecessor,
            reason=record["reason"],
            step=record["step"],
        )


def _json_safe(value):
    """Tuples and frozensets as lists (frozensets sorted); everything else as it is."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _from_json(value):
    """Lists back as tuples (the config records and ``Workload`` normalise them further)."""
    if isinstance(value, list):
        return tuple(_from_json(item) for item in value)
    return value


@runtime_checkable
class Planner(Protocol):
    """Turns a workload into a Plan and a Plan into a running Engine."""

    def plan(self, traits: Any, atoms: Any, integrator: IntegratorConfig, hardware: Hardware) -> Plan:
        """Decide the plan for model ``traits`` (``fastmace.model.capabilities``), an ASE ``Atoms``,
        the integrator settings and the hardware. Refuses unsupported inputs (non-3D-PBC,
        constraints, inexact padding for the graph tier) with reasons, before any compilation."""
        raise NotImplementedError

    def build(self, plan: Plan, model: Any, atoms: Any) -> Engine:
        """Build the ForceField, SimulationState, Integrator and Engine that ``plan`` describes.

        The only place in the library that instantiates an engine.
        """
        raise NotImplementedError
