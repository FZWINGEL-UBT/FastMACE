"""The minimal planner of phase 1b: rules, not tables; refusals with reasons (SPEC §9.1-§9.5, §8.7).

Plan task 1b-T12. ``MinimalPlanner`` plans for an ``fm.ForceField``, an ``Atoms``, an integrator
config and a ``Workload`` (§9.2), and builds what the plan describes. ``Simulation``, ``fm.md``,
``fm.calc`` and the ``fastmace.ase`` classes get their engines and force-field copies only from it.

What 1b decides (the rules of §9.3 that have their tiers in 1b):

- **hard constraints 1-7**, every reason in one ``PlanRefused`` (``SpeciesNotInSuperset`` when rule 2
  is among them): non-3D-PBC, constraints, a variable cell without a barostat integrator,
  ``allow_reference`` (refused in 1b, owner decision D6); a structure species outside the
  superset; PolarMACE on plain compiled cuEq; inexact padding for the graph tier ("compiled MD
  tier: phase 3", D6); ``reproducible=True`` (no family qualified, phase 2); naive cuEq kernels
  unless ``allow_naive_cueq``; ``atom_count_changes``;
- **owner decision D3**: the superset must equal the structure's species (compared as sets);
  a larger one is phase 3 (§8.5);
- **mode**: graph; every other mode refuses with the phase that brings it. ``plan_evaluation``
  plans the one-off compiled evaluation of ``fm.ForceField.evaluate`` and ``fm.calc``;
- **backend** (§9.4, 1a-T00): hybrid when OpenEquivariance can build and MACE was not imported
  before FastMACE could prepare it, else plain cuEq with the reason. The process's compile family
  is respected (§8.7): the first compiled plan fixes it, a later plan of the other family raises
  ``ProcessConstraintError``;
- **preferences in 1b**: rule 9 keeps the graph tier with a host sync at each callback (D6),
  rule 10 takes the stress path and k-vector headroom from the integrator, rule 11 records
  ``max_temperature_K`` and keeps 1a's skin and buckets. Rules 8 and 12 are not in 1b.

The rewrite set and the engine options come from 1a's resolution (``FixedPlanner._resolve``),
unchanged: the minimal plan of a ``main.py`` configuration has the rewrites and engine options of
``FixedPlanner``'s plan. Every decision has a stable key; reasons that reports match on are
``Reason`` records with a code (docs/design/plan-1b.md).

The module holds no state; a planner keeps only its configs, its software traits and the process
record it enforces.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Mapping
from typing import Any

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig, PolarConfig, PrecisionConfig, RewriteConfig
from fastmace.errors import ModelElementError, PlanRefused, Reason, SpeciesNotInSuperset

from . import fixed
from .plan import Hardware, Plan, SelfCheck, SoftwareTraits

MODES = ("graph", "compiled", "eager", "reference")
# Modes 1b does not have for MD, and the phase that brings each (SPEC §16; owner decision D6).
MODE_REFUSALS = {
    "compiled": "mode 'compiled' for MD is not in 1b: the compiled MD tier comes in phase 3 (SPEC §16, owner "
                "decision D6)",
    "eager": "mode 'eager' for MD is not in 1b: the eager tier comes with the fallback algorithm in phase 2 "
             "(SPEC §9.4, §16)",
    "reference": "mode 'reference' is refused in 1b: reference MD needs its own loop over MACE's calculator "
                 "(owner decision D6)",
}
# 1b's wording of refusals that FixedPlanner phrases for 1a.
TEXTS = {
    "inexact-padding": lambda text: text.replace("so the graph tier is refused (SPEC §9.3 rule 4)",
                                                 "so the graph tier is refused (SPEC §9.3 rule 4); compiled MD "
                                                 "tier: phase 3 (owner decision D6)"),
    "constraints": lambda text: text.replace("refused in 1a", "refused in 1b"),
}


def detect_software(device: str = "cuda:0", modules: Mapping[str, Any] | None = None) -> SoftwareTraits:
    """The software traits of SPEC §9.2 in this process; imports neither MACE nor OpenEquivariance.

    ``mace_imported_first`` is empty when the OEq runtime is prepared, or when no module that
    makes it too late has been imported yet (``loading.hybrid_blockers``).
    """
    from fastmace import _patches, kernels
    from fastmace.model import loading

    try:
        found = importlib.util.find_spec("openequivariance") is not None
    except (ImportError, ValueError) as error:
        oeq = (False, f"openequivariance cannot be looked up ({type(error).__name__}: {error})")
    else:
        oeq = ((True, "openequivariance is installed (its extension is built when the hybrid model is loaded)")
               if found else (False, "openequivariance is not installed"))
    triton, nvrtc = kernels.triton_available(), kernels.nvrtc_available()
    error = loading._ops_import_error()
    fused = (True, "cuequivariance_ops_torch imports") if error is None else (
        False, f"cuequivariance_ops_torch cannot be imported ({error}); cuEquivariance would run its naive "
               "kernels (MACE PR 1614)")
    blockers = () if _patches.record_of("oeq_runtime") is not None else tuple(loading.hybrid_blockers(modules))
    return SoftwareTraits(oeq=oeq, triton=(triton.available, triton.reason), nvrtc=(nvrtc.available, nvrtc.reason),
                          fused_cueq=fused, mace_imported_first=blockers)


def default_planner(force_field) -> MinimalPlanner:
    """The planner of ``fm.ForceField.evaluate``: CUDA if available, else the CPU without compilation."""
    import torch

    if torch.cuda.is_available():
        return MinimalPlanner()
    return MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"))


def _cuda_initialised() -> bool:
    """Whether this process has initialised CUDA (without importing torch to find out)."""
    import sys

    torch = sys.modules.get("torch")
    return torch is not None and bool(torch.cuda.is_initialized())


def _in_child(plan, reason):
    """``plan`` marked to run in a child through the subprocess runner (SPEC §8.7, 2-T07)."""
    return dataclasses.replace(plan, runner="subprocess", decisions=(*plan.decisions, ("runner", "subprocess", reason)))


def _symbols(numbers):
    from ase.data import chemical_symbols

    return " ".join(chemical_symbols[z] for z in sorted(numbers))


def species_numbers(species) -> tuple[int, ...]:
    """Symbols or atomic numbers as sorted, distinct atomic numbers (a set: order and repeats do not matter)."""
    from ase.data import atomic_numbers

    if isinstance(species, (str, int)):
        species = [species]
    out = set()
    for item in species:
        if isinstance(item, str):
            if item not in atomic_numbers:
                raise ValueError(f"unknown chemical symbol {item!r} in species=")
            out.add(atomic_numbers[item])
        elif isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"species= takes chemical symbols or atomic numbers, got {item!r}")
        else:
            out.add(int(item))
    if not out:
        raise ValueError("species= is empty")
    return tuple(sorted(out))


class MinimalPlanner:
    """The rule-based planner of phase 1b.

    ``engine``, ``neighbours`` and ``polar`` are the planner-level options (``"auto"`` resolved as
    1a resolves them); ``precision`` carries ``float64_energy`` and ``tf32``, while the force
    precision itself belongs to the ``ForceField`` (``precision.dtype`` must stay ``"auto"``).
    ``software`` (default: ``detect_software`` at each plan) and ``process`` (default: the process
    record of ``fastmace.engine.constraints``) can be given for tests; ``environ`` and
    ``torch_version`` as in ``FixedPlanner``.
    """

    def __init__(self, engine: EngineConfig | None = None, neighbours: NeighbourConfig | None = None,
                 polar: PolarConfig | None = None, precision: PrecisionConfig | None = None, *, registry=None,
                 environ: Mapping[str, str] | None = None, torch_version: str | None = None,
                 software: SoftwareTraits | None = None, process=None):
        self.engine = engine or EngineConfig()
        self.neighbours = neighbours or NeighbourConfig()
        self.polar = polar or PolarConfig()
        self.precision = precision or PrecisionConfig()
        if self.precision.dtype != "auto":
            raise ValueError("the force precision belongs to the ForceField (fm.ForceField(precision=...)); leave "
                             "MinimalPlanner's PrecisionConfig.dtype at 'auto'")
        self.registry = registry
        self.environ = environ
        self.torch_version = torch_version
        self.software = software
        if process is None:
            from fastmace.engine import constraints

            process = constraints.PROCESS
        self.process = process

    # ------------------------------------------------------------------ helpers

    def _fixed(self, rewrites, precision):
        if isinstance(rewrites, str):
            if rewrites != "auto":
                raise ValueError(f"rewrites must be 'auto' or a RewriteConfig, got {rewrites!r}")
            rewrites = RewriteConfig()
        elif not isinstance(rewrites, RewriteConfig):
            raise TypeError(f"rewrites must be 'auto' or a RewriteConfig, got {type(rewrites).__name__}")
        return fixed.FixedPlanner(rewrites=rewrites, engine=self.engine,
                                  precision=dataclasses.replace(self.precision, dtype=precision),
                                  neighbours=self.neighbours, polar=self.polar, registry=self.registry,
                                  environ=self.environ, torch_version=self.torch_version)

    def _device_type(self):
        import torch

        return torch.device(self.engine.device).type

    def hardware(self) -> Hardware:
        """The engine device's ``Hardware``, after the process's allocator settings (as ``FixedPlanner``)."""
        return self._fixed("auto", "float32").hardware()

    def _backend(self, book, software):
        """(backend, Reason) of the backend rule (§9.4, 1a-T00, §8.7), with refusals in ``book``."""
        from fastmace.engine import constraints

        request = self.engine.backend
        compiled = self.engine.compile_mode != "none"
        family = self.process.family
        if self._device_type() != "cuda":
            code, why = "backend-cueq-cpu", "CPU device: the hybrid backend needs CUDA"
        elif not software.oeq[0]:
            code, why = "backend-cueq-oeq-unavailable", f"OpenEquivariance cannot build: {software.oeq[1]}"
        elif software.mace_imported_first:
            code, why = ("backend-cueq-mace-imported-first",
                         f"{', '.join(software.mace_imported_first)} was imported before FastMACE could prepare "
                         "the OpenEquivariance runtime (SPEC §9.4, 1a-T00); import fastmace and plan before "
                         "importing MACE")
        elif compiled and family == constraints.AUTOGRAD_FAMILY:
            code, why = ("backend-cueq-process-family",
                         f"this process already compiled a {family} model ({self.process.family_owner}); one "
                         "compile family per process (SPEC §8.7)")
        else:
            code, why = "backend-hybrid", "OpenEquivariance builds and the process allows it (SPEC §9.4)"
        if request == "auto":
            backend = "hybrid" if code == "backend-hybrid" else "cueq"
            return backend, Reason(f"auto: {why}", code=code)
        if request == "hybrid":
            if code not in ("backend-hybrid", "backend-cueq-cpu", "backend-cueq-process-family"):
                # The CPU device is FixedPlanner's refusal; the family is ProcessConstraintError's.
                book.refuse(f"EngineConfig.backend='hybrid' cannot be honoured: {why}", "hybrid-unavailable")
            return "hybrid", Reason("requested 'hybrid'", code="backend-hybrid")
        return "cueq", Reason("requested 'cueq'", code="backend-cueq-requested")

    def _common(self, book, traits, atoms, species, precision, software, backend, allow_naive_cueq, loaded):
        """Rules 2, 6 and D3, and the superset, software and precision decisions.

        Returns ``(superset, force precision, requested)``. The force precision is the effective one:
        on a MACE whose float64 energy forward does not hash it is promoted to float64, and the
        requested precision is returned as a ``requested`` entry (2-T04, SPEC §18 decision 2).
        PolarMACE without its hashed forward is refused (rule 1).
        """
        from fastmace.model import loading
        from fastmace.rewrites import versions

        if precision not in ("float32", "float64"):
            raise ValueError(f"force precision must be 'float32' or 'float64', got {precision!r}")
        structure = tuple(sorted({int(z) for z in atoms.numbers}))
        superset = structure if species is None else species_numbers(species)
        known = set(traits.atomic_numbers)
        if known:
            unknown = sorted((set(superset) | set(structure)) - known)
            if unknown:
                raise ModelElementError(
                    f"species {_symbols(unknown)} have no embedding in the model (it has {_symbols(known)})",
                    why="the model was not trained on them, so it has no input for them (SPEC §8.5)")
        missing = sorted(set(structure) - set(superset))
        if missing:
            book.refuse(f"species {_symbols(missing)} of the structure are not in the ForceField's declared superset "
                        f"({_symbols(superset)}) (SPEC §8.5, §9.3 rule 2)", "species-not-in-superset", 2)
        extra = sorted(set(superset) - set(structure))
        if extra:
            book.refuse(f"species superset beyond the structure ({_symbols(extra)} absent): phase 3 (§8.5); in 1b "
                        "species= equals the structure's species (owner decision D3)", "superset-beyond-structure",
                        8)
        book.decide("superset", _symbols(superset), Reason(
            "the structure's species" if species is None else "declared (species=), equal to the structure's",
            code="superset"))
        effective, promoted = versions.effective_precision(precision, polar_model=traits.polar,
                                                           float64_energy=self.precision.float64_energy)
        book.decide("force_precision", effective, promoted or "the ForceField's precision (SPEC §7.1, §9.2)")
        requested = (("force_precision", precision),) if promoted else ()
        polar_support = versions.polar_support() if traits.polar else None
        if polar_support is not None and not polar_support.supported:
            book.refuse(polar_support.reason, "polar-unqualified-source", 1)
        for name in ("oeq", "triton", "nvrtc", "fused_cueq"):
            available, reason = getattr(software, name)
            book.decide(f"software.{name}", "on" if available else "off", reason)
        book.decide("software.mace_imported_first", ", ".join(software.mace_imported_first) or "none",
                    "modules imported before FastMACE could prepare the OEq runtime")

        # Rule 6: cuEquivariance's fused kernels must load (both backends use its symmetric contraction).
        problems = []
        if self._device_type() == "cuda":
            if not software.fused_cueq[0]:
                problems.append(software.fused_cueq[1])
            if loaded is not None and getattr(loaded, "backend", None) == backend and backend in ("cueq", "hybrid"):
                problems += loading.fused_cueq_problems(loaded.model, "cuda")
        if problems and not allow_naive_cueq:
            book.refuse("cuEquivariance's fused kernels do not load, so the run would be correct but slow: "
                        + "; ".join(problems) + ". Pass allow_naive_cueq=True to run anyway (recorded; SPEC §9.2, "
                        "§9.3 rule 6)", "naive-cueq", 6)
        # Only naive kernels in use get the downgrade code: the report warns on it (1b-T22b).
        book.decide("allow_naive_cueq", "on" if allow_naive_cueq else "off", Reason(
            ("allowed: " + "; ".join(problems)) if problems and allow_naive_cueq else
            ("requested; the fused kernels load" if allow_naive_cueq else "not needed: no naive cuEq kernel")
            if not problems else "refused: naive cuEq kernels",
            code="allow-naive-cueq" if problems else "fused-cueq"))
        return superset, effective, requested

    def _claim_family(self, plan, owner, *, offer_runner=False):
        """The process's compile family (§8.7): the first compiled plan fixes it; returns the plan.

        With ``offer_runner`` (MD plans), a plan that a process constraint forbids here is returned with
        ``runner="subprocess"`` and a ``runner`` decision instead of raising ``ProcessConstraintError``
        (2-T07): a compiled plan of the other family, or ``reproducible=True`` after CUDA was
        initialised. Such a plan claims nothing in this process; the child plans for itself.
        """
        from fastmace.engine import constraints
        from fastmace.errors import ProcessConstraintError

        workload = plan.workload
        if (offer_runner and workload is not None and workload.reproducible
                and str(plan.engine.device).startswith("cuda") and _cuda_initialised()):
            return _in_child(plan, Reason(
                "reproducible=True after CUDA was initialised in this process: CUBLAS_WORKSPACE_CONFIG and "
                "deterministic algorithms must be set before (SPEC §8.7, owner decision 14), so the plan runs in a "
                "child process", code="runner-subprocess-reproducible"))
        if plan.engine.compile_mode == "none":
            return plan
        family = constraints.OEQ_FAMILY if plan.engine.backend == "hybrid" else constraints.AUTOGRAD_FAMILY
        try:
            self.process.claim_family(family, owner)
        except ProcessConstraintError:
            if not offer_runner:
                raise
            return _in_child(plan, Reason(
                f"this process already compiled a {self.process.family} model ({self.process.family_owner}); one "
                f"compile family per process, so this {family} plan runs in a child process (SPEC §8.7)",
                code="runner-subprocess-family"))
        return plan

    # ------------------------------------------------------------------ plans

    def plan(self, force_field, atoms, integrator: IntegratorConfig, workload, hardware: Hardware | None = None, *,
             variable_cell: bool | None = None, allow_naive_cueq: bool = False, allow_reference: bool = False,
             mode: str = "graph") -> Plan:
        """The MD plan for ``force_field`` (an ``fm.ForceField``), ``atoms``, ``integrator`` and ``workload``.

        ``variable_cell`` (default: the integrator's) says whether the run changes the cell. Raises
        ``PlanRefused`` (``SpeciesNotInSuperset``) with every reason, ``ModelElementError``, or
        ``ProcessConstraintError``; nothing is loaded onto the GPU or compiled.
        """
        superset = force_field.superset_for(atoms)
        return self.plan_traits(force_field.traits, atoms, integrator, workload, hardware or self.hardware(),
                                precision=force_field.precision, rewrites=force_field.rewrites, species=superset,
                                loaded=force_field.model, variable_cell=variable_cell,
                                allow_naive_cueq=allow_naive_cueq, allow_reference=allow_reference, mode=mode)

    def plan_traits(self, traits, atoms, integrator: IntegratorConfig, workload, hardware: Hardware, *,
                    precision: str, rewrites: Any = "auto", species=None, loaded=None,
                    variable_cell: bool | None = None, allow_naive_cueq: bool = False,
                    allow_reference: bool = False, mode: str = "graph") -> Plan:
        """``plan`` from model ``traits`` (``capabilities.ModelTraits``) and the ForceField's facts.

        ``precision``, ``rewrites`` and ``species`` are the ForceField's; ``loaded`` is its model, if
        loaded (rule 6 is checked on it when it already has the plan's backend).
        """
        from fastmace.planner.workload import Workload

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not isinstance(workload, Workload):
            raise TypeError(f"workload must be an fm.Workload, got {type(workload).__name__}")
        software = self.software or detect_software(self.engine.device)
        book = fixed._Book()
        book.decide("mode", mode, Reason("molecular dynamics on the graph tier (SPEC §8.1)" if mode == "graph"
                                         else MODE_REFUSALS[mode], code=f"mode-{mode}"))
        if mode != "graph":
            book.refuse(MODE_REFUSALS[mode], f"mode-{mode}")
        if allow_reference:
            book.refuse("allow_reference=True: the reference tier is refused in 1b (owner decision D6)",
                        "reference-tier-not-in-1b", 1)
        npt = integrator.ensemble == "npt"
        if variable_cell and not npt:
            book.refuse(f"a variable cell needs a barostat integrator, got ensemble={integrator.ensemble!r} "
                        "(SPEC §4.4, §9.3 rule 1)", "variable-cell-without-barostat", 1)
        if workload.reproducible:
            book.refuse("reproducible=True: no (backend, model family) is qualified for bitwise reproducibility yet; "
                        "phase 2 qualifies the first (SPEC §9.3 rule 5, §16)", "reproducible-no-family-qualified", 5)
        if workload.atom_count_changes:
            book.refuse("atom_count_changes=True: changing atom counts is not in 1.0 (SPEC §8.6, §9.3 rule 7)",
                        "atom-count-changes", 7)
        backend = self._backend(book, software)
        superset, precision, requested = self._common(book, traits, atoms, species, precision, software, backend[0],
                                                      allow_naive_cueq, loaded)
        self._workload_decisions(book, workload)
        parts = self._fixed(rewrites, precision)._resolve(book, traits, atoms, integrator, hardware,
                                                          backend_choice=backend, texts=TEXTS)
        book.raise_refusals()
        parts["requested"] = parts["requested"] + requested
        plan = Plan(**parts, superset=superset, workload=workload, software=software,
                    allowances=("allow_naive_cueq",) if allow_naive_cueq else (), reason="minimal plan (phase 1b)")
        plan = self._claim_family(plan, f"a {plan.engine.backend} MD plan", offer_runner=True)
        return plan

    def _workload_decisions(self, book, workload):
        book.decide("workload.max_temperature_K", f"{workload.max_temperature_K:g} K", Reason(
            "recorded (rule 11); skin and edge buckets as in phase 1a: adaptive skin and headroom are phase 3",
            code="max-temperature-recorded"))
        every = workload.host_callbacks_every
        if every is None:
            book.decide("workload.host_callbacks_every", "none", Reason("no host callbacks", code="no-host-callbacks"))
        else:
            book.decide("workload.host_callbacks_every", f"graph, host sync every {every} steps", Reason(
                "rule 9 (owner decision D6): the graph tier stays, with a host sync at each callback; the compiled "
                "tier is phase 3", code="host-callbacks-graph-sync"))
        moves = ", ".join(workload.mc_moves) or "none"
        book.decide("workload.mc_moves", moves, Reason(
            "no MC" if not workload.mc_moves else "MC: phase 3; Simulation.run refuses mc= blocks",
            code="no-mc" if not workload.mc_moves else "mc-phase-3"))
        book.decide("workload.species_changes", "on" if workload.species_changes else "off",
                    "rule 8 is not in 1b: species-dependent rewrites are built for the structure's species")
        book.decide("workload.needs_per_atom", ", ".join(sorted(workload.needs_per_atom)) or "none", "recorded")

    def plan_evaluation(self, force_field, atoms, hardware: Hardware | None = None, *,
                        allow_naive_cueq: bool = False) -> Plan:
        """The one-off evaluation plan (``fm.ForceField.evaluate``, ``fm.calc``): compiled, with stress, no workload."""
        superset = force_field.superset_for(atoms)
        return self.plan_evaluation_traits(force_field.traits, atoms, hardware or self.hardware(),
                                           precision=force_field.precision, rewrites=force_field.rewrites,
                                           species=superset, loaded=force_field.model,
                                           allow_naive_cueq=allow_naive_cueq)

    def plan_evaluation_traits(self, traits, atoms, hardware: Hardware, *, precision: str, rewrites: Any = "auto",
                               species=None, loaded=None, allow_naive_cueq: bool = False) -> Plan:
        """``plan_evaluation`` from model ``traits``; the rewrite set is the graph plan's (1a's resolution)."""
        software = self.software or detect_software(self.engine.device)
        book = fixed._Book()
        book.decide("mode", "compiled", Reason("one-off evaluation: compiled model, no CUDA graphs (SPEC §7.4)",
                                               code="mode-evaluation"))
        backend = self._backend(book, software)
        superset, precision, requested = self._common(book, traits, atoms, species, precision, software, backend[0],
                                                      allow_naive_cueq, loaded)
        parts = self._fixed(rewrites, precision)._resolve(book, traits, atoms, None, hardware, backend_choice=backend,
                                                          evaluation=True, texts=TEXTS)
        book.raise_refusals()
        parts["requested"] = parts["requested"] + requested
        plan = Plan(**parts, superset=superset, software=software,
                    allowances=("allow_naive_cueq",) if allow_naive_cueq else (),
                    reason="one-off evaluation (phase 1b)")
        self._claim_family(plan, f"a {plan.engine.backend} evaluation plan")
        return plan

    # ------------------------------------------------------------------ build

    def _execution_model(self, plan, force_field):
        """The plan's converted copy of the ForceField's pristine model; rule 6 refusals as ``PlanRefused``."""
        from fastmace.errors import NaiveCueqError

        try:
            return force_field.execution_model(plan)
        except NaiveCueqError as error:
            raise PlanRefused.from_reasons([Reason(str(error), rule=6, code="naive-cueq")]) from error

    def _check_atoms(self, plan, atoms):
        structure = tuple(sorted({int(z) for z in atoms.numbers}))
        if plan.superset and structure != plan.superset:
            raise SpeciesNotInSuperset.from_reasons([Reason(
                f"the structure's species ({_symbols(structure)}) are not the plan's superset "
                f"({_symbols(plan.superset)}); plan again for this structure (SPEC §8.5)", rule=2,
                code="species-not-in-superset")])

    def build(self, plan: Plan, force_field, atoms, *, compile_cache=None, check: bool = True,
              temperature_K: float | None = None):
        """The ready ``GraphEngine`` of the MD ``plan``: an execution copy of the ForceField's model, its
        force field, state, integrator (at ``temperature_K``, default the integrator's target or 0) and
        engine. The only place 1b's public API gets an engine from (SPEC §5)."""
        if plan.tier != fixed.TIER:
            raise ValueError(f"MinimalPlanner.build builds tier {fixed.TIER!r} plans, not {plan.tier!r}; "
                             "use build_evaluation for an evaluation plan")
        self._check_atoms(plan, atoms)
        model = self._execution_model(plan, force_field)
        if temperature_K is None:
            temperature_K = plan.integrator.temperature_K or 0.0
        engine = self._fixed(RewriteConfig(), plan.precision.dtype).build(
            plan, model, atoms, temperature_K=temperature_K, compile_cache=compile_cache, constraints=self.process,
            check=check)
        # The built plan with its per-record self-checks (SPEC §6.4, 2-T04); the caller records it.
        engine.plan = dataclasses.replace(plan, selfchecks=tuple(
            SelfCheck.from_mapping(record) for record in engine.force_field.selfcheck_records()))
        return engine

    def build_evaluation(self, plan: Plan, force_field, atoms, *, compile_cache=None, check: bool = True):
        """The ``MACEForceField`` of an evaluation ``plan`` (``evaluate`` path, with stress, no graphs)."""
        from fastmace.model.forces import MACEForceField
        from fastmace.rewrites import registry as reg

        if plan.tier != "compiled":
            raise ValueError(f"build_evaluation builds evaluation plans, not tier {plan.tier!r}")
        self._check_atoms(plan, atoms)
        model = self._execution_model(plan, force_field)
        inner = self._fixed(RewriteConfig(), plan.precision.dtype)
        inner.configure_process(plan.engine)
        registry = self.registry or reg.default_registry()
        resolution = inner.rewrite_resolution(plan, model.model, registry)
        return MACEForceField(model, atoms, rewrites=resolution, engine=plan.engine, neighbours=plan.neighbours,
                              polar=plan.polar or PolarConfig(), compute_stress=True, compile_cache=compile_cache,
                              registry=registry, check=check, float64_energy=plan.precision.float64_energy)


__all__ = ("MODES", "MinimalPlanner", "SoftwareTraits", "default_planner", "detect_software", "species_numbers")
