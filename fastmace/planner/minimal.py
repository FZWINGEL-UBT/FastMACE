"""The minimal planner of phase 1b: rules, not tables; refusals with reasons (SPEC §9.1-§9.5, §8.7).

Plan task 1b-T12. ``MinimalPlanner`` plans for an ``fm.ForceField``, an ``Atoms``, an integrator
config and a ``Workload`` (§9.2), and builds what the plan describes. ``Simulation``, ``fm.md``,
``fm.calc`` and the ``fastmace.ase`` classes get their engines and force-field copies only from it.

What 1b decides (the rules of §9.3 that have their tiers in 1b):

- **hard constraints 1-7**, every reason in one ``PlanRefused`` (``SpeciesNotInSuperset`` when rule 2
  is among them): non-3D-PBC, constraints, a variable cell without a barostat integrator,
  a structure species outside the
  superset; PolarMACE on plain compiled cuEq; inexact padding for the graph tier ("compiled MD
  tier: phase 3", D6); ``reproducible=True`` outside plain cuEq MH-1 with the unfused convolution
  and deterministic kernels (2-T16, ``planner.reproducible``; a qualified plan prepares the
  process at ``build`` and checks two evaluations there); naive cuEq kernels unless
  ``allow_naive_cueq``; ``atom_count_changes``;
- **owner decision D3**: the superset must equal the structure's species (compared as sets);
  a larger one is phase 3 (§8.5);
- **mode**: graph; eager (2-T02, SPEC §8.3): the graph plan's backend, rewrites and precision
  on the graph engine without capture or compilation (no compile family claimed, no speculative
  graphs, rule 4 lifted: exact edge counts); or reference (2-T03, SPEC §8.4): the reference
  calculator as the force field of FastMACE's integrators on an uncaptured, unpadded
  ``GraphEngine``, every rewrite off, the record's stock uncompiled cuEq backend.
  ``allow_reference=True`` is recorded (rule 1) and lifts no §4.4 refusal, since the tier shares
  the integrators; ``reference_fits`` decides whether the tier fits (§9.4 step 5). Every other
  mode refuses with the phase that brings it. ``plan_evaluation`` plans the one-off compiled
  evaluation of ``fm.ForceField.evaluate`` and ``fm.calc``;
- **backend** (§9.4, 1a-T00): hybrid when OpenEquivariance can build and MACE was not imported
  before FastMACE could prepare it, else plain cuEq with the reason. The process's compile family
  is respected (§8.7): the first compiled plan fixes it, a later plan of the other family raises
  ``ProcessConstraintError``;
- **preferences in 1b**: rule 9 keeps the graph tier with a host sync at each callback (D6),
  rule 10 takes the stress path and k-vector headroom from the integrator, rule 11 (3-T05)
  sizes the skin, pair-buffer headroom and re-capture budget of MC and species-change workloads
  (``neighbors.adaptive``) and records what it would give MD-only ones, which keep 1a's sizes.
  Rule 12 is not in 1b.
- **D3 (3b-T02)**, with ``d3_cutoff_A`` given: rule 11 sizes the D3 list's own skin
  (``NeighbourConfig.d3_skin``) for every workload, MD-only too, from the fastest atom and, under a
  barostat, its thermal strain (``neighbors.adaptive.rule_11_d3``); rule 12 records the D3 list's pairs
  and memory and warns above a quarter of the GPU memory (``rule_12_d3``).

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
import warnings
from collections.abc import Mapping
from typing import Any

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig, PolarConfig, PrecisionConfig, RewriteConfig
from fastmace.errors import ModelElementError, PlanRefused, Reason, SpeciesNotInSuperset
from fastmace.rewrites import versions

from . import fixed
from . import reproducible as reproducible_mode
from .plan import Hardware, Plan, SelfCheck, SoftwareTraits

MODES = ("graph", "compiled", "eager", "reference")
# Modes the minimal planner does not have for MD, and the phase that brings each (SPEC §16; owner decision D6).
# Empty since 3-T06 brought the compiled MD tier.
MODE_REFUSALS: dict[str, str] = {}
# Rule 9 as amended by decision 38 (3-T06b): host callbacks keep the graph tier. 3-T06 measured, on the local
# RTX 5060 Ti with MH-1 hybrid float32, compiled 1.3-4.7x slower than graph with a callback every 1, 5 or 10 steps.
RULE_9_REASON = ("rule 9 (decision 38): host callbacks keep the graph tier, with a host sync at each callback; with "
                 "a callback every 1-10 steps the compiled tier was 1.3-4.7x slower (3-T06, NaCl and NbCl5)")
# The modes with an MD tier, and their mode decision.
MODE_TIERS = {
    "graph": Reason("molecular dynamics on the graph tier (SPEC §8.1)", code="mode-graph"),
    "compiled": Reason("molecular dynamics on the compiled tier (mode='compiled'): the graph plan's backend, "
                       "rewrites and precision on the compiled model, without CUDA graphs (SPEC §8.2)",
                       code="mode-compiled-tier"),
    "eager": Reason("molecular dynamics on the eager tier: the graph plan's backend, rewrites and precision, "
                    "without CUDA graphs or compilation (SPEC §8.3)", code="mode-eager-tier"),
    "reference": Reason("molecular dynamics on the reference tier: MACE's calculator with the reference record's "
                        "corrections, FastMACE's integrators, noise, output and restart (SPEC §8.4)",
                        code="mode-reference"),
}
EAGER_CAPTURE = "eager tier: no CUDA graphs"
COMPILED_CAPTURE = "compiled tier: no CUDA graphs"
REFERENCE_TIER = "reference"
# The reference tier's reasons (plan task 2-T03; SPEC §8.4).
REFERENCE_REASONS = {
    "tier": "reference tier: MACE's calculator through the versioned reference record (SPEC §8.4, §12.1), no "
            "CUDA graphs, no padding, MACE's own neighbour list",
    "backend": "reference tier: the record's stock calculator, MACE's cuEquivariance model (SPEC §12.1)",
    "rewrites": "reference tier: no rewrites (SPEC §8.4)",
    "capture": "reference tier: eager stages, no CUDA graphs",
    "compile": "reference tier: MACE's uncompiled model (SPEC §8.4)",
    "pipelined": "reference tier: every step checked (the reference evaluates on a host copy of the positions)",
}
# Refusals of the graph tier that do not apply to the reference tier: it has no padding and no compiled model.
REFERENCE_LIFTS = ("inexact-padding", "polar-plain-compiled-cueq")
# 1b's wording of refusals that FixedPlanner phrases for 1a.
TEXTS = {
    "inexact-padding": lambda text: text.replace("so the graph tier is refused (SPEC §9.3 rule 4)",
                                                 "so the graph tier is refused (SPEC §9.3 rule 4); pass "
                                                 'mode="compiled", or no mode, for the compiled tier with exact '
                                                 "edge counts"),
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


def torch_device_type(device: str) -> str:
    import torch

    return torch.device(device).type


def _error_text(error) -> str:
    from fastmace.planner.fallback import error_text

    why = getattr(error, "why", None)
    return error_text(error, getattr(error, "what", None) or None) + (f" ({why})" if why else "")


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


def check_mc_moves(moves, *, superset, charge_inputs: bool, fixcm: bool,
                   projection: str = "mass_weighted") -> tuple[tuple[str, str, str], ...]:
    """Check swap and transmutation records against the force field and the integrator (SPEC §10.4; 3-T09).

    - Every species once: a swap's two species and a transmutable set's members differ as atomic numbers
      (code ``mc-same-species``).
    - Rule 2: every species a move names is in ``superset`` (atomic numbers), code
      ``species-not-in-superset`` (``SpeciesNotInSuperset``).
    - ``fixcm`` (the integrator removes the centre-of-mass noise): the single-atom ``rescale`` and ``redraw``
      would change the total momentum behind its back, so they are refused (code ``mc-velocities-fixcm``);
      ``redraw_all`` draws (3N - 3) kT, so it needs the integrator's ``projection`` (``fixcm_projection``) to
      be ``"mass_weighted"``: with ``"ase"`` (or ``"auto"`` outside a plan, which reads as ``"ase"``) it is
      refused (code ``mc-redraw-all-ase-projection``; 3-T22b).
    - Charge and spin, only when the model or head takes charge or spin inputs (``charge_inputs``): every
      pair of a transmutable set needs ΔZ − Δq even (code ``transmute-charge-parity``,
      ``ChargeParityRefused``); otherwise the decision records "parity: not applicable".

    Raises with every reason at once; returns the decisions ``(dimension, choice, why)``.
    """
    from fastmace.errors import CHARGE_PARITY_CODE, ChargeParityRefused, PlanRefused, Reason
    from fastmace.mc.moves import resolved_velocities, species_number

    superset = set(int(z) for z in superset)
    reasons, parity = [], []
    for move in moves:
        kind = getattr(move, "move", None)
        if kind not in ("swap", "transmute"):
            continue
        named = ([species_number(item) for pair in move.pairs for item in pair] if kind == "swap"
                 else [species_number(item) for item in move.species])
        if len(set(named)) != len(named) or (kind == "swap" and any(
                species_number(a) == species_number(b) for a, b in move.pairs)):
            reasons.append(Reason(f"the {kind} move names one species twice ({_symbols(set(named))}); a swap needs two "
                                  "species and a transmutable set distinct ones (SPEC §10.4)", code="mc-same-species"))
        outside = sorted(set(named) - superset)
        if outside:
            reasons.append(Reason(f"the {kind} move names {_symbols(outside)}, outside the ForceField's superset "
                                  f"({_symbols(superset)}) (SPEC §8.5, §10.4)", rule=2,
                                  code="species-not-in-superset"))
        velocities = resolved_velocities(move, fixcm)
        if fixcm and velocities in ("rescale", "redraw"):
            reasons.append(Reason(
                f"the {kind} move's velocities={velocities!r} change the total momentum, and the integrator fixes "
                "the centre of mass (fixcm); use 'redraw_all' (or 'exchange' for swaps) (SPEC §10.4 \"Centre of "
                "mass\")", code="mc-velocities-fixcm"))
        if fixcm and velocities == "redraw_all" and projection != "mass_weighted":
            reasons.append(Reason(
                f"the {kind} move's velocities='redraw_all' draw (3N - 3) kT at zero momentum, and the integrator's "
                f"fixcm_projection={projection!r} holds 3N kT, so every accepted block would take 1/N of the kinetic "
                "energy; use fixcm_projection='mass_weighted' (the plan's choice for MC under fixcm) or 'exchange' "
                "for swaps (docs/design/fixcm-projection.md)", code="mc-redraw-all-ase-projection"))
        if kind == "transmute" and charge_inputs:
            charges = {species_number(species): value for species, value in move.formal_charges}
            for index, first in enumerate(named):
                for second in named[index + 1:]:
                    odd = (second - first) - (charges.get(second, 0) - charges.get(first, 0))
                    if odd % 2:
                        parity.append(Reason(
                            f"transmuting {_symbols([first])} <-> {_symbols([second])} changes Z - q by {odd}, an odd "
                            "number, which makes the spin multiplicity invalid; give formal charges so that every "
                            "ΔZ − Δq is even (SPEC §10.4 \"Charge and spin\")", code=CHARGE_PARITY_CODE))
    reasons += parity
    if reasons:
        raise (ChargeParityRefused if parity else PlanRefused).from_reasons(reasons)
    transmutes = any(getattr(move, "move", None) == "transmute" for move in moves)
    if not transmutes:
        return ()
    if charge_inputs:
        return (("parity", "checked", "ΔZ − Δq even for every pair of every transmutable set"),)
    return (("parity", "not applicable", "the model takes no charge or spin inputs"),)


class MinimalPlanner:
    """The rule-based planner of phase 1b.

    ``engine``, ``neighbours`` and ``polar`` are the planner-level options (``"auto"`` resolved as
    1a resolves them); ``precision`` carries ``float64_energy`` and ``tf32``, while the force
    precision itself belongs to the ``ForceField`` (``precision.dtype`` must stay ``"auto"``).
    ``software`` (default: ``detect_software`` at each plan) and ``process`` (default: the process
    record of ``fastmace.engine.constraints``) can be given for tests; ``environ`` and
    ``torch_version`` as in ``FixedPlanner``; ``versions`` (default: the installed ones) the package
    versions rule 5 checks (``planner.reproducible.installed_versions``).
    """

    def __init__(self, engine: EngineConfig | None = None, neighbours: NeighbourConfig | None = None,
                 polar: PolarConfig | None = None, precision: PrecisionConfig | None = None, *, registry=None,
                 environ: Mapping[str, str] | None = None, torch_version: str | None = None,
                 software: SoftwareTraits | None = None, process=None,
                 versions: Mapping[str, str | None] | None = None):
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
        self.versions = versions
        if process is None:
            from fastmace.engine import constraints

            process = constraints.PROCESS
        self.process = process

    # ------------------------------------------------------------------ helpers

    def _fixed(self, rewrites, precision, engine=None):
        if isinstance(rewrites, str):
            if rewrites != "auto":
                raise ValueError(f"rewrites must be 'auto' or a RewriteConfig, got {rewrites!r}")
            rewrites = RewriteConfig()
        elif not isinstance(rewrites, RewriteConfig):
            raise TypeError(f"rewrites must be 'auto' or a RewriteConfig, got {type(rewrites).__name__}")
        return fixed.FixedPlanner(rewrites=rewrites, engine=engine or self.engine,
                                  precision=dataclasses.replace(self.precision, dtype=precision),
                                  neighbours=self.neighbours, polar=self.polar, registry=self.registry,
                                  environ=self.environ, torch_version=self.torch_version)

    def _device_type(self):
        import torch

        return torch.device(self.engine.device).type

    def hardware(self) -> Hardware:
        """The engine device's ``Hardware``, after the process's allocator settings (as ``FixedPlanner``)."""
        return self._fixed("auto", "float32").hardware()

    def _backend(self, book, software, engine=None, reproducible=False):
        """(backend, Reason) of the backend rule (§9.4, 1a-T00, §8.7), with refusals in ``book``.

        ``reproducible``: plain cuEq (rule 5, 2-T16); an explicit hybrid is refused by rule 5.
        """
        from fastmace.engine import constraints

        engine = engine or self.engine
        request = engine.backend
        if reproducible:
            if request == "hybrid":
                return "hybrid", Reason("requested 'hybrid'", code="backend-hybrid")  # refused by rule 5
            return "cueq", Reason("reproducible=True: plain cuEq, the only qualified backend (SPEC §9.3 rule 5)",
                                  code="backend-cueq-reproducible")
        compiled = engine.compile_mode != "none"
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
        if extra:  # rule 8: graph mode stays; species-dependent rewrites are built for the superset (3-T02)
            why = Reason(f"declared (species=), beyond the structure ({_symbols(extra)} absent): graph mode, "
                         "species-dependent rewrites and tables built for the superset (SPEC §8.5, §9.3 rule 8)",
                         rule=8, code="superset")
        else:
            why = Reason("the structure's species" if species is None else "declared (species=), equal to the "
                         "structure's", code="superset")
        book.decide("superset", _symbols(superset), why)
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

    def _qualify(self, plan, traits):
        """Rule 5 (2-T16): the reproducible ``plan`` with its qualification records, or ``PlanRefused``
        naming every blocking component."""
        versions = self.versions if self.versions is not None else reproducible_mode.installed_versions()
        qualification = reproducible_mode.qualify(plan, traits, versions=versions, registry=self.registry)
        if not qualification.qualified:
            raise PlanRefused.from_reasons(list(qualification.reasons))
        return dataclasses.replace(plan, decisions=(*plan.decisions,
                                                    *reproducible_mode.decisions(qualification, versions)))

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
                and getattr(self.process, "reproducible", None) is not True
                and str(plan.engine.device).startswith("cuda") and _cuda_initialised()):
            return _in_child(plan, Reason(
                "reproducible=True after CUDA was initialised in this process: CUBLAS_WORKSPACE_CONFIG and "
                "deterministic algorithms must be set before (SPEC §8.7, owner decision 14), so the plan runs in a "
                "child process", code="runner-subprocess-reproducible"))
        # A process is reproducible or not (owner decision 14, 2-T16): the other kind runs in a child.
        kind = getattr(self.process, "reproducible", None)
        if offer_runner and workload is not None and kind is not None and kind != workload.reproducible:
            if workload.reproducible:
                return _in_child(plan, Reason(
                    f"reproducible=True in a process that is not reproducible ({self.process.reproducible_owner}): a "
                    "process is reproducible or not (SPEC §8.7, owner decision 14), so the plan runs in a child "
                    "process", code="runner-subprocess-reproducible"))
            return _in_child(plan, Reason(
                f"this process is reproducible ({self.process.reproducible_owner}): a process is reproducible or not "
                "(SPEC §8.7, owner decision 14), so this plan runs in a child process",
                code="runner-subprocess-not-reproducible"))
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
             mode: str | None = None, d3_cutoff_A: float | None = None) -> Plan:
        """The MD plan for ``force_field`` (an ``fm.ForceField``), ``atoms``, ``integrator`` and ``workload``.

        ``mode`` (default ``None``) is the execution mode; without one the planner picks the graph tier,
        or the compiled tier where rule 4 (inexact padding) says so; host callbacks keep the graph tier
        (rule 9, decision 38).
        ``d3_cutoff_A``: the largest D3 cutoff (rules 11 and 12 size its list, 3b-T02); default, with D3 on
        the force field, the larger of its record's pair and CN cutoffs. The plan carries the record
        (``Plan.dispersion``; 3b-T04).
        ``variable_cell`` (default: the integrator's) says whether the run changes the cell. Raises
        ``PlanRefused`` (``SpeciesNotInSuperset``) with every reason, ``ModelElementError``, or
        ``ProcessConstraintError``; nothing is loaded onto the GPU or compiled. A reproducible plan that
        runs here prepares the process (``planner.reproducible.prepare_process``).
        """
        record = getattr(force_field, "dispersion", None)
        if record is not None and d3_cutoff_A is None:
            from fastmace.longrange.d3_term import cutoff_A

            d3_cutoff_A = cutoff_A(record)
        superset = force_field.superset_for(atoms)
        if (workload.reproducible and str(self.engine.device).startswith("cuda") and not _cuda_initialised()
                and getattr(self.process, "reproducible", None) is not False):
            # The first reproducible plan makes the process reproducible before anything starts CUDA:
            # the hardware detection below does, and so may the gate's references (SPEC §8.7, decision 14).
            reproducible_mode.prepare_process(self.process, owner="a reproducible MD plan")
        plan = self.plan_traits(force_field.traits, atoms, integrator, workload, hardware or self.hardware(),
                                precision=force_field.precision, rewrites=force_field.rewrites, species=superset,
                                loaded=force_field.model, variable_cell=variable_cell,
                                allow_naive_cueq=allow_naive_cueq, allow_reference=allow_reference, mode=mode,
                                d3_cutoff_A=d3_cutoff_A)
        if workload.reproducible and plan.runner == "in-process":  # the CPU device, or already done above
            reproducible_mode.prepare_process(self.process, owner=f"a reproducible {plan.engine.backend} MD plan")
        return _with_dispersion(plan, record)

    def plan_traits(self, traits, atoms, integrator: IntegratorConfig, workload, hardware: Hardware, *,
                    precision: str, rewrites: Any = "auto", species=None, loaded=None,
                    variable_cell: bool | None = None, allow_naive_cueq: bool = False,
                    allow_reference: bool = False, mode: str | None = None,
                    d3_cutoff_A: float | None = None) -> Plan:
        """``plan`` from model ``traits`` (``capabilities.ModelTraits``) and the ForceField's facts.

        ``precision``, ``rewrites`` and ``species`` are the ForceField's; ``loaded`` is its model, if
        loaded (rule 6 is checked on it when it already has the plan's backend). ``d3_cutoff_A``: the
        largest D3 cutoff (pair and coordination number) when the ForceField has D3, else ``None``.
        """
        from fastmace.planner.workload import Workload

        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be None or one of {MODES}, got {mode!r}")
        if not isinstance(workload, Workload):
            raise TypeError(f"workload must be an fm.Workload, got {type(workload).__name__}")
        software = self.software or detect_software(self.engine.device)
        book = fixed._Book()
        reference = mode == REFERENCE_TIER
        mode, mode_reason = self._mode(mode, traits, workload)
        capture = versions.capture_support(traits)
        if mode == "graph" and not capture.supported:
            # 2-T04b: no capture-safe readout mask on this MACE, so a multi-head model runs eager from revision 0.
            mode, mode_reason = "eager", Reason(capture.reason, code=versions.CAPTURE_CODE)
        book.decide("mode", mode, mode_reason)
        if mode not in MODE_TIERS:
            book.refuse(MODE_REFUSALS[mode], f"mode-{mode}")
        engine = self._mode_engine(mode)
        book.decide("allow_reference", "on" if allow_reference else "off", Reason(
            "allowed (SPEC §9.3 rule 1); the reference tier runs FastMACE's integrators, so it lifts no refusal of "
            "§4.4 (3D PBC, no constraints, a barostat for a variable cell)" if allow_reference else "not requested",
            code="allow-reference" if allow_reference else "no-allow-reference"))
        npt = integrator.ensemble == "npt"
        if variable_cell and not npt:
            book.refuse(f"a variable cell needs a barostat integrator, got ensemble={integrator.ensemble!r} "
                        "(SPEC §4.4, §9.3 rule 1)", "variable-cell-without-barostat", 1)
        if workload.reproducible:
            for reason in reproducible_mode.explicit_conflicts(engine):
                book.refuse(reason, reason.code)
        if workload.atom_count_changes:
            book.refuse("atom_count_changes=True: changing atom counts is not in 1.0 (SPEC §8.6, §9.3 rule 7)",
                        "atom-count-changes", 7)
        if reference:
            backend = ("cueq", Reason(REFERENCE_REASONS["backend"], code="backend-reference-stock"))
        else:
            backend = self._backend(book, software, engine, workload.reproducible)
        superset, precision, requested = self._common(book, traits, atoms, species, precision, software, backend[0],
                                                      allow_naive_cueq, loaded)
        rule_11 = self._rule_11(traits, atoms, superset, integrator, workload)
        self._workload_decisions(book, workload, rule_11, mode)
        if reference:
            parts = self._reference_parts(book, traits, atoms, integrator, hardware, rewrites, precision, backend,
                                          rule_11, mc=(workload.mc_moves, workload.mc_energy_precision))
        else:
            parts = self._fixed(rewrites, precision, engine)._resolve(book, traits, atoms, integrator, hardware,
                                                                      backend_choice=backend, texts=TEXTS,
                                                                      reproducible=workload.reproducible,
                                                                      rule_11=rule_11,
                                                                      mc=(workload.mc_moves,
                                                                          workload.mc_energy_precision))
            if d3_cutoff_A is not None:
                parts["neighbours"] = self._d3_list(book, parts["neighbours"], atoms, superset, integrator,
                                                    workload, hardware, float(d3_cutoff_A))
                parts["decisions"] = tuple(book.decisions)
        if mode == "eager":
            self._eager(book, parts, traits)
        elif mode == fixed.COMPILED_TIER:
            self._compiled(book, parts, traits, mode_reason)
        book.raise_refusals()
        parts["requested"] = parts["requested"] + requested
        allowances = tuple(name for name, used in (("allow_naive_cueq", allow_naive_cueq),
                                                   ("allow_reference", allow_reference)) if used)
        plan = Plan(**parts, superset=superset, workload=workload, software=software, allowances=allowances,
                    reason="reference tier (phase 2)" if reference else "minimal plan (phase 1b)")
        if workload.reproducible:
            plan = self._qualify(plan, traits)
        plan = self._claim_family(plan, f"a {plan.engine.backend} MD plan", offer_runner=True)
        return plan

    def _reference_parts(self, book, traits, atoms, integrator, hardware, rewrites, precision, backend, rule_11=None,
                         mc=None):
        """The reference tier's plan parts (plan task 2-T03): 1a's resolution on the uncompiled stock backend,
        then every rewrite off, no capture, no pipelining, and the graph tier's padding refusals lifted."""
        from fastmace.reference import records

        engine = dataclasses.replace(self.engine, backend="cueq", compile_mode="none", pipelined="off",
                                     speculative_graph=False,
                                     cueq_conv_fusion="on" if records.STOCK["cueq_conv_fusion"] else "off")
        inner = fixed._Book()
        parts = self._fixed(rewrites, precision, engine)._resolve(inner, traits, atoms, integrator, hardware,
                                                                  backend_choice=backend, texts=TEXTS, rule_11=rule_11,
                                                                  mc=mc)
        book.refusals.extend(reason for reason in inner.refusals if reason.code not in REFERENCE_LIFTS)
        toggles = [name for name in fixed.REWRITE_FIELDS if getattr(parts["rewrites"], name) in ("on", "off")]
        replaced = {"tier": (REFERENCE_TIER, REFERENCE_REASONS["tier"]),
                    "capture": ("off", REFERENCE_REASONS["capture"]),
                    "EngineConfig.compile_mode": ("none", REFERENCE_REASONS["compile"]),
                    "EngineConfig.pipelined": ("off", REFERENCE_REASONS["pipelined"]),
                    "EngineConfig.speculative_graph": ("off", REFERENCE_REASONS["pipelined"]),
                    "RewriteConfig.polar_uvu": ("off", REFERENCE_REASONS["rewrites"]),
                    "RewriteConfig.inplace_block_linear": ("off", REFERENCE_REASONS["rewrites"]),
                    **{f"RewriteConfig.{name}": ("off", REFERENCE_REASONS["rewrites"]) for name in toggles}}
        for item, choice, reason in inner.decisions:
            book.decide(item, *replaced.get(item, (choice, reason)))
        parts.update(
            tier=REFERENCE_TIER,
            rewrites=dataclasses.replace(parts["rewrites"], **dict.fromkeys(toggles, "off"), polar_uvu=False,
                                         inplace_block_linear=False),
            engine=dataclasses.replace(parts["engine"], compile_mode="none", pipelined="off", speculative_graph=False),
            decisions=tuple(book.decisions),
            requested=self._fixed(rewrites, precision)._requested())
        return parts

    @staticmethod
    def _mode(mode, traits, workload):
        """``(mode, reason)``: the given mode, or without one the graph tier unless rule 4 plans the compiled
        tier (3-T06). Host callbacks keep the graph tier (rule 9, decision 38). An explicit ``"graph"`` with
        inexact padding is refused by rule 4, a hard rule, in ``_resolve``."""
        if mode is not None:
            return mode, MODE_TIERS.get(mode) or Reason(MODE_REFUSALS[mode], code=f"mode-{mode}")
        if not traits.padding_is_exact:
            return fixed.COMPILED_TIER, Reason(
                "no mode given: padded edges would change this model's energies, so the compiled tier with exact "
                "edge counts (SPEC §9.3 rule 4, §8.2)", rule=4, code="mode-compiled-rule-4")
        return "graph", MODE_TIERS["graph"]

    def _mode_engine(self, mode):
        """The planner's ``EngineConfig`` for ``mode``: the eager tier compiles nothing and captures no graphs;
        the compiled tier captures no graphs and, when it compiles, marks the edge dimension dynamic."""
        if mode == "eager":
            return dataclasses.replace(self.engine, compile_mode="none", speculative_graph=False)
        if mode == fixed.COMPILED_TIER:
            return dataclasses.replace(self.engine, speculative_graph=False,
                                       dynamic_edges=self.engine.compile_mode != "none")
        return self.engine

    def _eager(self, book, parts, traits):
        """The graph tier's decisions made eager (2-T02): tier, capture, and rule 4 lifted (exact edge counts).

        ``parts`` holds the decisions as ``_resolve`` left them, and the eager engine's options as if
        requested; the requested options stay the caller's.
        """
        book.refusals = [reason for reason in book.refusals if getattr(reason, "code", None) != "inexact-padding"]
        replaced = {
            "tier": (fixed.EAGER_TIER, "the graph engine without CUDA graphs or compilation, exact edge counts "
                                       "(SPEC §8.3)"),
            "capture": ("off", f"{EAGER_CAPTURE} (SPEC §8.3)"),
            "EngineConfig.compile_mode": ("none", "eager tier: no compilation (SPEC §8.3)"),
            "EngineConfig.speculative_graph": ("off", "eager tier: speculative combined graphs need CUDA graphs"),
        }
        book.decisions = [(key, *replaced[key]) if key in replaced else (key, choice, reason)
                          for key, choice, reason in book.decisions]
        if not traits.padding_is_exact:
            book.decide("padding", "exact edge counts", Reason(
                "edge padding is not exact for this model; the eager tier uses exact edge counts, so rule 4 does "
                "not apply (SPEC §9.3 rule 4)", rule=4, code="eager-exact-edges"))
        own = set(self._fixed("auto", "float32")._requested())
        parts.update(tier=fixed.EAGER_TIER, decisions=tuple(book.decisions),
                     requested=tuple(entry for entry in parts["requested"]
                                     if entry in own or not entry[0].startswith("EngineConfig.")))

    def _compiled(self, book, parts, traits, mode_reason):
        """The graph tier's decisions on the compiled tier (3-T06): tier, capture, dynamic edges, and rule 4
        lifted (exact edge counts). ``parts`` as ``_resolve`` left them; the requested options stay the caller's."""
        book.refusals = [reason for reason in book.refusals if getattr(reason, "code", None) != "inexact-padding"]
        why = {"mode-compiled-rule-4": "rule 4: padded edges would change this model's energies"}.get(
            getattr(mode_reason, "code", None), "mode='compiled'")
        replaced = {
            "tier": (fixed.COMPILED_TIER, "the graph engine on the compiled model without CUDA graphs, exact edge "
                                          "counts (SPEC §8.2)"),
            "capture": ("off", f"{COMPILED_CAPTURE} ({why}; SPEC §8.2)"),
            "EngineConfig.speculative_graph": ("off", "compiled tier: speculative combined graphs need CUDA graphs"),
        }
        book.decisions = [(key, *replaced[key]) if key in replaced else (key, choice, reason)
                          for key, choice, reason in book.decisions]
        dynamic = parts["engine"].dynamic_edges
        book.decide("EngineConfig.dynamic_edges", "on" if dynamic else "off",
                    "exact edge counts: the edge dimension is dynamic from the first call, so a neighbour rebuild "
                    "with a new edge count does not compile again (2-T13)" if dynamic else
                    "nothing is compiled (compile_mode='none'): exact edge counts on the uncompiled model")
        if not traits.padding_is_exact:
            book.decide("padding", "exact edge counts", Reason(
                "edge padding is not exact for this model; the compiled tier uses exact edge counts, so rule 4 "
                "does not apply (SPEC §9.3 rule 4)", rule=4, code="compiled-exact-edges"))
        own = set(self._fixed("auto", "float32")._requested())
        parts.update(tier=fixed.COMPILED_TIER, decisions=tuple(book.decisions),
                     requested=tuple(entry for entry in parts["requested"]
                                     if entry in own or not entry[0].startswith("EngineConfig.")))

    def _rule_11(self, traits, atoms, superset, integrator, workload):
        """``(applies, Rule11)``: rule 11's sizes for the workload, and whether they are used (3-T05).

        The lightest mass is the lightest of the structure's masses and the superset species' standard
        masses (conductor answer 3-T05 5). The rule applies to workloads with MC moves or declared
        species changes; MD-only plans keep phase 1a's sizes (answer 1-2).
        """
        from ase.data import atomic_masses

        from fastmace.neighbors.adaptive import rule_11

        masses = [float(mass) for mass in atoms.get_masses()] + [float(atomic_masses[z]) for z in superset]
        sized = rule_11(max_temperature_K=workload.max_temperature_K, lightest_mass_u=min(masses),
                        timestep_fs=integrator.timestep_fs, natoms=len(atoms),
                        cutoff_A=float(traits.r_max) if traits.r_max else None,
                        edge_bucket_fraction=self.neighbours.edge_bucket_fraction,
                        max_displacement_A=workload.max_displacement_A)
        return bool(workload.mc_moves or workload.species_changes), sized

    def _d3_list(self, book, neighbours, atoms, superset, integrator, workload, hardware, cutoff_A):
        """Rules 11 and 12 for the D3 pair list (3b-T02): its skin into ``neighbours``, and its memory."""
        from ase import units
        from ase.data import atomic_masses

        from fastmace.neighbors.adaptive import D3_MEMORY_WARN_FRACTION, barostat_strain_rate, rule_11_d3, rule_12_d3
        from fastmace.neighbors.d3_list import DEFAULT_HEADROOM

        masses = [float(mass) for mass in atoms.get_masses()] + [float(atomic_masses[z]) for z in superset]
        rate = 0.0
        if integrator.ensemble == "npt":
            p_tau_fs = (integrator.barostat_tau_fs if integrator.barostat_tau_fs is not None
                        else 20.0 / integrator.friction / units.fs)  # ASE's default, 20 T_tau
            rate = barostat_strain_rate(temperature_K=workload.max_temperature_K, natoms=len(atoms),
                                        P_tau_fs=p_tau_fs, mass_factor=integrator.barostat_mass_factor,
                                        cell_mode=integrator.cell_mode)
        sized = rule_11_d3(max_temperature_K=workload.max_temperature_K, lightest_mass_u=min(masses),
                           timestep_fs=integrator.timestep_fs, natoms=len(atoms), cutoff_A=cutoff_A,
                           strain_rate_per_fs=rate)
        if self.neighbours.d3_skin == "auto":
            skin = sized.skin_A
            why = Reason(fixed._why("auto", fixed.AUTO_RULES["NeighbourConfig.d3_skin"], sized.text()), rule=11,
                         code="rule-11-d3")
        else:
            skin = float(self.neighbours.d3_skin)
            why = Reason(f"{skin:g} A kept as given (rule 11 is a preference); rule 11 would give {sized.text()}",
                         rule=11, code="rule-11-d3-given")
        book.decisions[:] = [entry for entry in book.decisions if entry[0] != "NeighbourConfig.d3_skin"]
        book.decide("NeighbourConfig.d3_skin", f"{skin:g} A", why)
        memory = rule_12_d3(natoms=len(atoms), volume_A3=float(atoms.get_volume()), radius_A=cutoff_A + skin,
                            headroom=DEFAULT_HEADROOM, gpu_memory_bytes=hardware.memory_bytes)
        free = self._free_memory_bytes(hardware)
        if free and memory.list_bytes > free:
            book.refuse(Reason(f"{memory.text()}: the D3 list's buffers alone need more than the "
                               f"{free / 2**30:.1f} GiB of free GPU memory (SPEC §9.3 rule 12)", rule=12,
                               code="rule-12-d3-memory"),
                        "rule-12-d3-memory", 12)
        if memory.warn:
            text = (f"{memory.text()}: above {D3_MEMORY_WARN_FRACTION:.0%} of the GPU memory; a D3 build may run "
                    "out of memory (SPEC §9.3 rule 12)")
            warnings.warn(text, ResourceWarning, stacklevel=4)
            book.decide("d3 list memory", "warn", Reason(text, rule=12, code="rule-12-d3-warn"))
        else:
            book.decide("d3 list memory", "fits", Reason(memory.text(), rule=12, code="rule-12-d3"))
        return dataclasses.replace(neighbours, d3_skin=skin)

    def _free_memory_bytes(self, hardware):
        """Free device memory at plan time: CUDA's own count once this process has initialised CUDA (the planner
        never initialises it for this), else the device's total memory; ``None`` off CUDA."""
        if not str(self.engine.device).startswith("cuda"):
            return None
        if _cuda_initialised():
            import torch

            return int(torch.cuda.mem_get_info(torch.device(self.engine.device))[0])
        return hardware.memory_bytes

    def _workload_decisions(self, book, workload, rule_11, mode="graph"):
        applies, sized = rule_11
        if applies:
            why = Reason(f"rule 11 sizes the skin and headroom: {sized.text()}", rule=11,
                         code="rule-11")
        else:
            why = Reason(f"MD only: phase 1a's skin and headroom; rule 11 would give {sized.text()} (it applies to "
                         "MC and species-change workloads)", rule=11, code="rule-11-md-only")
        book.decide("workload.max_temperature_K", f"{workload.max_temperature_K:g} K", why)
        every = workload.host_callbacks_every
        if every is None:
            book.decide("workload.host_callbacks_every", "none", Reason("no host callbacks", code="no-host-callbacks"))
        elif mode == fixed.COMPILED_TIER:
            book.decide("workload.host_callbacks_every", f"compiled, host callbacks every {every} steps", Reason(
                "the compiled tier (rule 4 or mode='compiled') launches every step from the host, so a callback "
                "costs no graph break (rule 9, decision 38)", rule=9, code="host-callbacks-compiled"))
        else:
            book.decide("workload.host_callbacks_every", f"{mode}, host sync every {every} steps",
                        Reason(RULE_9_REASON, rule=9, code="host-callbacks-graph-sync"))
        moves = ", ".join(workload.mc_moves) or "none"
        book.decide("workload.mc_moves", moves, Reason(
            "no MC" if not workload.mc_moves else "MC moves are attached through Simulation.run(mc=...) (SPEC §7.6)",
            code="no-mc" if not workload.mc_moves else "mc-moves"))
        book.decide("workload.species_changes", "on" if workload.species_changes else "off",
                    "rule 8 is not in 1b: species-dependent rewrites are built for the structure's species")
        book.decide("workload.needs_per_atom", ", ".join(sorted(workload.needs_per_atom)) or "none", "recorded")

    def plan_evaluation(self, force_field, atoms, hardware: Hardware | None = None, *,
                        allow_naive_cueq: bool = False) -> Plan:
        """The one-off evaluation plan (``fm.ForceField.evaluate``, ``fm.calc``): compiled, with stress, no workload."""
        superset = force_field.superset_for(atoms)
        plan = self.plan_evaluation_traits(force_field.traits, atoms, hardware or self.hardware(),
                                           precision=force_field.precision, rewrites=force_field.rewrites,
                                           species=superset, loaded=force_field.model,
                                           allow_naive_cueq=allow_naive_cueq)
        return _with_dispersion(plan, getattr(force_field, "dispersion", None))

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
        dynamic_edges = self.engine.compile_mode != "none"
        book.decide("EngineConfig.dynamic_edges", "on" if dynamic_edges else "off",
                    "exact edge counts: the edge dimension is dynamic from the first call, so a new edge count "
                    "does not compile again (2-T13)" if dynamic_edges else "nothing is compiled")
        parts = self._fixed(rewrites, precision)._resolve(book, traits, atoms, None, hardware, backend_choice=backend,
                                                          evaluation=True, texts=TEXTS)
        book.raise_refusals()
        parts["requested"] = parts["requested"] + requested
        parts["engine"] = dataclasses.replace(parts["engine"], dynamic_edges=dynamic_edges)
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
        # Within the superset, not equal to it: a superset species may be absent (rule 8; 3-T09).
        structure = tuple(sorted({int(z) for z in atoms.numbers}))
        if plan.superset and not set(structure) <= set(plan.superset):
            raise SpeciesNotInSuperset.from_reasons([Reason(
                f"the structure's species ({_symbols(structure)}) are not within the plan's superset "
                f"({_symbols(plan.superset)}); plan again for this structure (SPEC §8.5)", rule=2,
                code="species-not-in-superset")])

    def build(self, plan: Plan, force_field, atoms, *, compile_cache=None, check: bool = True,
              temperature_K: float | None = None):
        """The ready ``GraphEngine`` of the MD ``plan`` (a ``CompiledEngine`` on the compiled tier): an
        execution copy of the ForceField's model, its force field, state, integrator (at ``temperature_K``,
        default the integrator's target or 0) and engine. The only place 1b's public API gets an engine
        from (SPEC §5). A reference-tier plan gets its ``ReferenceForceField`` on an uncaptured, unpadded
        ``GraphEngine`` (``build_reference``)."""
        if plan.tier == REFERENCE_TIER:
            return self.build_reference(plan, force_field, atoms, temperature_K=temperature_K)
        tiers = (fixed.TIER, fixed.COMPILED_TIER, fixed.EAGER_TIER)
        if plan.tier not in tiers or (plan.tier == fixed.COMPILED_TIER and plan.workload is None):
            raise ValueError(f"MinimalPlanner.build builds MD plans of tiers {tiers}, not {plan.tier!r}"
                             f"{' without a workload' if plan.workload is None else ''}; "
                             "use build_evaluation for an evaluation plan")
        self._check_atoms(plan, atoms)
        reproducible = plan.workload is not None and plan.workload.reproducible
        if plan.workload is not None:  # before the model goes to the GPU (SPEC §8.7, owner decision 14)
            if reproducible:
                reproducible_mode.prepare_process(self.process, owner=f"a reproducible {plan.engine.backend} MD plan")
            else:
                self.process.claim_reproducible(False, f"a {plan.engine.backend} MD plan")
        model = self._execution_model(plan, force_field)
        if temperature_K is None:
            temperature_K = plan.integrator.temperature_K or 0.0
        engine = self._fixed(RewriteConfig(), plan.precision.dtype).build(
            plan, model, atoms, temperature_K=temperature_K, compile_cache=compile_cache, constraints=self.process,
            check=check, capture=False if plan.tier in (fixed.EAGER_TIER, fixed.COMPILED_TIER) else None)
        if reproducible:
            self._check_reproducible(engine)
        # The built plan with its per-record self-checks (SPEC §6.4, 2-T04); the caller records it.
        engine.plan = dataclasses.replace(plan, selfchecks=tuple(
            SelfCheck.from_mapping(record) for record in engine.force_field.selfcheck_records()))
        return engine

    @staticmethod
    def _check_reproducible(engine):
        """Rule 5 at build (2-T16): the records applied are deterministic, and two evaluations of the
        start positions are bitwise equal (the check of the pinned cuEq variable)."""
        from fastmace.errors import ReproducibilityError

        declared = reproducible_mode.record_determinism()
        applied = [getattr(entry, "name", entry) for entry in getattr(engine.force_field, "applied", ())]
        blocking = tuple(f"rewrite {name}" for name in applied if declared.get(name) is not True)
        if blocking:
            raise ReproducibilityError(f"reproducible=True: applied rewrites are not declared deterministic: "
                                       f"{', '.join(blocking)}", blocking=blocking)
        field, positions = engine.force_field, engine.state.positions.detach().clone()

        def evaluate():
            out = field(positions)
            return out.energy, out.forces

        kernels = ("cuEquivariance uniform_1d",) if str(engine.state.positions.device).startswith("cuda") else (
            "torch CPU kernels",)
        reproducible_mode.check_repeatable(evaluate, kernels=kernels)

    def build_reference(self, plan: Plan, force_field, atoms, *, temperature_K: float | None = None):
        """The reference tier's engine (plan task 2-T03, SPEC §8.4): the reference calculator as the force
        field of FastMACE's state, integrator and ``GraphEngine``, without capture or padding."""
        from fastmace.engine.graph import GraphEngine
        from fastmace.engine.reference import ReferenceForceField
        from fastmace.integrators.baoab import LangevinBAOAB
        from fastmace.integrators.langevin import Langevin
        from fastmace.state import SimulationState

        if plan.tier != REFERENCE_TIER:
            raise ValueError(f"build_reference builds reference-tier plans, not tier {plan.tier!r}")
        book = fixed._Book()
        if not all(atoms.pbc):
            book.refuse("periodic boundaries in all three directions (3D PBC) are required (SPEC §4.4)",
                        "non-3d-pbc", 1)
        if atoms.constraints:
            book.refuse("constraints are refused in phase 2 (SPEC §4.4)", "constraints", 1)
        book.raise_refusals()
        self._check_atoms(plan, atoms)
        if temperature_K is None:
            temperature_K = plan.integrator.temperature_K or 0.0
        npt = plan.integrator.ensemble == "npt"
        field = ReferenceForceField(force_field, atoms, device=plan.engine.device, dtype=plan.precision.dtype,
                                    polar=plan.polar, compute_stress=npt)
        state = SimulationState.from_atoms(atoms, field.device, plan.integrator.seed, compute_stress=npt)
        if npt:
            integrator = LangevinBAOAB(state, plan.integrator, temperature_K=temperature_K,
                                       **fixed.integrator_keywords(plan.integrator))
        else:
            integrator = Langevin(state, plan.integrator, temperature_K=temperature_K)
        engine = GraphEngine(plan.engine, field, integrator, state, neighbours=plan.neighbours, capture=False,
                             pad_edges=False, constraints=self.process)
        engine.plan = plan  # no rewrites, so no per-record self-checks (as build's plan, 2-T04)
        return engine

    def reference_fits(self, plan: Plan, force_field, atoms, references=None) -> tuple[bool, str]:
        """Whether the reference tier fits for ``atoms`` on ``plan``'s device (SPEC §9.4 step 5): (fits, reason).

        The gate's reference result decides where there is one (``selfcheck.References``): it ran on the
        plan's device and the production cell, or it could not run. Otherwise one trial evaluation decides:
        in this process on the CPU, and in a child process on CUDA, because an out-of-memory error leaves
        the process's CUDA context unusable (2-T05). Any error, non-zero exit or kill means "does not fit".
        No table: the structure, the model and the device decide.
        """
        from fastmace.planner import selfcheck

        if references is not None and not references.skipped:
            if references.unavailable is not None:
                return False, f"the gate's reference could not run: {references.unavailable.reason}"
            info = references.info or {}
            device = info.get("device")
            if device is not None and str(device) != str(plan.engine.device):
                return False, f"the gate's reference ran on {device}, not on the plan's device {plan.engine.device}"
            if info.get("reduced_cell"):
                return False, "the gate's reference ran on a reduced cell only, not on the production cell"
            if references.values:
                return True, f"the gate's reference ran on {plan.engine.device} on the production cell"
        if torch_device_type(plan.engine.device) == "cuda":
            copy = selfcheck.GateCopy(kind="production", atoms=atoms.copy())
            job = selfcheck._job(plan, force_field, [copy])
            try:
                selfcheck.run_child(job)
            except Exception as error:
                return False, f"the trial evaluation in a child process failed: {_error_text(error)}"
            return True, f"a trial evaluation on {plan.engine.device} in a child process ran"
        try:
            from fastmace.engine.reference import ReferenceForceField

            field = ReferenceForceField(force_field, atoms, device=plan.engine.device, dtype=plan.precision.dtype,
                                        polar=plan.polar)
            field.evaluate(atoms)
        except Exception as error:
            return False, f"the trial evaluation failed: {_error_text(error)}"
        finally:
            field = None
        return True, f"a trial evaluation on {plan.engine.device} ran"

    def build_evaluation(self, plan: Plan, force_field, atoms, *, compile_cache=None, check: bool = True):
        """The ``MACEForceField`` of an evaluation ``plan`` (``evaluate`` path, with stress, no graphs); with D3, a
        ``DispersedForceField`` around it (3b-T04)."""
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
        field = MACEForceField(model, atoms, rewrites=resolution, engine=plan.engine, neighbours=plan.neighbours,
                               polar=plan.polar or PolarConfig(), compute_stress=True, compile_cache=compile_cache,
                               registry=registry, check=check, float64_energy=plan.precision.float64_energy,
                               species=plan.superset or None)
        return fixed._with_d3(field, plan, atoms)


def _with_dispersion(plan, record):
    """``plan`` carrying the force field's D3 record (3b-T04), with its decision; unchanged without D3."""
    if record is None:
        return plan
    from fastmace.longrange import d3

    decisions = plan.decisions + (("dispersion", d3.describe(record), Reason(
        "the force field's D3 record: MACE plus D3 in the force graph, its own half list; results are "
        "non-stock (SPEC §11.2, §12.1)", code="d3"),),)
    return dataclasses.replace(plan, dispersion=record, decisions=decisions)


__all__ = ("MODES", "MinimalPlanner", "SoftwareTraits", "default_planner", "detect_software", "species_numbers")
