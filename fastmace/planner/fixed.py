"""The fixed-plan planner: one plan, the one the research engine's ``main.py`` runs (SPEC §9.1, §9.5, §16).

Plan task 1a-T15. ``FixedPlanner`` takes the caller's config objects (any field may be ``"auto"``)
and resolves them as the research engine does, in its two stages (docs/design/switch-map.md,
"Auto resolution rules"):

- **stage 1** (``workflow.parse_args``, before the model is loaded): from the backend, the graph
  engine and the other options only. ``model_settings`` gives the part that loading needs
  (backend, dtype, device, cuEq convolution fusion);
- **stage 2** (``workflow.adapt_options_to_model``): every model-dependent option that stage 1
  left on but the model's ``rewrite_support`` does not support is switched off if it was
  ``"auto"`` (``Plan.auto_options_off`` keeps the reason) and refused if it was ``"on"``.

On top of that, FastMACE's own hard rules: the workload of SPEC §4.4 (3D PBC, no constraints),
exact edge padding for the graph tier, PolarMACE never on compiled plain cuEq (zero forces), no
reassociation on plain cuEq (its probe refuses it, conductor decision after T11), streaming only
on receiver-sorted edges (decision after T07/T08), ``inplace_block_linear`` not carried over in
1a. Every refusal is collected, and ``PlanRefused`` reports them together, before anything is
loaded onto the GPU or compiled.

The plan is a frozen ``Plan`` with an ``(item, choice, reason)`` decision for every option; a
reason names the switch-map rule when the request was ``"auto"``. ``build`` turns a plan into a
ready ``GraphEngine``: it is the only code in the library that instantiates force fields,
integrators and engines. What ``main.py`` on FastMACE (1a-T17) calls::

    planner = FixedPlanner(rewrites=..., engine=..., precision=..., neighbours=..., polar=...)
    hardware = planner.hardware()            # sets the allocator first (expandable segments)
    loaded = planner.load("mh1-r2scan")      # the stage-1 backend, dtype and fusion
    plan = planner.plan(loaded, atoms, IntegratorConfig(...), hardware)
    engine = planner.build(plan, loaded, atoms, temperature_K=0.0, compile_cache=cache)

Copied and refactored from ``workflow.parse_args``, ``adapt_options_to_model``,
``GPUEngine.__init__`` and ``baoab_settings`` (ResearchModule tag research-engine-baseline-20261017).
The module holds no state; the planner keeps only the configs it was given.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from fastmace.config import (
    EngineConfig,
    IntegratorConfig,
    NeighbourConfig,
    PolarConfig,
    PrecisionConfig,
    RewriteConfig,
)
from fastmace.errors import PlanRefused, Reason

from .plan import Hardware, Plan

TIER = "graph"
EAGER_TIER = "eager"  # the graph engine without capture or compilation (SPEC §8.3, 2-T02)

# The switch map's rule for every field that admits "auto" (tests/cpu/test_config.py AUTO_RULES).
AUTO_RULES = MappingProxyType({
    "EngineConfig.backend": "backend_hybrid_for_graph_engine",
    "EngineConfig.pipelined": "on_with_graph_engine_unless_speculative",
    "EngineConfig.async_output": "on_with_graph_engine",
    "EngineConfig.cueq_conv_fusion": "on_for_plain_cueq_unless_reproducible",
    "EngineConfig.expandable_segments": "on_for_torch_2_14_unless_alloc_conf_set",
    "PrecisionConfig.dtype": "float32_for_hybrid_float64_for_cueq",
    "NeighbourConfig.sort_receivers": "on_for_hybrid_graph_engine",
    "NeighbourConfig.gpu_neighbour_list": "on_with_graph_engine",
    "RewriteConfig.factor_species": "on_for_hybrid_graph_engine_if_model_supports",
    "RewriteConfig.fold_cutoff": "on_with_factor_species_if_model_supports",
    "RewriteConfig.density_first": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.reassociate_first": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.joint_radial_input": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.fused_folded_backward": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.segmented_first_sum": "on_with_reassociate_first_and_sort_receivers",
    "RewriteConfig.streaming_conv": "on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports",
    "RewriteConfig.species_skip": "on_with_graph_engine_if_model_supports",
    "RewriteConfig.fused_radial_hidden": "never_automatic",
    "RewriteConfig.block_linear": "on_for_hybrid_graph_engine",
    "RewriteConfig.grouped_linear": "on_with_block_linear",
    "RewriteConfig.merged_residual": "on_with_block_linear",
    "RewriteConfig.single_graph_strain": "on",
    "PolarConfig.kspace_headroom": "variable_cell_0_1_else_0",
})

# adapt_options_to_model's ``requires``: option -> rewrite_support key, in its order.
STAGE2_SUPPORT = (
    ("density_first", "fold_cutoff"), ("joint_radial_input", "fold_cutoff"), ("fused_radial_hidden", "fold_cutoff"),
    ("fused_folded_backward", "fused_folded_backward"), ("species_skip", "species_skip"),
    ("streaming_conv", "streaming_conv"), ("fold_cutoff", "fold_cutoff"), ("factor_species", "factor_species"),
    ("reassociate_first", "reassociate_first"), ("segmented_first_sum", "reassociate_first"),
)
# The rewrite fields in the order the research engine resolves them (parse_args).
REWRITE_FIELDS = ("factor_species", "fold_cutoff", "density_first", "reassociate_first", "joint_radial_input",
                  "fused_folded_backward", "segmented_first_sum", "fused_radial_hidden", "streaming_conv",
                  "block_linear", "grouped_linear", "merged_residual", "species_skip", "single_graph_strain")

# Workload constraints of the rewrite records (SPEC §6.1) and what each needs from the plan.
SINGLE_GRAPH = "single_graph"  # one structure per evaluation: always, production is one structure
FIXED_SPECIES_SUPERSET = "fixed_species_superset"  # species fixed at setup: always in 1a (no MC)
SORT_RECEIVERS = "sort_receivers"  # NeighbourConfig.sort_receivers on
OEQ = "oeq"  # the hybrid backend
PER_ENGINE_EDGE_COUNTER = "per_engine_edge_counter"  # the force field's own edge count (MACEForceField)
WORKLOAD_CONSTRAINTS = (SINGLE_GRAPH, FIXED_SPECIES_SUPERSET, SORT_RECEIVERS, OEQ, PER_ENGINE_EDGE_COUNTER)

NOT_CARRIED_OVER_1A = "not carried over in 1a (SPEC §6.2: opt-in, slower)"
PLAIN_CUEQ_REASSOCIATION = ("the reassociation probe refuses plain cuEq (coefficients differ within a path; "
                            "conductor decision after T11)")


@dataclass(frozen=True)
class ModelSettings:
    """Stage-1 settings that loading needs (``fastmace.model.loading.load`` arguments)."""

    backend: str  # "hybrid" or "cueq"
    dtype: str  # "float32" or "float64"
    device: str
    cueq_conv_fusion: bool | None  # plain cuEq: True or False; the hybrid: None (not applicable)


def detect_hardware(device: str = "cuda:0") -> Hardware:
    """The ``Hardware`` record of ``device`` ("cpu" for a CPU device).

    Touches CUDA: call ``FixedPlanner.configure_process`` (or use ``FixedPlanner.hardware``) first.
    """
    import torch

    if torch.device(device).type != "cuda":
        return Hardware("cpu", (0, 0), 0)
    properties = torch.cuda.get_device_properties(torch.device(device))
    return Hardware(properties.name, (properties.major, properties.minor), int(properties.total_memory))


def integrator_keywords(config: IntegratorConfig) -> dict[str, Any]:
    """Constructor keywords of the integrator beyond its config, as ``workflow.baoab_settings`` passes them.

    NPT: ``externalstress`` in ASE units from ``effective_external_stress_GPa()`` (``-pressure`` unless
    ``external_stress_GPa`` is set: bitwise 1a's ``-pressure_GPa * units.GPa``; six Voigt components
    as an array) and ``disable_cell_langevin`` from the config (False by default, ASE's default;
    ``baoab_settings`` does not pass it). The cell mode, barostat time constant and mass factor
    travel in the config. NVT: none.
    """
    if config.ensemble != "npt":
        return {}
    import numpy as np
    from ase import units

    stress = config.effective_external_stress_GPa()
    externalstress = stress * units.GPa if isinstance(stress, float) else np.asarray(stress) * units.GPa
    return {"externalstress": externalstress, "disable_cell_langevin": bool(config.disable_cell_langevin)}


def workload_problems(names, plan: Plan, registry) -> list[str]:
    """Why the records ``names`` cannot run under ``plan`` (their workload constraints), [] if they can."""
    problems = []
    for name in names:
        for constraint in registry[name].workload_constraints:
            if constraint not in WORKLOAD_CONSTRAINTS:
                problems.append(Reason(f"{name}: unknown workload constraint {constraint!r}",
                                       code="unknown-workload-constraint"))
            elif constraint == SORT_RECEIVERS and plan.neighbours.sort_receivers != "on":
                problems.append(Reason(
                    f"{name} needs receiver-sorted edges (NeighbourConfig.sort_receivers='on'); in any other "
                    "order the streaming kernels give wrong results while every self-check passes (1a-T07)",
                    code="streaming-needs-sorted-receivers"))
            elif constraint == OEQ and plan.engine.backend != "hybrid":
                problems.append(Reason(f"{name} needs the hybrid backend (OpenEquivariance kernels)",
                                       code="rewrite-needs-hybrid"))
    return problems


class _Book:
    """Decisions and refusals collected while one plan is resolved."""

    def __init__(self):
        self.decisions = []
        self.refusals = []

    def decide(self, item, choice, reason):
        self.decisions.append((item, str(choice), reason))

    def refuse(self, reason, code, rule=None):
        """Record a refusal: a ``Reason`` with its stable ``code`` and §9.3 ``rule`` (None: an option request)."""
        self.refusals.append(reason if isinstance(reason, Reason) else Reason(reason, rule=rule, code=code))

    def raise_refusals(self):
        if self.refusals:
            raise PlanRefused.from_reasons(self.refusals)


def _toggle(value):
    return "on" if value else "off"


def _stress_text(config):
    """The integrator decision's externalstress, in ASE units, with where it came from."""
    keywords = integrator_keywords(config)
    stress = keywords["externalstress"]
    if config.external_stress_GPa is None:
        return f"{stress:.6g} (-{config.pressure_GPa:g} GPa)"
    if isinstance(stress, float):
        return f"{stress:.6g} ({config.external_stress_GPa:g} GPa)"
    return "(" + ", ".join(f"{value:.6g}" for value in stress) + ") (external_stress_GPa, Voigt)"


def _why(request, rule, explanation):
    """The reason string of one resolved option."""
    if request == "auto":
        return f"auto ({rule}): {explanation}"
    return f"requested {request!r}: {explanation}"


class FixedPlanner:
    """The fixed plan of phase 1a: what ``main.py`` runs, from the caller's (possibly "auto") configs.

    ``environ`` and ``torch_version`` default to the process's; tests pass their own so that the
    allocator rule has no side effects. ``registry`` defaults to a fresh default rewrite registry.
    """

    tier = TIER

    def __init__(self, rewrites: RewriteConfig | None = None, engine: EngineConfig | None = None,
                 precision: PrecisionConfig | None = None, neighbours: NeighbourConfig | None = None,
                 polar: PolarConfig | None = None, *, registry=None, environ: Mapping[str, str] | None = None,
                 torch_version: str | None = None):
        self.rewrites = rewrites or RewriteConfig()
        self.engine = engine or EngineConfig()
        self.precision = precision or PrecisionConfig()
        self.neighbours = neighbours or NeighbourConfig()
        self.polar = polar or PolarConfig()
        self.registry = registry
        self.environ = os.environ if environ is None else environ
        self.torch_version = torch_version

    # ------------------------------------------------------------------ stage 1: what loading needs

    def _backend(self, book, choice=None, reproducible=False):
        """(backend, dtype, cueq_conv_fusion) of stage 1, with their decisions and refusals.

        ``choice``: ``(backend, reason)`` decided by another planner (the minimal planner's backend
        rule, SPEC §9.3); default the research engine's rule. ``reproducible``: the unfused
        convolution under "auto" (owner decision 13; the minimal planner refuses an explicit "on").
        """
        engine, precision = self.engine, self.precision
        if choice is None:
            backend = "hybrid" if engine.backend == "auto" else engine.backend
            book.decide("EngineConfig.backend", backend, _why(engine.backend, AUTO_RULES["EngineConfig.backend"],
                                                              "graph engine"))
        else:
            backend, reason = choice
            book.decide("EngineConfig.backend", backend, reason)
        dtype = ("float32" if backend == "hybrid" else "float64") if precision.dtype == "auto" else precision.dtype
        book.decide("PrecisionConfig.dtype", dtype, _why(precision.dtype, AUTO_RULES["PrecisionConfig.dtype"],
                                                         f"{backend} backend"))
        if backend == "hybrid" and self._device_type() != "cuda":
            book.refuse(f"the hybrid backend needs a CUDA device (EngineConfig.device={engine.device!r})",
                        "hybrid-needs-cuda")
        rule = AUTO_RULES["EngineConfig.cueq_conv_fusion"]
        if backend == "cueq":
            fusion = engine.cueq_conv_fusion == "on" if reproducible else engine.cueq_conv_fusion != "off"
            book.decide("EngineConfig.cueq_conv_fusion", _toggle(fusion), _why(
                engine.cueq_conv_fusion, rule, "plain cuEq: cuEquivariance's fused convolution (MACE PR 1641)"
                if fusion else "reproducible=True: the unfused convolution, whose messages MACE's scatter_add_ sums "
                "under the deterministic flag (owner decision 13)" if reproducible
                else "plain cuEq, unfused convolution"))
        else:
            if engine.cueq_conv_fusion == "on":
                book.refuse("EngineConfig.cueq_conv_fusion='on' requires backend 'cueq' (the hybrid fuses its OEq "
                            "convolution anyway)", "conv-fusion-needs-cueq")
            fusion = None
            book.decide("EngineConfig.cueq_conv_fusion", "off", _why(
                engine.cueq_conv_fusion, rule, "not applicable: the hybrid fuses its OEq convolution anyway "
                "(the research engine records None)"))
        return backend, dtype, fusion

    def _device_type(self):
        import torch

        return torch.device(self.engine.device).type

    def model_settings(self) -> ModelSettings:
        """Backend, dtype, device and cuEq fusion for ``loading.load`` (stage 1; raises PlanRefused)."""
        book = _Book()
        backend, dtype, fusion = self._backend(book)
        book.raise_refusals()
        return ModelSettings(backend, dtype, self.engine.device, fusion)

    def _expandable_segments(self, book):
        from fastmace.engine.constraints import wants_expandable_segments

        rule = AUTO_RULES["EngineConfig.expandable_segments"]
        request = self.engine.expandable_segments
        if self._device_type() != "cuda":
            book.decide("EngineConfig.expandable_segments", "off", _why(request, rule, "CPU device: no CUDA allocator"))
            return "off"
        wanted = wants_expandable_segments(self.engine, self.environ, self.torch_version)
        current = self.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        explanation = ("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before the first CUDA allocation" if wanted
                       else f"allocator left alone (PYTORCH_CUDA_ALLOC_CONF={current!r}, torch "
                            f"{self.torch_version or 'installed'})")
        book.decide("EngineConfig.expandable_segments", _toggle(wanted), _why(request, rule, explanation))
        return _toggle(wanted)

    def configure_process(self, engine: EngineConfig | None = None) -> str | None:
        """Set the expandable-segments allocator when it resolves on; returns PYTORCH_CUDA_ALLOC_CONF.

        Must run before the first CUDA allocation of the process (``hardware``, ``load`` and ``build``
        call it); the engine's process constraints refuse an allocator that was set too late.
        """
        from fastmace.model import compile as compiling

        if engine is None:
            engine = dataclasses.replace(self.engine, expandable_segments=self._expandable_segments(_Book()))
        if engine.expandable_segments == "on":
            return compiling.configure_allocator(environ=self.environ, torch_version=self.torch_version)
        return self.environ.get("PYTORCH_CUDA_ALLOC_CONF")

    def hardware(self) -> Hardware:
        """``detect_hardware`` of the engine's device, after ``configure_process``."""
        self.configure_process()
        return detect_hardware(self.engine.device)

    def load(self, source, head=None, **options):
        """``fastmace.model.loading.load`` with the stage-1 backend, dtype, device and fusion.

        Calls ``configure_process`` first (loading is the first CUDA allocation of a run). ``options``
        go to ``load`` (``checkpoint``, ``trusted``, ``charge``, ...).
        """
        from fastmace.model import loading

        settings = self.model_settings()
        self.configure_process()
        return loading.load(source, head, backend=settings.backend, dtype=settings.dtype, device=settings.device,
                            cueq_conv_fusion=settings.cueq_conv_fusion, **options)

    # ------------------------------------------------------------------ the plan

    def plan(self, traits, atoms, integrator: IntegratorConfig, hardware: Hardware) -> Plan:
        """The fixed plan for model ``traits`` (a ``ModelTraits``, a model or a ``LoadedModel``), ``atoms``,
        ``integrator`` and ``hardware``; raises ``PlanRefused`` with every reason."""
        book = _Book()
        parts = self._resolve(book, traits, atoms, integrator, hardware)
        book.raise_refusals()
        return Plan(**parts, reason="fixed plan of phase 1a (main.py)")

    def _resolve(self, book, traits, atoms, integrator, hardware, *, backend_choice=None, evaluation=False,
                 texts=None, reproducible=False):
        """The plan's parts (``Plan`` keywords except the reason), decisions and refusals in ``book``.

        Shared with the minimal planner (1b-T12): ``backend_choice`` is its backend rule;
        ``evaluation`` plans a one-off compiled evaluation with stress (no integrator, no graph
        tier, so exact padding is not required); ``texts`` maps a refusal code to a function that
        rewrites its text; ``reproducible`` (minimal planner, rule 5) takes the unfused convolution.
        """
        from fastmace.longrange import polar as longrange
        from fastmace.model.capabilities import ModelTraits, model_traits

        texts = texts or {}

        def refuse(text, code, rule=None):
            book.refuse(texts[code](text) if code in texts else text, code, rule)

        if not isinstance(traits, ModelTraits):
            traits = model_traits(traits.model if hasattr(traits, "calculator") else traits)
        requested = self._requested()
        explicit = {item.split(".", 1)[1] for item, value in requested if item.startswith("RewriteConfig.")
                    and value == "on"}
        npt = evaluation or integrator.ensemble == "npt"

        # Workload (SPEC §4.4, rule 1) and the graph tier's exact padding (rule 4).
        if evaluation:
            book.decide("tier", "compiled", "one-off evaluation: compiled model, exact edge counts, no CUDA graphs")
        else:
            book.decide("tier", TIER, "the research engine's --engine gpu: CUDA graphs, edge buckets (SPEC §8.1)")
        pbc = tuple(bool(flag) for flag in atoms.pbc)
        if not all(pbc):
            refuse(f"periodic boundaries in all three directions (3D PBC) are required, got pbc={pbc} (SPEC §4.4)",
                   "non-3d-pbc", 1)
        if atoms.constraints:
            names = ", ".join(sorted({type(c).__name__ for c in atoms.constraints}))
            refuse(f"constraints are refused in 1a ({names}; SPEC §4.4)", "constraints", 1)
        if not traits.padding_is_exact and not evaluation:
            refuse("edge padding is not exact for this model (edges beyond the cutoff get nonzero weights: "
                   "apply_cutoff=True with biased or species-dependent radial networks), so the graph tier "
                   "is refused (SPEC §9.3 rule 4)", "inexact-padding", 4)
        book.decide("workload", "ok", f"{len(atoms)} atoms, 3D PBC, no constraints" if all(pbc) and not
                    atoms.constraints else "refused")
        book.decide("hardware", hardware.name, f"compute capability {hardware.compute_capability[0]}."
                    f"{hardware.compute_capability[1]}, {hardware.memory_bytes / 2**30:.1f} GiB (recorded only)")

        backend, dtype, fusion = self._backend(book, backend_choice, reproducible)
        hybrid = backend == "hybrid"
        values = self._stage1(book, hybrid, npt)
        self._stage2(book, values, traits, explicit)
        self._fastmace_rules(book, values, hybrid, explicit)
        rewrite_values = {name: _toggle(values[name]) for name in REWRITE_FIELDS}

        rewrites = self.rewrites
        if rewrites.inplace_block_linear:
            book.refuse(f"RewriteConfig.inplace_block_linear=True: {NOT_CARRIED_OVER_1A}",
                        "inplace-block-linear-not-carried-over")
        book.decide("RewriteConfig.inplace_block_linear", _toggle(rewrites.inplace_block_linear), NOT_CARRIED_OVER_1A)
        book.decide("RewriteConfig.polar_uvu", _toggle(rewrites.polar_uvu),
                    "PolarMACE only (MACE PR 1711)" if traits.polar else "no effect: not a PolarMACE model")

        # PolarMACE (SPEC §9.3 rule 3; workflow.GPUEngine.__init__) and its k-space headroom.
        polar = None
        if traits.polar:
            supported, reason = longrange.route_support(backend, self.engine.compile_mode)
            if not supported:
                book.refuse(reason, "polar-plain-compiled-cueq", 3)
            headroom = self.polar.kspace_headroom
            resolved = longrange.resolve_kspace_headroom(self.polar, npt)
            book.decide("PolarConfig.kspace_headroom", resolved, _why(
                "auto" if headroom == "auto" else str(headroom), AUTO_RULES["PolarConfig.kspace_headroom"],
                "variable cell (NPT)" if npt else "fixed cell"))
            polar = dataclasses.replace(self.polar, kspace_headroom=resolved)
        else:
            book.decide("PolarConfig.kspace_headroom", "n/a", _why(
                "auto" if self.polar.kspace_headroom == "auto" else str(self.polar.kspace_headroom),
                AUTO_RULES["PolarConfig.kspace_headroom"], "not a PolarMACE model, no long range"))

        # Engine options (stage 1 of parse_args) and the process constraints.
        engine = self.engine
        expandable = self._expandable_segments(book)
        book.decide("EngineConfig.compile_mode", engine.compile_mode, "main.py's --compile-mode")
        if evaluation:
            pipelined = async_output = False
            for name in ("speculative_graph", "pipelined", "async_output"):
                book.decide(f"EngineConfig.{name}", "off", "one-off evaluation: no graph engine")
            book.decide("capture", "off", "one-off evaluation: no CUDA graphs")
            book.decide("integrator", "none", "one-off evaluation")
            book.decide("compute_stress", "on", "one-off evaluation: stress for any cell (fm.calc, cell filters)")
            engine = dataclasses.replace(engine, speculative_graph=False)
        else:
            if npt and engine.speculative_graph:
                book.refuse("EngineConfig.speculative_graph is not available with NPT (variable cell)",
                            "speculative-graph-npt")
            book.decide("EngineConfig.speculative_graph", _toggle(engine.speculative_graph),
                        "requested" if engine.speculative_graph else "off unless requested (main.py default)")
            if engine.pipelined == "on" and engine.speculative_graph:
                book.refuse("EngineConfig.pipelined='on' excludes speculative_graph", "pipelined-excludes-speculative")
            pipelined = not engine.speculative_graph and engine.pipelined != "off"
            book.decide("EngineConfig.pipelined", _toggle(pipelined), _why(
                engine.pipelined, AUTO_RULES["EngineConfig.pipelined"],
                "off: speculative_graph" if engine.speculative_graph else "graph engine"))
            async_output = engine.async_output != "off"
            book.decide("EngineConfig.async_output", _toggle(async_output), _why(
                engine.async_output, AUTO_RULES["EngineConfig.async_output"], "graph engine; the run loop's choice"))
            book.decide("capture", "on" if self._device_type() == "cuda" else "off",
                        "CUDA graphs" if self._device_type() == "cuda" else "CPU device: eager stages")

            # Integrator (workflow.GPUEngine.__init__, baoab_settings).
            if npt:
                book.decide("integrator", "LangevinBAOAB", f"NPT, {integrator.cell_mode} cell, externalstress="
                            f"{_stress_text(integrator)}, disable_cell_langevin="
                            f"{bool(integrator.disable_cell_langevin)} (workflow.baoab_settings)")
            else:
                book.decide("integrator", "Langevin", f"NVT, fixcm={integrator.fixcm} (workflow.GPUEngine)")
            book.decide("compute_stress", _toggle(npt), "variable cell (NPT)" if npt else
                        "fixed cell: the graph engine computes no stress")

        return dict(
            tier="compiled" if evaluation else TIER,
            rewrites=dataclasses.replace(rewrites, **rewrite_values),
            engine=dataclasses.replace(engine, backend=backend, pipelined=_toggle(pipelined),
                                       async_output=_toggle(async_output), cueq_conv_fusion=_toggle(bool(fusion)),
                                       expandable_segments=expandable),
            precision=dataclasses.replace(self.precision, dtype=dtype),
            neighbours=dataclasses.replace(self.neighbours, sort_receivers=_toggle(values["sort_receivers"]),
                                           gpu_neighbour_list=_toggle(values["gpu_neighbour_list"])),
            polar=polar,
            integrator=None if evaluation else integrator,
            decisions=tuple(book.decisions),
            requested=requested,
            auto_options_off=tuple(values["_auto_off"].items()),
        )

    def _requested(self):
        """(item, value) of every option the caller did not leave at "auto" or its default."""
        out = []
        for config in (self.rewrites, self.engine, self.precision, self.neighbours, self.polar):
            defaults = type(config)()
            for field in dataclasses.fields(config):
                value = getattr(config, field.name)
                if value != "auto" and value != getattr(defaults, field.name):
                    out.append((f"{type(config).__name__}.{field.name}", str(value)))
                elif value == "on":
                    out.append((f"{type(config).__name__}.{field.name}", "on"))
        return tuple(out)

    def _stage1(self, book, hybrid, npt):
        """``workflow.parse_args`` for the graph engine: {option: bool}, decisions after stage 2."""
        r, n = self.rewrites, self.neighbours
        v = {"_auto_off": {}, "_why": {}}

        def note(name, value, explanation):
            v[name] = bool(value)
            v["_why"][name] = explanation

        def needs(name, value, condition, requirement):
            """An option that needs ``condition``: "on" without it is refused, "auto" follows it."""
            request = getattr(r, name)
            if request == "on" and not condition:
                book.refuse(f"RewriteConfig.{name}='on' requires {requirement}", "rewrite-requirement-missing")
            note(name, condition and value, f"stage 1: {'on' if condition and value else 'off'}"
                 + ("" if request == "off" else f" ({'with' if condition else 'without'} {requirement})"))

        backend_name = "hybrid" if hybrid else "plain cuEq"
        note("factor_species", hybrid if r.factor_species == "auto" else r.factor_species == "on",
             f"stage 1: {backend_name} backend")
        needs("fold_cutoff", r.fold_cutoff != "off", v["factor_species"], "factor_species")
        request = n.sort_receivers
        note("sort_receivers", hybrid if request == "auto" else request == "on", f"{backend_name} backend")
        for name in ("density_first", "reassociate_first", "joint_radial_input", "fused_folded_backward"):
            needs(name, getattr(r, name) != "off", v["fold_cutoff"], "fold_cutoff")
        needs("segmented_first_sum", r.segmented_first_sum != "off", v["reassociate_first"] and v["sort_receivers"],
              "reassociate_first and sort_receivers")
        note("gpu_neighbour_list", n.gpu_neighbour_list != "off", "graph engine")
        needs("fused_radial_hidden", r.fused_radial_hidden == "on", v["fold_cutoff"], "fold_cutoff")
        if r.fused_radial_hidden == "auto":
            v["_why"]["fused_radial_hidden"] = "off: only an explicit 'on' enables it (no measured gain)"
        needs("streaming_conv", r.streaming_conv != "off", hybrid and v["fold_cutoff"] and v["sort_receivers"],
              "the hybrid backend, fold_cutoff (stage 1) and sort_receivers (a hard requirement: streaming "
              "kernels on unsorted edges give wrong results)")
        note("block_linear", hybrid if r.block_linear == "auto" else r.block_linear == "on",
             f"stage 1: {backend_name} backend")
        needs("grouped_linear", r.grouped_linear != "off", v["block_linear"], "block_linear")
        needs("merged_residual", r.merged_residual != "off", v["block_linear"], "block_linear")
        note("species_skip", r.species_skip != "off", "stage 1: graph engine, any backend")
        note("single_graph_strain", r.single_graph_strain != "off",
             "acts where stress is computed (NPT)" if npt else "process-wide; acts only where stress is computed")
        return v

    def _stage2(self, book, v, traits, explicit):
        """``workflow.adapt_options_to_model``: unsupported options off (auto) or refused (on)."""
        support = traits.rewrite_support
        for name, key in STAGE2_SUPPORT:
            supported, reason = support[key]
            if v[name] and not supported:
                if name in explicit:
                    book.refuse(f"RewriteConfig.{name}='on': not supported by the model ({reason})",
                                "rewrite-unsupported-by-model")
                    continue
                v[name] = False
                v["_auto_off"][name] = reason
                v["_why"][name] += f"; stage 2: off, not supported by the model ({reason})"
            elif v[name]:
                v["_why"][name] += f"; stage 2: supported ({reason})"

    def _fastmace_rules(self, book, v, hybrid, explicit):
        """FastMACE's hard rules on the rewrites, then one decision per rewrite field and neighbour option."""
        if not hybrid and v["reassociate_first"]:
            if "reassociate_first" in explicit:
                book.refuse(f"RewriteConfig.reassociate_first='on' on plain cuEq: {PLAIN_CUEQ_REASSOCIATION}",
                            "reassociation-plain-cueq")
            v["reassociate_first"] = False
            v["_why"]["reassociate_first"] += f"; off on plain cuEq: {PLAIN_CUEQ_REASSOCIATION}"
        if v["segmented_first_sum"] and not v["reassociate_first"]:
            if "segmented_first_sum" in explicit:
                book.refuse("RewriteConfig.segmented_first_sum='on' requires reassociate_first, which is off",
                            "rewrite-requirement-missing")
            v["segmented_first_sum"] = False
            v["_why"]["segmented_first_sum"] += "; off: requires reassociate_first, which is off"
        if v["streaming_conv"] and not v["sort_receivers"]:  # cannot happen after stage 1; kept as the hard rule
            book.refuse("RewriteConfig.streaming_conv needs sort_receivers (NeighbourConfig.sort_receivers='on')",
                        "streaming-needs-sorted-receivers")
        for name in REWRITE_FIELDS:
            item = f"RewriteConfig.{name}"
            book.decide(item, _toggle(v[name]), _why(getattr(self.rewrites, name), AUTO_RULES[item], v["_why"][name]))
        for name in ("sort_receivers", "gpu_neighbour_list"):
            item = f"NeighbourConfig.{name}"
            book.decide(item, _toggle(v[name]), _why(getattr(self.neighbours, name), AUTO_RULES[item], v["_why"][name]))

    # ------------------------------------------------------------------ build

    def rewrite_resolution(self, plan: Plan, model, registry=None):
        """The rewrite records for ``model`` under ``plan``: a ``Resolution`` of the registry.

        Each record is asked for "on" when its field resolved on (polar_uvu: its default) and "off"
        otherwise. A record that cannot be enabled is off with its reason, as the research engine's
        switches are no-ops on models they do not fit (``merged_residual`` without ``density_first``,
        MH-1's ``segmented_first_sum`` beside ``generic_reassociate``, which has its own
        ``generic_segmented_first_sum``; MACE-OFF's non-generic first block).
        Only a field the caller set to "on" explicitly with no record of it enabled is refused.
        """
        from fastmace.rewrites import registry as reg

        registry = registry or self.registry or reg.default_registry()
        resolution = registry.resolve(model, registry.requested(plan.rewrites), strict=False)
        explicit = {item.split(".", 1)[1] for item, value in plan.requested
                    if item.startswith("RewriteConfig.") and value == "on"}
        enabled_fields = {registry.config_field(d.name) for d in resolution.decisions if d.enabled}
        decisions, refusals = [], []
        for d in resolution.decisions:
            if d.refused:
                field = registry.config_field(d.name)
                if field in explicit and field not in enabled_fields:
                    refusals.append(f"{d.name}: {d.reason}")
                d = dataclasses.replace(d, refused=False,
                                        reason=f"{d.reason} (RewriteConfig.{field} is on; this record stays off)")
            decisions.append(d)
        if refusals:
            raise reg.RewriteRefusedError("requested rewrites cannot be enabled: " + "; ".join(refusals))
        resolution = reg.Resolution(tuple(decisions))
        problems = workload_problems(resolution.enabled, plan, registry)
        if problems:
            raise PlanRefused.from_reasons(problems)
        return resolution

    def build(self, plan: Plan, model, atoms, *, temperature_K: float = 0.0, compile_cache=None, constraints=None,
              check: bool = True, capture: bool | None = None):
        """The ready ``GraphEngine`` of ``plan`` for ``model`` (a ``LoadedModel`` from ``load``) and ``atoms``.

        Builds the force field (rewrites resolved for the model and self-checked), the state, the
        integrator at ``temperature_K`` and the engine, which does the initial force evaluation.
        ``compile_cache`` (``fastmace.model.compile.CompiledModelCache``) is shared by the engines of
        one run, as the research engine shares its compiled models between crystals; ``constraints``
        is the process record (default the process's). ``capture`` is the engine's (default: captured
        graphs on CUDA); the eager tier (2-T02) passes False and builds a plan of tier ``"eager"``.
        """
        from fastmace.engine.graph import GraphEngine
        from fastmace.integrators.baoab import LangevinBAOAB
        from fastmace.integrators.langevin import Langevin
        from fastmace.model.forces import MACEForceField
        from fastmace.state import SimulationState

        if plan.tier not in (TIER, EAGER_TIER):
            raise ValueError(f"FixedPlanner builds tier {TIER!r} and {EAGER_TIER!r} plans, not {plan.tier!r}")
        if plan.tier == EAGER_TIER and capture is not False:
            raise ValueError(f"a tier {EAGER_TIER!r} plan runs without CUDA graphs: pass capture=False")
        for part in (plan.rewrites, plan.engine, plan.precision, plan.neighbours, plan.polar):
            if part is not None and "auto" in dataclasses.astuple(part):
                raise ValueError(f"{type(part).__name__} still has 'auto' values; build takes a plan from plan()")
        backend = getattr(model, "backend", None)
        dtype = getattr(model, "dtype", None)
        if backend is not None and (backend, dtype) != (plan.engine.backend, plan.precision.dtype):
            raise ValueError(f"the model was loaded as {backend} {dtype}, the plan needs {plan.engine.backend} "
                             f"{plan.precision.dtype}; load it with FixedPlanner.load")
        book = _Book()
        if not all(atoms.pbc):
            book.refuse("periodic boundaries in all three directions (3D PBC) are required (SPEC §4.4)",
                        "non-3d-pbc", 1)
        if atoms.constraints:
            book.refuse("constraints are refused in 1a (SPEC §4.4)", "constraints", 1)
        book.raise_refusals()
        self.configure_process(plan.engine)

        from fastmace.rewrites import registry as reg

        registry = self.registry or reg.default_registry()
        source = model.model if hasattr(model, "calculator") else model.models[0]
        resolution = self.rewrite_resolution(plan, source, registry)
        npt = plan.integrator.ensemble == "npt"
        field = MACEForceField(model, atoms, rewrites=resolution, engine=plan.engine, neighbours=plan.neighbours,
                               polar=plan.polar or PolarConfig(), compute_stress=npt, compile_cache=compile_cache,
                               registry=registry, check=check, float64_energy=plan.precision.float64_energy)
        state = SimulationState.from_atoms(atoms, field.device, plan.integrator.seed, compute_stress=npt)
        if npt:
            integrator = LangevinBAOAB(state, plan.integrator, temperature_K=temperature_K,
                                       **integrator_keywords(plan.integrator))
        else:
            integrator = Langevin(state, plan.integrator, temperature_K=temperature_K)
        return GraphEngine(plan.engine, field, integrator, state, neighbours=plan.neighbours, constraints=constraints,
                           capture=capture)


def auto_options_off(plan: Plan) -> dict[str, str]:
    """{option: reason} that auto switched off for the model (metadata.json's ``auto_options_off``)."""
    return dict(plan.auto_options_off)


def enabled_options(plan: Plan) -> tuple[str, ...]:
    """The rewrite fields and neighbour options the plan turned on, in the research engine's order."""
    out = [name for name in REWRITE_FIELDS if getattr(plan.rewrites, name) == "on"]
    out += [name for name in ("sort_receivers", "gpu_neighbour_list") if getattr(plan.neighbours, name) == "on"]
    return tuple(out)


__all__ = (
    "AUTO_RULES", "EAGER_TIER", "FixedPlanner", "ModelSettings", "PlanRefused", "STAGE2_SUPPORT", "TIER",
    "WORKLOAD_CONSTRAINTS", "auto_options_off", "detect_hardware", "enabled_options", "integrator_keywords",
    "workload_problems",
)
