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
    DEFAULT_SKIN_A,
    EngineConfig,
    IntegratorConfig,
    NeighbourConfig,
    PolarConfig,
    PrecisionConfig,
    RewriteConfig,
)
from fastmace.errors import PlanRefused, Reason
from fastmace.rewrites import versions

from . import rules
from .plan import Hardware, Plan

TIER = "graph"
# The plan record's integrator name per scheme other than Langevin/BAOAB (6-T11): the fastmace.integrators class.
SCHEME_CLASS_NAMES = MappingProxyType({"nve": "NVE", "csvr": "CSVR", "nhc": "NoseHooverChain", "mtk": "MTK",
                                       "berendsen": "Berendsen", "langevin": "Langevin", "baoab": "LangevinBAOAB"})
EAGER_TIER = "eager"  # the graph engine without capture or compilation (SPEC §8.3, 2-T02)
COMPILED_TIER = "compiled"  # the compiled model without capture, exact edge counts (SPEC §8.2, 3-T06)

# The switch map's rule for every field that admits "auto" (tests/cpu/test_config.py AUTO_RULES).
AUTO_RULES = MappingProxyType({
    "EngineConfig.backend": "backend_hybrid_for_graph_engine",
    "EngineConfig.pipelined": "on_with_graph_engine_unless_speculative",
    "EngineConfig.async_output": "on_with_graph_engine",
    "EngineConfig.cueq_conv_fusion": "on_for_plain_cueq_unless_reproducible",
    "EngineConfig.expandable_segments": "on_for_torch_2_14_unless_alloc_conf_set",
    "EngineConfig.fp64_emulation": "on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible",
    "EngineConfig.fp64_emulation_mantissa_bits": "dynamic_decision_65_d1_suspended_after_recapture_defect",
    "PrecisionConfig.dtype": "float32_for_hybrid_float64_for_cueq",
    "NeighbourConfig.sort_receivers": "on_for_hybrid_graph_engine",
    "NeighbourConfig.gpu_neighbour_list": "on_with_graph_engine",
    "NeighbourConfig.skin": "rule_11_skin_from_workload",
    "NeighbourConfig.headroom": "rule_11_headroom_from_workload",
    "NeighbourConfig.d3_skin": "rule_11_d3_skin_from_workload",
    "IntegratorConfig.fixcm_projection": "mass_weighted_under_nvt_fixcm_else_ase",
    "IntegratorConfig.dof": "owner_decision_d2_default",
    "EngineConfig.recapture_budget": "recapture_budget_from_workload",
    "EngineConfig.force_graphs": "fit_when_capturing_on_cuda_else_keep",
    "PrecisionConfig.mc_energy": "force_precision_if_move_energy_gate_passes_else_float64",
    "RewriteConfig.factor_species": "on_for_hybrid_graph_engine_if_model_supports",
    "RewriteConfig.fold_cutoff": "on_with_factor_species_if_model_supports",
    "RewriteConfig.density_first": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.reassociate_first": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.joint_radial_input": "on_with_fold_cutoff_if_model_supports",
    "RewriteConfig.fused_folded_backward": "on_with_stage1_fold_cutoff_if_model_supports",
    "RewriteConfig.segmented_first_sum": "on_with_reassociate_first_and_sort_receivers",
    "RewriteConfig.streaming_conv": "on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports",
    "RewriteConfig.chunked_conv_backward": "on_with_streaming_conv_and_fused_folded_backward",
    "RewriteConfig.fused_folded_forward": "on_for_mh1_float32_unless_reproducible",
    "RewriteConfig.radial_tables": "on_with_fold_cutoff_unless_reproducible",
    "RewriteConfig.w_tables": "on_for_mh1_float64_on_measured_gpus_decision_65",
    "RewriteConfig.wprime_backward": "on_for_mh1_float32_unless_reproducible_on_decision_66_gpus",
    "RewriteConfig.recompute_conv_weights": "on_by_rule_12_when_the_plan_does_not_fit_without_it",
    "RewriteConfig.species_skip": "on_with_graph_engine_if_model_supports",
    "RewriteConfig.fused_radial_hidden": "never_automatic",
    "RewriteConfig.direct_node_feats": "never_automatic",
    "RewriteConfig.folded_constants": "never_automatic",
    "RewriteConfig.folded_backward_tiles": "on_with_fused_folded_backward",
    "RewriteConfig.block_linear": "on_for_hybrid_graph_engine",
    "RewriteConfig.grouped_linear": "on_with_block_linear",
    "RewriteConfig.merged_residual": "on_with_block_linear",
    "RewriteConfig.gated_rows": "on_with_merged_residual",
    "RewriteConfig.segment_gemm": "on_with_gated_rows_float32_unless_reproducible",
    "RewriteConfig.gate_epilogue": "never_automatic",
    "RewriteConfig.grouped_launch":
        "on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus",
    "RewriteConfig.combo_kernels": "never_automatic",
    "RewriteConfig.save_transcendentals": "on_for_mh1_float64_on_measured_gpus_decision_65",
    "RewriteConfig.tf32x3_gemms": "never_automatic",
    "RewriteConfig.fx_e3nn_linear": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    "RewriteConfig.polar_graph_sums": "on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints",
    "RewriteConfig.species_tables": "on_unless_reproducible_if_model_supports",
    "RewriteConfig.fold_readout_tail": "on_with_block_linear_unless_reproducible_if_model_supports",
    "RewriteConfig.fold_product_linear": "on_with_species_tables_and_block_linear_if_model_supports",
    "RewriteConfig.species_residual": "never_automatic",
    "RewriteConfig.single_graph_strain": "on",
    "PolarConfig.kspace_headroom": "variable_cell_0_1_else_0",
})

# adapt_options_to_model's ``requires``: option -> rewrite_support key, in its order.
STAGE2_SUPPORT = (
    ("density_first", "fold_cutoff"), ("joint_radial_input", "fold_cutoff"), ("fused_radial_hidden", "fold_cutoff"),
    ("gated_rows", "fold_cutoff"), ("segment_gemm", "fold_cutoff"), ("gate_epilogue", "fold_cutoff"),
    ("folded_backward_tiles", "fold_cutoff"), ("direct_node_feats", "fold_cutoff"), ("folded_constants", "fold_cutoff"),
    ("chunked_conv_backward", "fold_cutoff"), ("fused_folded_forward", "fold_cutoff"),
    ("radial_tables", "fold_cutoff"), ("w_tables", "fold_cutoff"), ("wprime_backward", "fold_cutoff"),
    ("species_tables", "species_tables"),
    ("fold_product_linear", "species_tables"),
    ("species_residual", "species_tables"),
    ("fold_readout_tail", "fold_readout_tail"),
    ("fused_folded_backward", "fused_folded_backward"), ("species_skip", "species_skip"),
    ("streaming_conv", "streaming_conv"), ("fold_cutoff", "fold_cutoff"), ("factor_species", "factor_species"),
    ("reassociate_first", "reassociate_first"), ("segmented_first_sum", "reassociate_first"),
)
# The rewrite fields in the order the research engine resolves them (parse_args).
REWRITE_FIELDS = ("factor_species", "fold_cutoff", "density_first", "reassociate_first", "joint_radial_input",
                  "fused_folded_backward", "folded_backward_tiles", "segmented_first_sum", "fused_radial_hidden",
                  "direct_node_feats", "folded_constants", "streaming_conv", "chunked_conv_backward",
                  "fused_folded_forward", "radial_tables", "w_tables", "wprime_backward", "recompute_conv_weights",
                  "block_linear", "grouped_linear", "merged_residual", "gated_rows", "segment_gemm", "gate_epilogue",
                  "grouped_launch", "species_skip",
                  "single_graph_strain", "species_tables", "fold_readout_tail", "fold_product_linear",
                  "species_residual",
                  "combo_kernels", "save_transcendentals", "tf32x3_gemms",
                  "fx_e3nn_linear", "polar_graph_sums")

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
    return Hardware(properties.name, (properties.major, properties.minor), int(properties.total_memory),
                    int(properties.multi_processor_count))


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


def refuse_unsupported(book: _Book, config: IntegratorConfig, atoms, optimizer=None) -> None:
    """Rule 1 again at build (4-T01): the plan's integrator scheme (or its ``optimizer``'s, 4-T05) on ``atoms``'
    constraints, from the scheme records. The plan refused MC with constraints already."""
    from fastmace.integrators import registry as schemes

    fixed_atoms, other = schemes.constraint_names(atoms)
    if optimizer is not None:
        reasons = schemes.optimizer_refusals(optimizer, fix_atoms=bool(fixed_atoms), other_constraints=other)
    else:
        reasons = schemes.integrator_refusals(config, fix_atoms=bool(fixed_atoms), other_constraints=other)
    for reason in reasons:
        book.refuse(reason, reason.code, 1)


def build_stepper(state, plan: Plan, *, temperature_K: float | None = None):
    """What steps ``state`` in ``plan``'s engine: the integrator from the registry, or for a relaxation the
    optimiser over its DOF view in the integrator's place (``optimize.driver.OptimizerStepper``, 4-T05)."""
    from fastmace.integrators import registry as schemes

    if plan.optimizer is None:
        return schemes.build_integrator(state, plan.integrator, temperature_K=temperature_K)
    from fastmace.optimize.driver import OptimizerStepper

    dof = schemes.build_dof(state, plan.optimizer)
    return OptimizerStepper(schemes.build_optimizer(dof, plan.optimizer))


def _workload_text(atoms, pbc):
    """The workload decision's reason: atom count, PBC and constraints."""
    if not all(pbc):
        return "refused"
    fixed_atoms = sum(len(c.get_indices()) for c in atoms.constraints if type(c).__name__ == "FixAtoms")
    if len(atoms.constraints) and not fixed_atoms:
        return "refused"
    return f"{len(atoms)} atoms, 3D PBC, " + (f"FixAtoms on {fixed_atoms} atoms" if fixed_atoms else "no constraints")


def _toggle(value):
    return "on" if value else "off"


# Decision 59's exception (5-T03 rerun, 24 October 2026): PolarMACE's variable-cell MD fails its CUDA graph
# capture with a dynamic edge dimension ("AccumulateGrad has a stale reference to the default stream"), on
# POLAR-1-S, -M and -L, NaCl and NbCl5; fixed-cell PolarMACE and every other family capture. Off until the cause
# is found.
DYNAMIC_EDGES_POLAR_NPT = Reason(
    "off: PolarMACE's variable-cell MD fails its CUDA graph capture with a dynamic edge dimension (stale "
    "default-stream reference in AccumulateGrad; 5-T03 rerun, 24 October 2026), so the first capture compiles "
    "again as before decision 59", code="dynamic-edges-polar-npt")

# Owner decision 58, 24 October 2026 (OPT4-T02): the POLAR-1 checkpoints whose force gate passed with
# fx_e3nn_linear and polar_graph_sums on (S and M on NaCl and NbCl5 2x2x2; L, the owner's condition, after its
# local NbCl5 2x2x2 gate). They are label-table labels, which identify a checkpoint by its SHA-256
# (fastmace.model.labels), not by its file name.
POLAR_GRAPH_LABELS = ("polar1-s", "polar1-m", "polar1-l")
POLAR_GRAPH_RECORDS = ("fx_e3nn_linear", "polar_graph_sums")


def _qualified_polar(traits, model_label):
    """PolarMACE with a checkpoint of ``POLAR_GRAPH_LABELS`` (decision 58)."""
    return bool(traits.polar) and model_label in POLAR_GRAPH_LABELS


def _mh1(model_label):
    """A label-table label of the MH-1 checkpoint (decision 64: ``fused_folded_forward``'s default)."""
    from fastmace.model import labels

    entry = labels.LABELS.get(model_label) if model_label is not None else None
    return entry is not None and entry.checkpoint.family == "mh1"


def _label_of(traits):
    """The label-table label of a ``LoadedModel`` (by label, or by SHA-256 and head); None for anything else."""
    if not hasattr(traits, "sha256"):
        return None
    from . import warnings as known_issues

    return known_issues.label_of(getattr(traits, "label", None), getattr(traits, "head", None), traits.sha256)


def _generic_mace_family(traits):
    """MP-0 / MPA-0 / OMAT-0 / MatPES-class models: every interaction MP-0's residual block or the density
    family's (GENERIC_INTERACTIONS), not PolarMACE. MACE-OFF (a plain first interaction), MH-1, POLAR-1 and
    OMOL-0 are other families (decision 57)."""
    from fastmace.model.capabilities import GENERIC_INTERACTIONS

    interactions = tuple(traits.interaction_classes)
    return bool(interactions) and not traits.polar and all(name in GENERIC_INTERACTIONS for name in interactions)


def _stress_text(config):
    """The integrator decision's externalstress, in ASE units, with where it came from."""
    keywords = integrator_keywords(config)
    stress = keywords["externalstress"]
    if config.external_stress_GPa is None:
        return f"{stress:.6g} (-{config.pressure_GPa:g} GPa)"
    if isinstance(stress, float):
        return f"{stress:.6g} ({config.external_stress_GPa:g} GPa)"
    return "(" + ", ".join(f"{value:.6g}" for value in stress) + ") (external_stress_GPa, Voigt)"


def _cuda_available():
    import torch

    return torch.cuda.is_available()


# EngineConfig.fp64_emulation's threshold (decision 63): emulate where float64 runs below 1/8 of float32.
FP64_EMULATION_RATE = 1 / 8
FP64_EMULATION_MANTISSA_NAME = "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"

# Owner decision 65 (28 October 2026): "Make all 3 default on gpu where we measured a speed up but if slowdown or
# nothing measured dont enable it". The 56 fixed mantissa bits (D1), w_tables (D2) and save_transcendentals (D3)
# are "auto" on in float64 MH-1 plans that are not reproducible, on these GPU models only (same-process A/Bs in
# both orders, plans/ideas-20261027.md); the A100 measured slower (D2, D3). Exact device names: a compute
# capability would take in GPUs never measured (the L4, other sm_89 and sm_120 cards). Adding a GPU here needs its
# own A/B of all three in both orders.
DECISION_65_GPUS = ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")
# D1's bit count; its "auto" default is suspended (_fp64_mantissa), explicit requests stay.
DECISION_65_MANTISSA_BITS = 56
# w_tables: measured on two and three species (NaNbO3: 1.3 GB of tables); four would pass W_TABLE_MAX_BYTES.
DECISION_65_MAX_SPECIES = 3
# Decision 66 (owner, 29 October 2026): per-GPU defaults where same-process whole-step A/Bs in both build orders
# measured a speed-up, by exact device name. grouped_launch for MH-1 on the RTX 5060 Ti (current main, 29 October):
# NaCl 0.971 / 0.975, NbCl5 1.002 / 0.998, NaNbO3 1.001 / 0.999. The RTX 4000 Ada lost 0.5 % in OPT-T03b.
DECISION_66_GROUPED_LAUNCH_GPUS = ("NVIDIA GeForce RTX 5060 Ti",)
# wprime_backward (OPT9-T05, owner 29 October 2026): MD step b/a in both build orders, every block pair below 1.
# A100-SXM4-80GB: NaNbO3 0.815 / 0.812, NbCl5 0.889 / 0.889. RTX 5060 Ti: NaNbO3 0.937 / 0.940, NbCl5 0.970 / 0.971,
# NaCl 0.946 / 0.935. RTX 4000 Ada: NaNbO3 0.938 / 0.939, NbCl5 0.964 / 0.964, NaCl 0.934 / 0.934
# (docs/validation/opt9-T03.md). The A100-SXM4-40GB was not measured.
DECISION_66_WPRIME_GPUS = ("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")


def _decision_65(dtype, reproducible, hardware, model_label):
    """None where decision 65's float64 defaults apply, else why they stay off."""
    if dtype != "float64":
        return f"{dtype} plan, decision 65"
    if reproducible:
        return "reproducible=True, decision 65"
    if hardware.name not in DECISION_65_GPUS:
        return f"not a measured GPU, decision 65 ({hardware.name})"
    if not _mh1(model_label):
        return f"not MH-1, decision 65 (label {model_label})"
    return None


def _check_fp64_emulation(device):
    """Warn when the plan asked for cuBLAS's float64 emulation but float64 products still run native kernels: a
    cuBLAS handle created before ``configure_fp64_emulation`` (another plan or a float64 product earlier in the
    process) keeps its mode. The run stays correct; only the speed-up is missing."""
    import warnings

    from fastmace.model import compile as compiling

    if not compiling.fp64_emulation_active(device):
        warnings.warn("EngineConfig.fp64_emulation='on' did not take effect: float64 products run native cuBLAS "
                      "kernels (a cuBLAS handle existed before the plan set CUBLAS_EMULATE_DOUBLE_PRECISION; plan "
                      "first in a fresh process to get the emulation)", RuntimeWarning, stacklevel=3)


def _why(request, rule, explanation):
    """The reason string of one resolved option."""
    if request == "auto":
        return f"auto ({rule}): {explanation}"
    return f"requested {request!r}: {explanation}"


def _fixcm_projection(book, integrator, *, npt, mc_moves):
    """Resolve ``IntegratorConfig.fixcm_projection`` and record it (3-T22b, 3-T22c; SPEC §10.4, decision 37;
    docs/design/fixcm-projection.md).

    "auto" is "mass_weighted" under NVT ``fixcm``, for MD (code ``fixcm-projection-md``) as for MC
    (``fixcm-projection-mc``): every species holds the target, and the reported temperature counts 3N - 3
    degrees of freedom; MC's ``redraw_all`` draws (3N - 3) kT too. NPT (``LangevinBAOAB`` has no ``fixcm``)
    and ``fixcm`` off resolve to "ase" (``fixcm-projection-ase``), unchanged. An explicit value is kept:
    "ase" for ASE noise-stream equivalence and A/Bs against the research engine."""
    rule = AUTO_RULES["IntegratorConfig.fixcm_projection"]
    requested = integrator.fixcm_projection
    if requested != "auto":
        choice, why = requested, Reason(_why(requested, rule, "kept as given"), code="fixcm-projection-given")
    elif not npt and integrator.fixcm:
        what = "MC moves under fixcm: redraw_all draws (3N - 3) kT, " if mc_moves else "MD under fixcm: "
        choice, why = "mass_weighted", Reason(_why("auto", rule, (
            f"{what}the mass-weighted projection holds every species at the target, temperatures over 3N - 3 "
            "degrees of freedom (decision 37)")), code="fixcm-projection-mc" if mc_moves else "fixcm-projection-md")
    else:
        context = "NPT (LangevinBAOAB has no fixcm)" if npt else "fixcm off"
        choice, why = "ase", Reason(_why("auto", rule, f"{context}: no projection to choose"),
                                    code="fixcm-projection-ase")
    book.decide("IntegratorConfig.fixcm_projection", choice, why)
    return dataclasses.replace(integrator, fixcm_projection=choice)


def _dof(book, integrator):
    """Resolve ``IntegratorConfig.dof`` (4-T01, owner decision D2): ``"auto"`` is ``registry.DOF_AUTO``. Langevin
    and BAOAB count as they did whatever the value (``dof_rule="langevin"``), so their plans only record it."""
    from fastmace.integrators.registry import resolve_dof

    rule = AUTO_RULES["IntegratorConfig.dof"]
    choice = resolve_dof(integrator.dof)
    explanation = ("kept as given" if integrator.dof != "auto" else
                   "FastMACE's count: 3N - 3 under fixcm, 3(N - n_fixed) with FixAtoms (owner decision D2)"
                   if choice == "fastmace" else "the count of the scheme's ASE class (owner decision D2)")
    book.decide("IntegratorConfig.dof", choice, Reason(_why(integrator.dof, rule, explanation), code=f"dof-{choice}"))
    return dataclasses.replace(integrator, dof=choice)


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
        mace_rule = versions.mace_conv_fusion(engine.device) if backend == "cueq" else None
        if mace_rule is not None:
            # 2-T04b: the converter does not hash, so FastMACE cannot patch it; MACE's own rule decides.
            fusion = mace_rule
            if engine.cueq_conv_fusion != "auto" and (engine.cueq_conv_fusion == "on") != mace_rule:
                book.refuse(f"EngineConfig.cueq_conv_fusion={engine.cueq_conv_fusion!r} needs FastMACE's conversion "
                            f"patch, whose source does not hash on this MACE; MACE's own rule gives "
                            f"{_toggle(mace_rule)!r} on {engine.device!r} (SPEC §6.3)", versions.CONV_FUSION_CODE)
            book.decide("EngineConfig.cueq_conv_fusion", _toggle(fusion), Reason(
                f"MACE's own conversion rule ({_toggle(fusion)} on {engine.device!r}): FastMACE's fusion patch is "
                "not qualified on this MACE (SPEC §6.3, §18 decision 2)", code=versions.CONV_FUSION_CODE))
        elif backend == "cueq":
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
        call it); the engine's process constraints refuse an allocator that was set too late. It also puts
        the Inductor and Triton caches under the cache root where the environment names none (SPEC §14,
        ``kernels.triton.cache.configure_defaults``; OPT4-T05b), before any compilation or Triton kernel.
        """
        from fastmace.kernels.triton import cache as compiler_caches
        from fastmace.model import compile as compiling

        compiler_caches.configure_defaults(self.environ)
        if engine is None:
            engine = dataclasses.replace(self.engine, expandable_segments=self._expandable_segments(_Book()))
        if engine.expandable_segments == "on":
            return compiling.configure_allocator(environ=self.environ, torch_version=self.torch_version)
        return self.environ.get("PYTORCH_CUDA_ALLOC_CONF")

    def _fp64_emulation(self, book, dtype, reproducible, hardware):
        """``EngineConfig.fp64_emulation`` (owner decision 63, 27 October 2026): cuBLAS's float64 emulation on INT8
        tensor cores where it pays and keeps float64 accuracy (f64-T01: 1.2-1.8x per MD step on the RTX 5060 Ti,
        products more accurate than native DGEMM). A general hardware rule, measured: on where the GPU's float64
        multiply-add rate is below ``FP64_EMULATION_RATE`` of its float32 rate (consumer and workstation parts,
        about 1/64; data-centre parts are about 1/2). The compute capability cannot tell (sm_120 is 12.0)."""
        rule = AUTO_RULES["EngineConfig.fp64_emulation"]
        request = self.engine.fp64_emulation
        cuda = self._device_type() == "cuda"
        if request == "on" and reproducible:
            book.refuse("EngineConfig.fp64_emulation='on' is not available with reproducible=True (the reproducible "
                        "plans keep native float64 products)", "fp64-emulation-reproducible")
        if not cuda:
            value, why = False, "CPU device: no cuBLAS"
        elif dtype != "float64":
            value, why = False, f"{dtype} plan: no float64 matrix products"
        elif reproducible:
            value, why = False, "reproducible=True keeps native float64 products"
        elif request == "off":
            value, why = False, "requested off"
        elif not _cuda_available():
            value, why = False, "CUDA is not available in this process: the float64 rate cannot be measured"
        else:
            from fastmace.kernels.triton.fp64_rate import fp64_rate_ratio

            ratio = fp64_rate_ratio(self.engine.device)
            value = request == "on" or ratio < FP64_EMULATION_RATE
            why = (f"float64 rate 1/{1 / ratio:.0f} of float32 (measured), "
                   + ("CUBLAS_EMULATE_DOUBLE_PRECISION=1, CUBLAS_EMULATION_STRATEGY=eager before the first float64 "
                      "product" if value else f"at least {FP64_EMULATION_RATE}: native products"))
        book.decide("EngineConfig.fp64_emulation", _toggle(value), _why(request, rule, why))
        return _toggle(value)

    def _fp64_mantissa(self, book, fp64_emulation, decision_65):
        """``EngineConfig.fp64_emulation_mantissa_bits`` (ideas-20261027 D1, owner decision 65): "auto" is
        "dynamic" (cuBLAS's own choice). D1's ``DECISION_65_MANTISSA_BITS`` default on ``DECISION_65_GPUS`` is
        suspended (owner, 28 October 2026): with 56 fixed bits, a CUDA graph captured during a run computed wrong
        energies and forces (NbOCl3 3x3x3 relaxation: +0.34 eV/atom after the first re-capture, the same edges),
        and the step's small products were ~8 bits accurate (fp64-D1 fix). An explicit int still applies.
        Returns an int or "dynamic"."""
        rule = AUTO_RULES["EngineConfig.fp64_emulation_mantissa_bits"]
        request = self.engine.fp64_emulation_mantissa_bits
        if fp64_emulation != "on":
            value, why = (request if request != "auto" else "dynamic"), "no effect: fp64_emulation is off"
        elif request != "auto":
            value, why = request, ("cuBLAS's dynamic mantissa bits" if request == "dynamic"
                                   else f"{request} fixed mantissa bits ({FP64_EMULATION_MANTISSA_NAME})")
        elif decision_65 is None:
            value, why = "dynamic", (f"cuBLAS's dynamic mantissa bits: decision 65 D1 ({DECISION_65_MANTISSA_BITS} "
                                     "fixed bits) suspended, graphs captured during a run computed wrong energies "
                                     "with it")
        else:
            value, why = "dynamic", f"cuBLAS's dynamic mantissa bits: {decision_65}"
        book.decide("EngineConfig.fp64_emulation_mantissa_bits", str(value), _why(request, rule, why))
        return value

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

    def plan(self, traits, atoms, integrator: IntegratorConfig, hardware: Hardware, *, model_label=None) -> Plan:
        """The fixed plan for model ``traits`` (a ``ModelTraits``, a model or a ``LoadedModel``), ``atoms``,
        ``integrator`` and ``hardware``; raises ``PlanRefused`` with every reason. ``model_label``: the checkpoint's
        label-table label (default: a ``LoadedModel``'s), which decision 58's POLAR-1 defaults read."""
        book = _Book()
        parts = self._resolve(book, traits, atoms, integrator, hardware,
                              model_label=model_label if model_label is not None else _label_of(traits))
        book.raise_refusals()
        return Plan(**parts, reason="fixed plan of phase 1a (main.py)")

    def _resolve(self, book, traits, atoms, integrator, hardware, *, backend_choice=None, evaluation=False,
                 texts=None, reproducible=False, rule_11=None, mc=None, optimizer=None, model_label=None,
                 superset=None):
        """The plan's parts (``Plan`` keywords except the reason), decisions and refusals in ``book``.

        Shared with the minimal planner (1b-T12): ``backend_choice`` is its backend rule;
        ``evaluation`` plans a one-off compiled evaluation with stress (no integrator, no graph
        tier, so exact padding is not required); ``texts`` maps a refusal code to a function that
        rewrites its text; ``reproducible`` (minimal planner, rule 5) takes the unfused convolution;
        ``rule_11`` (minimal planner, 3-T05) is ``(applies, neighbors.adaptive.Rule11)`` for the workload;
        ``mc`` (minimal planner, 3-T11) is ``(mc_moves, Workload.mc_energy_precision)``; its moves also
        choose the fixcm projection (3-T22b). ``optimizer`` (4-T05) plans a relaxation with that
        ``OptimizerConfig``: rule 1 from the optimiser's record, checked steps only, the stress path for a
        cell filter (rule 10); ``integrator`` then only carries the MD-side defaults. ``model_label`` is the
        checkpoint's label-table label or None (decision 58: the POLAR-1 defaults need a qualified checkpoint).
        ``superset`` (minimal planner) holds the ForceField's species (atomic numbers); None takes the structure's
        (decision 65's species limit for ``w_tables``).
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
        npt = evaluation or (optimizer.variable_cell if optimizer is not None else integrator.ensemble == "npt")

        # Workload (SPEC §4.4, rule 1) and the graph tier's exact padding (rule 4).
        if evaluation:
            book.decide("tier", "compiled", "one-off evaluation: compiled model, exact edge counts, no CUDA graphs")
        else:
            book.decide("tier", TIER, "the research engine's --engine gpu: CUDA graphs, edge buckets (SPEC §8.1)")
        pbc = tuple(bool(flag) for flag in atoms.pbc)
        if not all(pbc):
            refuse(f"periodic boundaries in all three directions (3D PBC) are required, got pbc={pbc} (SPEC §4.4)",
                   "non-3d-pbc", 1)
        if evaluation:
            if atoms.constraints:
                names = ", ".join(sorted({type(c).__name__ for c in atoms.constraints}))
                refuse(f"constraints are refused in 1a ({names}; SPEC §4.4)", "constraints", 1)
        else:
            # Rule 1 from the scheme records (4-T01): the scheme, its ensemble and cell mode, its constraint rows.
            from fastmace.integrators import registry as schemes

            fixed_atoms, other = schemes.constraint_names(atoms)
            if optimizer is not None:
                reasons = schemes.optimizer_refusals(optimizer, fix_atoms=bool(fixed_atoms), other_constraints=other)
            else:
                reasons = schemes.integrator_refusals(integrator, fix_atoms=bool(fixed_atoms),
                                                      other_constraints=other, mc=bool(mc and mc[0]))
            for reason in reasons:
                refuse(reason, reason.code, 1)
        if not traits.padding_is_exact and not evaluation:
            refuse("edge padding is not exact for this model (edges beyond the cutoff get nonzero weights: "
                   "apply_cutoff=True with biased or species-dependent radial networks), so the graph tier "
                   "is refused (SPEC §9.3 rule 4)", "inexact-padding", 4)
        text = _workload_text(atoms, pbc)
        constraint_refused = any(getattr(r, "code", None) in ("constraints", "fix-atoms-with-fixcm",
                                                               "mc-with-constraints") for r in book.refusals)
        book.decide("workload", "refused" if text == "refused" or constraint_refused else "ok",
                    "refused" if constraint_refused else text)
        book.decide("hardware", hardware.name, f"compute capability {hardware.compute_capability[0]}."
                    f"{hardware.compute_capability[1]}, {hardware.memory_bytes / 2**30:.1f} GiB (recorded only)")

        backend, dtype, fusion = self._backend(book, backend_choice, reproducible)
        hybrid = backend == "hybrid"
        values = self._stage1(book, hybrid, npt)
        self._stage2(book, values, traits, explicit, model_label, hardware)
        decision_65 = _decision_65(dtype, reproducible, hardware, model_label)
        species = len(set(superset)) if superset else len(set(int(z) for z in atoms.get_atomic_numbers()))
        self._fastmace_rules(book, values, hybrid, explicit, dtype, reproducible, hardware.compute_capability,
                             decision_65=decision_65, species=species)
        fp64_emulation = self._fp64_emulation(book, dtype, reproducible, hardware)
        mantissa_bits = self._fp64_mantissa(book, fp64_emulation, decision_65)
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
        compiles = engine.compile_mode != "none"
        support = versions.compile_support() if compiles else None
        if support is not None and not support.supported:  # 2-T04b: a MACE without the compile API
            engine = dataclasses.replace(engine, compile_mode="none")
            book.decide("EngineConfig.compile_mode", "none", Reason(support.reason, code=versions.COMPILE_CODE))
        else:
            book.decide("EngineConfig.compile_mode", engine.compile_mode, "main.py's --compile-mode")
        if engine.compile_mode == "none" and engine.dynamic_edges:  # nothing to mark (decision 59, OPT4-T05)
            engine = dataclasses.replace(engine, dynamic_edges=False)
        if engine.dynamic_edges and traits.polar and npt and not evaluation:  # decision 59's exception (5-T03c)
            engine = dataclasses.replace(engine, dynamic_edges=False)
        if evaluation:
            pipelined = async_output = False
            for name in ("speculative_graph", "pipelined", "async_output"):
                book.decide(f"EngineConfig.{name}", "off", "one-off evaluation: no graph engine")
            book.decide("capture", "off", "one-off evaluation: no CUDA graphs")
            book.decide("integrator", "none", "one-off evaluation")
            book.decide("compute_stress", "on", "one-off evaluation: stress for any cell (fm.calc, cell filters)")
            engine = dataclasses.replace(engine, speculative_graph=False)
        elif optimizer is not None:
            # A relaxation (4-T05, conductor's choice 5): its converged flag decides each step before the force
            # call, so every step is a checked one, read with the neighbour flags in one host transfer.
            for name, value in (("speculative_graph", engine.speculative_graph), ("pipelined", engine.pipelined)):
                if value in (True, "on"):
                    book.refuse(f"EngineConfig.{name} is not available for a relaxation: an optimiser runs checked "
                                "steps, its converged flag read before each force call (SPEC §10.3)",
                                f"{name.replace('_', '-')}-optimizer")
                book.decide(f"EngineConfig.{name}", "off", Reason(
                    "relaxation: checked steps, the converged flag read with the neighbour flags",
                    code="optimizer-checked"))
            pipelined, engine = False, dataclasses.replace(engine, speculative_graph=False)
            async_output = engine.async_output != "off"
            book.decide("EngineConfig.async_output", _toggle(async_output), _why(
                engine.async_output, AUTO_RULES["EngineConfig.async_output"], "graph engine; the run loop's choice"))
            book.decide("capture", "on" if self._device_type() == "cuda" else "off",
                        "CUDA graphs" if self._device_type() == "cuda" else "CPU device: eager stages")
            parameters = ", ".join(f"{name}={value}" for name, value in optimizer.parameters)
            book.decide("integrator", "none", "relaxation: the optimiser steps the state")
            book.decide("optimizer", optimizer.scheme, Reason(
                f"{parameters}; " + (f"cell filter {optimizer.cell_filter}" if optimizer.cell_filter else "fixed cell"),
                code=f"optimizer-{optimizer.scheme}"))
            book.decide("compute_stress", _toggle(npt), Reason(
                "a cell filter needs the stress path and the deformation-aware neighbour test" if npt else
                "fixed cell: no stress", rule=10, code="rule-10-optimizer"))
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
            # The record names the scheme that runs (6-T11, phase 4 open finding 2); Langevin and BAOAB keep the
            # baseline's text, which the phase 4 plan comparison relies on.
            if npt and integrator.scheme == "baoab":
                book.decide("integrator", "LangevinBAOAB", f"NPT, {integrator.cell_mode} cell, externalstress="
                            f"{_stress_text(integrator)}, disable_cell_langevin="
                            f"{bool(integrator.disable_cell_langevin)} (workflow.baoab_settings)")
            elif not npt and integrator.scheme == "langevin":
                book.decide("integrator", "Langevin", f"NVT, fixcm={integrator.fixcm} (workflow.GPUEngine)")
            else:
                name = SCHEME_CLASS_NAMES.get(integrator.scheme, integrator.scheme)
                cell = f", {integrator.cell_mode} cell" if npt else ""
                book.decide("integrator", name, f"{integrator.ensemble.upper()}{cell}, fixcm={integrator.fixcm} "
                            f"(fastmace.integrators, scheme {integrator.scheme!r})")
            book.decide("compute_stress", _toggle(npt), "variable cell (NPT)" if npt else
                        "fixed cell: the graph engine computes no stress")
            integrator = _fixcm_projection(book, integrator, npt=npt, mc_moves=bool(mc and mc[0]))
            integrator = _dof(book, integrator)

        capturing = not evaluation and self._device_type() == "cuda"
        force_graphs = engine.force_graphs if engine.force_graphs != "auto" else "fit" if capturing else "keep"
        book.decide("EngineConfig.force_graphs", force_graphs, _why(
            engine.force_graphs, AUTO_RULES["EngineConfig.force_graphs"],
            "CUDA graphs: one force graph when the next padded size would not fit the free memory (CAP-T01)"
            if capturing else "no CUDA graphs"))
        phase_3 = self._phase_3_fields(book, dtype, rule_11, polar=bool(traits.polar), npt=npt,
                                       evaluation=evaluation, mc=mc, optimizer=optimizer)
        return dict(
            tier="compiled" if evaluation else TIER,
            rewrites=dataclasses.replace(rewrites, **rewrite_values),
            engine=dataclasses.replace(engine, backend=backend, pipelined=_toggle(pipelined),
                                       async_output=_toggle(async_output), cueq_conv_fusion=_toggle(bool(fusion)),
                                       expandable_segments=expandable, fp64_emulation=fp64_emulation,
                                       fp64_emulation_mantissa_bits=mantissa_bits, force_graphs=force_graphs,
                                       **phase_3["engine"]),
            precision=dataclasses.replace(self.precision, dtype=dtype, **phase_3["precision"]),
            neighbours=dataclasses.replace(self.neighbours, sort_receivers=_toggle(values["sort_receivers"]),
                                           gpu_neighbour_list=_toggle(values["gpu_neighbour_list"]),
                                           **phase_3["neighbours"]),
            polar=polar,
            integrator=None if evaluation else integrator,
            optimizer=optimizer,
            decisions=tuple(book.decisions),
            requested=requested,
            auto_options_off=tuple(values["_auto_off"].items()),
        )

    def _phase_3_fields(self, book, dtype, rule_11=None, *, polar=False, npt=False, evaluation=False, mc=None,
                        optimizer=None):
        """The phase 3 "auto" fields (plan task 3-T01): rule 11's skin and headroom (3-T05), the re-capture
        budget (3-T04) and the MC energy precision (3-T11, which changes it only for MC plans).

        ``rule_11`` is ``(applies, Rule11)`` from the minimal planner, which knows the workload. Rule 11
        sizes the "auto" skin and headroom only where it applies: workloads with MC moves or declared
        species changes (conductor answer 3-T05). Elsewhere, and without a workload (this planner's own
        plans, evaluations), "auto" keeps phase 1a's skin and headroom, and the decision records what the
        rule would have given.

        The re-capture budget (3-T04): the variant cap. A run that has re-captured as many graphs as it may
        hold at once has cycled through its whole working set of edge sizes and stages, so its variants do
        not fit, and the compiled tier (no graphs) is the better plan for it. The rule needs no run length,
        GPU or structure. With PolarMACE and a variable cell the plan also records the per-step k-vector
        check, one host sync after every force stage (3-T04, conductor answer 1).

        The MC energy precision (3-T11; SPEC §10.4, decision 35): ``mc`` is ``(mc_moves,
        Workload.mc_energy_precision)``. Without MC moves it is the force precision, which nothing reads.
        For an MC plan, "auto" is the force precision pending the move-energy gate, which keeps it if the
        gate passes and revises the plan to float64 otherwise (``fallback.mc_energy_revision``); an
        explicit precision is gated as it is, and float32 below a float64 force precision is refused.

        A relaxation (``optimizer``, 4-T05) sizes the "auto" skin from the optimiser's largest step instead
        (``neighbors.adaptive.optimizer_skin``): its atoms move up to ``maxstep_A`` per step, not thermally."""
        n, e, p = self.neighbours, self.engine, self.precision
        applies, sized = rule_11 if rule_11 is not None else (False, None)
        neighbours = {}
        # Rule 11's skin; an explicit other value is an override (SPEC §9.3, §9.5; conductor's choice 4).
        if optimizer is not None:
            from fastmace.neighbors.adaptive import OPTIMIZER_SKIN_STEPS, optimizer_skin

            maxstep = float(optimizer.parameter("maxstep_A", 0.2))
            rule_skin, code = optimizer_skin(maxstep), "rule-11-optimizer"
            explanation = (f"relaxation: {OPTIMIZER_SKIN_STEPS:g} x maxstep {maxstep:g} A, at least "
                           f"{DEFAULT_SKIN_A:g} A (at least 1.5 full steps between rebuilds)")
        elif applies:
            rule_skin, code, explanation = sized.skin_A, "rule-11", sized.text()
        else:
            rule_skin, code = DEFAULT_SKIN_A, "rule-11-fixed"
            would = "" if sized is None else f"; rule 11 would give {sized.skin_A:g} A"
            explanation = "phase 1a's skin: rule 11 sizes only MC and species-change workloads" + would
        if n.skin == "auto" or n.skin == rule_skin:
            skin = rule_skin
            why = Reason(_why(n.skin, AUTO_RULES["NeighbourConfig.skin"], explanation), rule=11, code=code)
        else:
            skin = n.skin
            why = rules.override(11, f"{rule_skin:g} A", detail=f"NeighbourConfig.skin={skin:g} A kept as given")
        book.decide("NeighbourConfig.skin", f"{skin:g} A", why)
        neighbours["skin"] = skin
        if n.headroom == "auto" and applies:
            neighbours["capacity_headroom"] = sized.capacity_headroom
            book.decide("NeighbourConfig.headroom", "fixed", Reason(_why(
                "auto", AUTO_RULES["NeighbourConfig.headroom"],
                f"capacity_headroom {sized.capacity_headroom:g} from the skin, edge_bucket_fraction as given"),
                rule=11, code="rule-11"))
        elif applies and n.capacity_headroom != sized.capacity_headroom:
            book.decide("NeighbourConfig.headroom", "fixed", rules.override(
                11, f"capacity_headroom {sized.capacity_headroom:g}",
                detail=f"NeighbourConfig.headroom='fixed', capacity_headroom {n.capacity_headroom:g} and "
                       "edge_bucket_fraction as given"))
        else:
            would = (f"; rule 11 would give {sized.capacity_headroom:g}"
                     if n.headroom == "auto" and sized is not None else "")
            book.decide("NeighbourConfig.headroom", "fixed", Reason(_why(
                n.headroom, AUTO_RULES["NeighbourConfig.headroom"],
                f"capacity_headroom {n.capacity_headroom:g} and edge_bucket_fraction as given" + would),
                rule=11, code="rule-11-fixed" if n.headroom == "auto" else "rule-11"))
        neighbours["headroom"] = "fixed"
        if n.d3_skin == "auto":  # the minimal planner sizes it when the force field has D3 (3b-T02)
            from fastmace.neighbors.adaptive import D3_FLOOR_SKIN_A

            neighbours["d3_skin"] = D3_FLOOR_SKIN_A
            book.decide("NeighbourConfig.d3_skin", "unused", _why(
                "auto", AUTO_RULES["NeighbourConfig.d3_skin"], f"no D3 list; {D3_FLOOR_SKIN_A:g} A, the floor"))
        budget = e.graph_variant_cap if e.recapture_budget == "auto" else e.recapture_budget
        buckets = ("" if not applies or sized.recapture_budget is None else
                   f"; rule 11 expects at most {sized.recapture_budget} padded edge buckets inside the capacity")
        if budget == e.graph_variant_cap:
            why = Reason(_why(
                e.recapture_budget, AUTO_RULES["EngineConfig.recapture_budget"],
                f"the graph variant cap ({e.graph_variant_cap}): past it the run warns, records and the report "
                "recommends the compiled tier (3-T04)" + buckets), rule=11, code="rule-11")
        else:
            why = rules.override(11, str(e.graph_variant_cap), detail=f"EngineConfig.recapture_budget={budget} kept "
                                 f"as given; the rule gives the graph variant cap{buckets}")
        book.decide("EngineConfig.recapture_budget", str(budget), why)
        if polar and npt and not evaluation:
            book.decide("overflow check", "per step", "PolarMACE with a variable cell: the k-vector flag is read "
                        "after every force stage, one host sync per step (3-T04)")
        moves, requested = mc if mc is not None else ((), "auto")
        rule = AUTO_RULES["PrecisionConfig.mc_energy"]
        if p.mc_energy != "auto":
            mc_energy, why = p.mc_energy, _why(p.mc_energy, rule, "kept as given")
        elif not moves:
            mc_energy, why = dtype, Reason(_why("auto", rule, "the force precision: no MC moves, no MC energies"),
                                           code="mc-energy-no-mc")
        elif requested == "auto":
            mc_energy = dtype
            why = Reason(_why("auto", rule, f"pending the move-energy gate: the force precision {dtype} if it passes, "
                              "float64 otherwise, while MD forces keep theirs (SPEC §10.4, §12.3, decision 35)"),
                         code="mc-energy-gate-pending")
        else:
            mc_energy = requested
            why = Reason(f"Workload.mc_energy_precision={requested!r}, checked by the move-energy gate (SPEC §10.4)",
                         code="mc-energy-requested")
            if requested == "float32" and dtype == "float64":
                book.refuse("Workload.mc_energy_precision='float32' is below the float64 force precision: MC energies "
                            "are never evaluated below the force precision (SPEC §10.4); use 'auto' or 'float64'",
                            "mc-energy-below-force-precision")
        book.decide("PrecisionConfig.mc_energy", mc_energy, why)
        return {"neighbours": neighbours, "engine": {"recapture_budget": budget},
                "precision": {"mc_energy": mc_energy}}

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
        # Owner decision, 21 October 2026 (OPT-DEF-2): on wherever fused_folded_backward is; bitwise the same.
        needs("folded_backward_tiles", r.folded_backward_tiles != "off", v["fused_folded_backward"],
              "fused_folded_backward")
        needs("segmented_first_sum", r.segmented_first_sum != "off", v["reassociate_first"] and v["sort_receivers"],
              "reassociate_first and sort_receivers")
        note("gpu_neighbour_list", n.gpu_neighbour_list != "off", "graph engine")
        needs("fused_radial_hidden", r.fused_radial_hidden == "on", v["fold_cutoff"], "fold_cutoff")
        if r.fused_radial_hidden == "auto":
            v["_why"]["fused_radial_hidden"] = "off: only an explicit 'on' enables it (no measured gain)"
        for name in ("direct_node_feats", "folded_constants"):  # opt-3 OPT3-T05 (exact, opt-in)
            needs(name, getattr(r, name) == "on", v["fold_cutoff"], "fold_cutoff")
            if getattr(r, name) == "auto":
                v["_why"][name] = "off: only an explicit 'on' enables it (OPT3-T05, opt-in)"
        needs("streaming_conv", r.streaming_conv != "off", hybrid and v["fold_cutoff"] and v["sort_receivers"],
              "the hybrid backend, fold_cutoff (stage 1) and sort_receivers (a hard requirement: streaming "
              "kernels on unsorted edges give wrong results)")
        # opt-6 OPT6-T02, owner decision 60 (27 October 2026): on wherever streaming_conv and fused_folded_backward
        # are; the op itself keeps cells whose weights fit a few L2s unchunked (UNCHUNKED_L2_MULTIPLE).
        needs("chunked_conv_backward", r.chunked_conv_backward != "off",
              v["streaming_conv"] and v["fused_folded_backward"], "streaming_conv and fused_folded_backward")
        # opt-8 OPT8-T02; owner decision 64 (27 October 2026): on for MH-1 (stage 2) in float32 plans that are not
        # reproducible (the FastMACE rules).
        needs("fused_folded_forward", r.fused_folded_forward != "off", v["chunked_conv_backward"],
              "chunked_conv_backward")
        # opt-6 OPT6-T05 (certified, owner decision 61); default on by owner decision 62 (27 October 2026) wherever
        # fold_cutoff is, float32 and not reproducible (stage 2, _fastmace_rules).
        needs("radial_tables", r.radial_tables != "off", v["fold_cutoff"], "fold_cutoff")
        # ideas-20261027 D2; owner decision 65: "auto" on in float64 MH-1 plans on the measured GPUs (the FastMACE
        # rules).
        needs("w_tables", r.w_tables != "off", v["fold_cutoff"], "fold_cutoff")
        # opt-9 OPT9-T03; decision 66 (owner, 29 October 2026): "auto" on for MH-1 on DECISION_66_WPRIME_GPUS (stage
        # 2), in float32 plans that are not reproducible, with radial_tables and chunked_conv_backward (the FastMACE
        # rules).
        needs("wprime_backward", r.wprime_backward != "off", v["fold_cutoff"], "fold_cutoff")
        note("block_linear", hybrid if r.block_linear == "auto" else r.block_linear == "on",
             f"stage 1: {backend_name} backend")
        needs("grouped_linear", r.grouped_linear != "off", v["block_linear"], "block_linear")
        needs("merged_residual", r.merged_residual != "off", v["block_linear"], "block_linear")
        needs("gated_rows", r.gated_rows != "off", v["merged_residual"], "merged_residual")
        needs("segment_gemm", r.segment_gemm != "off", v["gated_rows"], "gated_rows")
        # opt-10 OPT10-T01: opt-in until its whole-step A/Bs (decision 66: per-GPU defaults by exact device name).
        needs("gate_epilogue", r.gate_epilogue == "on", v["segment_gemm"], "segment_gemm")
        if r.gate_epilogue == "auto":
            v["_why"]["gate_epilogue"] = "off: only an explicit 'on' enables it (OPT10-T01, opt-in)"
        # Owner decision 57, 24 October 2026 (OPT4-T04): on by default for the generic MACE family only; stage 2
        # drops it for the other families, the FastMACE rules in float64 and reproducible plans.
        needs("grouped_launch", r.grouped_launch != "off", v["block_linear"], "block_linear")
        note("species_skip", r.species_skip != "off", "stage 1: graph engine, any backend")
        # OPT-T08 records, on by default (owner decision, 21 October 2026, OPT-DEF-2); stage 2 keeps them only
        # where the model takes them, and the FastMACE rules drop them in float64 and reproducible plans.
        note("species_tables", r.species_tables != "off", "stage 1: graph engine, any backend")
        needs("fold_readout_tail", r.fold_readout_tail != "off", v["block_linear"], "block_linear")
        # fp32 profile (28 October 2026): opt-in, needs the species table of the first linear_up and the merged
        # residual product it then replaces half of.
        needs("species_residual", r.species_residual == "on", v["species_tables"] and v["merged_residual"],
              "species_tables and merged_residual")
        if r.species_residual == "auto":
            v["_why"]["species_residual"] = "off: only an explicit 'on' enables it (fp32 profile, opt-in)"
        needs("fold_product_linear", r.fold_product_linear != "off", v["block_linear"] and v["species_tables"],
              "block_linear and species_tables")
        note("single_graph_strain", r.single_graph_strain != "off",
             "acts where stress is computed (NPT)" if npt else "process-wide; acts only where stress is computed")
        note("combo_kernels", r.combo_kernels == "on",
             "on: Inductor's pointwise combo kernels in the compiled model" if r.combo_kernels == "on"
             else "off: only an explicit 'on' enables it (opt-in until its A/B on both GPUs)")
        # ideas-20261027 D3; owner decision 65 as w_tables (the FastMACE rules).
        note("save_transcendentals", r.save_transcendentals != "off",
             "transcendental ops saved for the backward, not recomputed (ideas-20261027 D3)")
        needs("tf32x3_gemms", r.tf32x3_gemms == "on", v["block_linear"], "block_linear")
        if r.tf32x3_gemms == "auto":
            v["_why"]["tf32x3_gemms"] = "off: only an explicit 'on' enables it (OPT4-T03, accuracy screen)"
        # Owner decision 58, 24 October 2026 (OPT4-T02): on for the qualified POLAR-1 checkpoints on the hybrid;
        # stage 2 drops them for every other model, the FastMACE rules in float64 and reproducible plans.
        for name in POLAR_GRAPH_RECORDS:
            note(name, hybrid if getattr(r, name) == "auto" else getattr(r, name) == "on",
                 f"stage 1: {backend_name} backend")
        return v

    def _stage2(self, book, v, traits, explicit, model_label=None, hardware=None):
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
        if v["grouped_launch"] and "grouped_launch" not in explicit and not _generic_mace_family(traits):
            # Measured on the generic family (OPT4-T04); MH-1 lost 0.5 % on sm_89 (OPT-T03b) and gains on the
            # RTX 5060 Ti (decision 66).
            name = getattr(hardware, "name", None)
            if _mh1(model_label) and name in DECISION_66_GROUPED_LAUNCH_GPUS:
                v["_why"]["grouped_launch"] += f"; stage 2: on, MH-1 on a measured GPU, decision 66 ({name})"
            else:
                reason = (f"not the generic MACE family, decision 57 (interactions "
                          f"{sorted(set(traits.interaction_classes))}{', PolarMACE' if traits.polar else ''})")
                if _mh1(model_label):
                    reason += f"; MH-1 not on a measured GPU, decision 66 ({name})"
                v["grouped_launch"] = False
                v["_auto_off"]["grouped_launch"] = reason
                v["_why"]["grouped_launch"] += f"; stage 2: off, {reason}"
        for name in POLAR_GRAPH_RECORDS:
            if v[name] and name not in explicit and not _qualified_polar(traits, model_label):
                # Gated on the three POLAR-1 checkpoints (OPT4-T02); other models were not screened.
                found = f"PolarMACE, label {model_label}" if traits.polar else traits.model_class
                reason = (f"not a qualified POLAR-1 checkpoint, decision 58 ({found}; qualified: "
                          f"{', '.join(POLAR_GRAPH_LABELS)})")
                v[name] = False
                v["_auto_off"][name] = reason
                v["_why"][name] += f"; stage 2: off, {reason}"
            elif v[name] and name not in explicit:
                v["_why"][name] += f"; stage 2: {model_label}, decision 58"
        if v["fused_folded_forward"] and "fused_folded_forward" not in explicit:
            # Decision 64: measured faster on MH-1 only (POLAR-1-S neutral, OMOL-0 XL 3.5 % slower; OPT8-T02).
            if _mh1(model_label):
                v["_why"]["fused_folded_forward"] += f"; stage 2: {model_label}, decision 64"
            else:
                reason = f"not MH-1, decision 64 (label {model_label})"
                v["fused_folded_forward"] = False
                v["_auto_off"]["fused_folded_forward"] = reason
                v["_why"]["fused_folded_forward"] += f"; stage 2: off, {reason}"
        if v["wprime_backward"] and "wprime_backward" not in explicit:
            # Decision 66: measured faster for MH-1 on these GPU models only (OPT9-T05), matched by exact name.
            name = getattr(hardware, "name", None)
            if _mh1(model_label) and name in DECISION_66_WPRIME_GPUS:
                v["_why"]["wprime_backward"] += f"; stage 2: on, MH-1 on a measured GPU, decision 66 ({name})"
            else:
                reason = (f"not MH-1, decision 66 (label {model_label})" if not _mh1(model_label)
                          else f"not a measured GPU, decision 66 ({name})")
                v["wprime_backward"] = False
                v["_auto_off"]["wprime_backward"] = reason
                v["_why"]["wprime_backward"] += f"; stage 2: off, {reason}"

    def _fastmace_rules(self, book, v, hybrid, explicit, dtype="float32", reproducible=False, capability=None,
                        decision_65="no hardware given, decision 65", species=None):
        """FastMACE's hard rules on the rewrites, then one decision per rewrite field and neighbour option."""
        if v["tf32x3_gemms"]:
            from fastmace.kernels.triton.tf32x3_gemm import MIN_CAPABILITY

            # OPT4-T03: Triton's tf32x3 is float32 only and needs TF32 tensor cores (compute capability >= 8.0).
            if dtype != "float32":
                book.refuse(f"RewriteConfig.tf32x3_gemms='on' needs float32, not {dtype}", "tf32x3-needs-float32")
            elif capability not in (None, (0, 0)) and tuple(capability) < MIN_CAPABILITY:  # (0, 0): the CPU, PyTorch
                book.refuse(f"RewriteConfig.tf32x3_gemms='on' needs compute capability >= 8.0, the GPU has "
                            f"{capability[0]}.{capability[1]}", "tf32x3-needs-sm80")
            if v["grouped_launch"]:
                book.refuse("RewriteConfig.tf32x3_gemms and grouped_launch both reroute the block linears' products",
                            "rewrite-conflict")
        # Owner decision 61 (27 October 2026): the certified class bounds float32 rounding, so float32 plans only.
        # Decisions 62 and 63 (27 October 2026): "auto" is on with fold_cutoff in float32 and float64 plans that are
        # not reproducible (float64 tables are held to float64 rounding, f64-T01).
        if v["radial_tables"] and "radial_tables" not in explicit and (dtype not in ("float32", "float64")
                                                                       or reproducible):
            v["radial_tables"] = False
            v["_why"]["radial_tables"] += ("; off: reproducible=True keeps MACE's radial chain" if reproducible
                                           else f"; off: {dtype} plans keep MACE's radial chain")
        # Float64 tables (f64-T01): an explicit "on" is allowed in float64 plans outside reproducible ones.
        if v["radial_tables"] and (dtype not in ("float32", "float64") or (dtype == "float64" and reproducible)):
            book.refuse(f"RewriteConfig.radial_tables='on' needs float32, or float64 outside reproducible plans, not "
                        f"{dtype}{' (reproducible)' if reproducible else ''}", "radial-tables-needs-float32")
        # OPT-DEF-2 (owner, 21 October 2026): the rounding-class OPT-T08 records keep the stock path in float64 and
        # reproducible plans. species_tables too: its tables are not bitwise the per-atom products on the GPU
        # (reproducible mode, local RTX 5060 Ti: forces differ by 8e-16, docs/validation/opt-def-2.md).
        # Decision 63 (27 October 2026): also on in float64 plans that are not reproducible (f64-T01: forces within
        # the stock float64 backends' own scatter).
        for name in ("species_tables", "fold_readout_tail", "fold_product_linear", "species_residual"):
            if v[name] and name not in explicit and (dtype not in ("float32", "float64") or reproducible):
                v[name] = False
                v["_why"][name] += ("; off: reproducible=True keeps the stock products" if reproducible
                                    else f"; off: {dtype} keeps the stock products (rounding class)")
        if v["fold_product_linear"] and not v["species_tables"]:
            if "fold_product_linear" in explicit:
                book.refuse("RewriteConfig.fold_product_linear='on' requires species_tables, which is off",
                            "rewrite-requirement-missing")
            v["fold_product_linear"] = False
            v["_why"]["fold_product_linear"] += "; off: requires species_tables, which is off"
        if v["species_residual"] and not v["species_tables"]:
            if "species_residual" in explicit:
                book.refuse("RewriteConfig.species_residual='on' requires species_tables, which is off",
                            "rewrite-requirement-missing")
            v["species_residual"] = False
            v["_why"]["species_residual"] += "; off: requires species_tables, which is off"
        if v["segment_gemm"] and "segment_gemm" not in explicit and (dtype != "float32" or reproducible):
            # Owner decision, 20 October 2026: float64 and the reproducible mode keep the stock products and join.
            v["segment_gemm"] = False
            v["_why"]["segment_gemm"] += ("; off: reproducible=True keeps the stock products" if reproducible
                                          else f"; off: {dtype} keeps the stock products (the Triton GEMM is float32)")
        if v["gate_epilogue"]:
            # OPT10-T01: the fused kernel is float32 IEEE; it runs only where segment_gemm's joined products do.
            if dtype != "float32":
                book.refuse(f"RewriteConfig.gate_epilogue='on' needs float32, not {dtype} (the fused kernel is "
                            "float32)", "gate-epilogue-needs-float32")
            elif not v["segment_gemm"]:
                book.refuse("RewriteConfig.gate_epilogue='on' requires segment_gemm, which is off",
                            "rewrite-requirement-missing")
            if v["tf32x3_gemms"]:
                book.refuse("RewriteConfig.gate_epilogue and tf32x3_gemms both reroute linear_2's input gradients",
                            "rewrite-conflict")
        for name in POLAR_GRAPH_RECORDS:  # decision 58, as decision 57: float64 and reproducible keep MACE's code
            if v[name] and name not in explicit and (dtype != "float32" or reproducible):
                v[name] = False
                v["_why"][name] += ("; off: reproducible=True keeps MACE's code" if reproducible
                                    else f"; off: {dtype} keeps MACE's code")
        if v["grouped_launch"] and "grouped_launch" not in explicit and (dtype != "float32" or reproducible):
            v["grouped_launch"] = False
            v["_why"]["grouped_launch"] += ("; off: reproducible=True keeps the stock products" if reproducible
                                            else f"; off: {dtype} keeps the stock products (the Triton GEMM is "
                                                 "float32)")
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
        if v["chunked_conv_backward"] and not (v["streaming_conv"] and v["fused_folded_backward"]):
            if "chunked_conv_backward" in explicit:
                book.refuse("RewriteConfig.chunked_conv_backward='on' requires streaming_conv and "
                            "fused_folded_backward, which are not both on", "rewrite-requirement-missing")
            v["chunked_conv_backward"] = False
            v["_why"]["chunked_conv_backward"] += "; off: requires streaming_conv and fused_folded_backward"
        if v["fused_folded_forward"] and not v["chunked_conv_backward"]:
            if "fused_folded_forward" in explicit:
                book.refuse("RewriteConfig.fused_folded_forward='on' requires chunked_conv_backward, which is off",
                            "rewrite-requirement-missing")
            v["fused_folded_forward"] = False
            v["_why"]["fused_folded_forward"] += "; off: requires chunked_conv_backward"
        if (v["fused_folded_forward"] and "fused_folded_forward" not in explicit
                and (dtype != "float32" or reproducible)):
            v["fused_folded_forward"] = False
            v["_why"]["fused_folded_forward"] += ("; off: reproducible=True keeps the cuBLAS product" if reproducible
                                                  else f"; off: {dtype} plans keep the cuBLAS product")
        if v["fused_folded_forward"] and dtype != "float32":
            book.refuse(f"RewriteConfig.fused_folded_forward='on' needs float32, not {dtype} (the fused kernel is "
                        "float32; float64 plans keep the cuBLAS product)", "fused-folded-forward-needs-float32")
        # Owner decision 65 (28 October 2026): "auto" keeps w_tables and save_transcendentals only in float64 MH-1
        # plans on DECISION_65_GPUS; w_tables also needs its requirements and at most DECISION_65_MAX_SPECIES.
        for name in ("w_tables", "save_transcendentals"):
            if not v[name] or name in explicit:
                continue
            reason = decision_65
            if reason is None and name == "w_tables":
                if not (v["radial_tables"] and v["chunked_conv_backward"]):
                    reason = "requires radial_tables and chunked_conv_backward"
                elif species is not None and species > DECISION_65_MAX_SPECIES:
                    reason = (f"{species} species, decision 65 measured at most {DECISION_65_MAX_SPECIES} (the "
                              "tables grow with the square of the species)")
            if reason is None:
                v["_why"][name] += "; on: a measured GPU, decision 65"
            else:
                v[name] = False
                v["_why"][name] += f"; off: {reason}"
                if reason is not decision_65:  # the workload's limits; the plan's dtype and GPU are not listed
                    v["_auto_off"][name] = reason
        if decision_65 is not None:  # float64 records: stage 2's drops are not listed outside decision 65's plans
            for name in ("w_tables", "save_transcendentals"):
                if name not in explicit:
                    v["_auto_off"].pop(name, None)
        if v["w_tables"] and not (v["radial_tables"] and v["chunked_conv_backward"]):
            book.refuse("RewriteConfig.w_tables='on' requires radial_tables and chunked_conv_backward, which are not "
                        "both on", "rewrite-requirement-missing")
        if v["w_tables"] and (dtype != "float64" or reproducible):
            book.refuse(f"RewriteConfig.w_tables='on' needs float64 outside reproducible plans, not {dtype}"
                        f"{' (reproducible)' if reproducible else ''}", "w-tables-needs-float64")
        if v["wprime_backward"] and "wprime_backward" not in explicit:
            if dtype != "float32" or reproducible:
                v["wprime_backward"] = False
                v["_why"]["wprime_backward"] += ("; off: reproducible=True keeps the chained backward" if reproducible
                                                 else f"; off: {dtype} plans keep the chained backward")
            elif not (v["radial_tables"] and v["chunked_conv_backward"]):
                v["wprime_backward"] = False
                v["_why"]["wprime_backward"] += "; off: requires radial_tables and chunked_conv_backward"
        if v["wprime_backward"] and (dtype != "float32" or reproducible):
            book.refuse(f"RewriteConfig.wprime_backward='on' needs float32 outside reproducible plans, not {dtype}"
                        f"{' (reproducible)' if reproducible else ''}", "wprime-backward-needs-float32")
        elif v["wprime_backward"] and not (v["radial_tables"] and v["chunked_conv_backward"]):
            book.refuse("RewriteConfig.wprime_backward='on' requires radial_tables and chunked_conv_backward, which "
                        "are not both on", "rewrite-requirement-missing")
        # plans/capacity.md CAP-T07: "auto" stays off here; rule 12 turns it on only when the plan would not fit the
        # free memory without it (decision 67's counterpart, planner.minimal._rule_12).
        if self.rewrites.recompute_conv_weights == "on" and not v["wprime_backward"]:
            book.refuse("RewriteConfig.recompute_conv_weights='on' requires wprime_backward, which is off",
                        "rewrite-requirement-missing")
        v["recompute_conv_weights"] = self.rewrites.recompute_conv_weights == "on" and v["wprime_backward"]
        v["_why"]["recompute_conv_weights"] = (
            "on: requested" if v["recompute_conv_weights"] else
            "off: on only when memory would not be enough without it: rule 12 for a cell that cannot fit, or a "
            "rebuild after an out-of-memory error at build or capture (CAP-T07)"
            if self.rewrites.recompute_conv_weights == "auto" else "off: requested")
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
        graphs on CUDA); the eager tier (2-T02) passes False and builds a plan of tier ``"eager"``, and
        the compiled tier (3-T06) passes False and gets a ``CompiledEngine`` on exact edge counts.
        """
        from fastmace.engine.graph import GraphEngine
        from fastmace.model.forces import MACEForceField
        from fastmace.state import SimulationState

        if plan.tier not in (TIER, EAGER_TIER, COMPILED_TIER):
            raise ValueError(f"FixedPlanner builds tier {TIER!r}, {EAGER_TIER!r} and {COMPILED_TIER!r} plans, "
                             f"not {plan.tier!r}")
        if plan.tier in (EAGER_TIER, COMPILED_TIER) and capture is not False:
            raise ValueError(f"a tier {plan.tier!r} plan runs without CUDA graphs: pass capture=False")
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
        refuse_unsupported(book, plan.integrator, atoms, plan.optimizer)
        book.raise_refusals()
        self.configure_process(plan.engine)
        if plan.engine.fp64_emulation == "on":  # owner decision 63: before the model's first float64 product
            from fastmace.model import compile as compiling

            compiling.configure_fp64_emulation(self.environ, plan.engine.fp64_emulation_mantissa_bits)

        from fastmace.rewrites import registry as reg

        registry = self.registry or reg.default_registry()
        source = model.model if hasattr(model, "calculator") else model.models[0]
        resolution = self.rewrite_resolution(plan, source, registry)
        npt = plan.optimizer.variable_cell if plan.optimizer is not None else plan.integrator.ensemble == "npt"
        field = MACEForceField(model, atoms, rewrites=resolution, engine=plan.engine, neighbours=plan.neighbours,
                               polar=plan.polar or PolarConfig(), compute_stress=npt, compile_cache=compile_cache,
                               registry=registry, check=check, float64_energy=plan.precision.float64_energy,
                               species=plan.superset or None)
        field = _with_d3(field, plan, atoms)
        state = SimulationState.from_atoms(atoms, field.device, plan.integrator.seed, compute_stress=npt)
        integrator = build_stepper(state, plan, temperature_K=temperature_K)
        if plan.engine.fp64_emulation == "on":
            _check_fp64_emulation(field.device)
        if plan.tier == COMPILED_TIER:
            from fastmace.engine.compiled import CompiledEngine

            return CompiledEngine(plan.engine, field, integrator, state, neighbours=plan.neighbours,
                                  constraints=constraints, dynamic_edges=True)
        return GraphEngine(plan.engine, field, integrator, state, neighbours=plan.neighbours, constraints=constraints,
                           capture=capture)


def _with_d3(field, plan: Plan, atoms):
    """``field`` with the plan's D3 term (``fastmace.model.dispersed``, 3b-T04); ``field`` itself without D3."""
    if plan.dispersion is None:
        return field
    from fastmace.model import dispersed

    return dispersed.wrap(field, plan.dispersion, atoms, skin=float(plan.neighbours.d3_skin))


def auto_options_off(plan: Plan) -> dict[str, str]:
    """{option: reason} that auto switched off for the model (metadata.json's ``auto_options_off``)."""
    return dict(plan.auto_options_off)


def enabled_options(plan: Plan) -> tuple[str, ...]:
    """The rewrite fields and neighbour options the plan turned on, in the research engine's order."""
    out = [name for name in REWRITE_FIELDS if getattr(plan.rewrites, name) == "on"]
    out += [name for name in ("sort_receivers", "gpu_neighbour_list") if getattr(plan.neighbours, name) == "on"]
    return tuple(out)


__all__ = (
    "AUTO_RULES", "COMPILED_TIER", "EAGER_TIER", "FixedPlanner", "ModelSettings", "PlanRefused", "STAGE2_SUPPORT",
    "TIER",
    "WORKLOAD_CONSTRAINTS", "auto_options_off", "detect_hardware", "enabled_options", "integrator_keywords",
    "workload_problems",
)
