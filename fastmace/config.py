"""Configuration objects: every option of an engine travels in one of these (SPEC §3.2, §5).

The research engine sets its options as module globals (``gpu_langevin.FOLD_RADIAL_CUTOFF =
True``) from ``workflow.GPUEngine``. Here each option is a field of a frozen dataclass, and
``docs/design/switch-map.md`` maps every research-engine global and ``main.py`` flag to its
field or records why it was not carried over. The defaults are ``main.py``'s defaults of
17 October 2026.

``"auto"`` is a value, not a resolved setting: the planner resolves it with the named rule
of the switch map (from the backend, the execution tier and the model's ``rewrite_support``)
and records a reason. ``"on"`` asks for the option and is refused, with a reason, when the
model or the plan cannot take it; ``"off"`` disables it.

Phase 2 (plan task 2-T01) adds the records of the safety layer: ``FallbackConfig`` (§9.4),
``GateConfig`` (§12.3 "Resources") and ``RunnerConfig`` (§8.7), and the ``Mode`` of the planner.

Phase 3 (plan task 3-T01, ``docs/design/contracts-3.md``) adds rule 11's skin and headroom
(``NeighbourConfig``), the re-capture policy (``EngineConfig``), the MC energy precision
(``PrecisionConfig``), the MC driver's options (``MCConfig``) and the MC streams' seeds.

Phase 4 (plan task 4-T01, ``docs/design/physics.md``) adds the integrator ``scheme``, its ``dof`` convention
and scheme ``parameters`` to ``IntegratorConfig``, the ``"nve"`` ensemble, and ``OptimizerConfig``.

Phase 5 (plan task 5-T01, ``docs/design/planner-5.md``) adds ``AdaptationConfig`` (SPEC §9.6) and
``CalibrationConfig`` (§9.1). 5-T07 sets ``AdaptationConfig``'s defaults to decisions 52 and 53, and 5-T09
``CalibrationConfig``'s to decision 55; ``Simulation`` refuses the options they did not choose.

This module holds no state: no caches, no registries, no switches.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import typing
from dataclasses import dataclass
from typing import Literal

Toggle = Literal["on", "off", "auto"]
# Execution modes, the tiers of SPEC §8 (§9.3 "execution mode"): the planner's ``mode=``.
Mode = Literal["graph", "compiled", "eager", "reference"]
MODES = typing.get_args(Mode)
# MC energy precision (SPEC §10.4): "auto" = the force precision if the move-energy gate passes, else float64.
MCEnergyPrecision = Literal["auto", "float32", "float64"]
MC_ENERGY_PRECISIONS = typing.get_args(MCEnergyPrecision)
# The skin of phase 1a and today's default, Angstrom; rule 11's lower bound (3-T05).
DEFAULT_SKIN_A = 0.3
# The three seeded device streams of the MC driver (SPEC §8.5 "Random numbers per outcome").
MC_STREAMS = ("proposal", "acceptance", "velocity")
# Momentum treatments of moves that change masses (SPEC §10.4): exchange is for swaps only.
SWAP_VELOCITIES = ("exchange", "rescale", "redraw", "redraw_all")
TRANSMUTE_VELOCITIES = ("rescale", "redraw", "redraw_all")


def _check_choices(instance):
    """Raise ValueError for a Literal field outside its choices or a bool field that is not a bool."""
    hints = typing.get_type_hints(type(instance))
    for field in dataclasses.fields(instance):
        hint = hints[field.name]
        value = getattr(instance, field.name)
        if typing.get_origin(hint) is Literal:
            choices = typing.get_args(hint)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"{type(instance).__name__}.{field.name}: {value!r} is not one of {choices}")
        elif hint is bool and not isinstance(value, bool):
            raise ValueError(f"{type(instance).__name__}.{field.name}: {value!r} is not a bool")


def _positive(instance, *names):
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{type(instance).__name__}.{name} must be positive and finite, got {value!r}")


def _count(instance, name, minimum=0):
    value = getattr(instance, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{type(instance).__name__}.{name} must be an int of at least {minimum}, got {value!r}")


def _nonnegative(instance, *names):
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{type(instance).__name__}.{name} must be finite and nonnegative, got {value!r}")


@dataclass(frozen=True)
class RewriteConfig:
    """Model rewrites of SPEC §6.2 (registry records of ``fastmace.rewrites``).

    One field per user-visible switch. Where the research engine drives an MH-1 global and
    its ``GENERIC_*`` twin from one flag, the field covers both and the rewrite registry
    picks the variant that fits the model's interactions.
    """

    # MH-1 class: species-factored radial input (FACTOR_SPECIES_RADIAL).
    factor_species: Toggle = "auto"
    # Cutoff folded into the last radial layer (FOLD_RADIAL_CUTOFF).
    fold_cutoff: Toggle = "auto"
    # Density division before the bias-free linear_1 (DENSITY_BEFORE_LINEAR).
    density_first: Toggle = "auto"
    # First interaction summed per receiver and sender species before its last radial layer
    # (REASSOCIATE_FIRST_INTERACTION; GENERIC_REASSOCIATE for MP-0 and the density family).
    reassociate_first: Toggle = "auto"
    # One first layer for both radial networks (JOINT_RADIAL_INPUT).
    joint_radial_input: Toggle = "auto"
    # Triton input gradient of the folded last radial layer (FUSED_FOLDED_BACKWARD;
    # GENERIC_FUSED_RADIAL_BACKWARD for MP-0 and the density family).
    fused_folded_backward: Toggle = "auto"
    # That input gradient's tiles by the waves rule (kernels.triton.skinny_linear.tile_choice): 128-row tiles
    # where the 64-row grid is 2-8 waves over the SMs; bitwise the same (opt-2 OPT2-T03); auto = on with
    # fused_folded_backward (owner decision, 21 October 2026).
    folded_backward_tiles: Toggle = "auto"
    # Run-segmented Triton sum of the reassociated first interaction (SEGMENTED_FIRST_SUM).
    segmented_first_sum: Toggle = "auto"
    # Receiver-streaming OEq convolution, OEq PR 223 kernels (STREAMING_CONVOLUTION;
    # GENERIC_STREAMING for MP-0 and the density family).
    streaming_conv: Toggle = "auto"
    # Element-dependent skip connection restricted to the structure's species (GENERIC_SPECIES_SKIP).
    species_skip: Toggle = "auto"
    # Radial hidden block as one Triton kernel pair (FUSED_RADIAL_HIDDEN); opt-in, no measured gain.
    fused_radial_hidden: Toggle = "auto"
    # Without LAMMPS, the folded MH-1 forward skips the join of node features and attributes for handle_lammps,
    # whose strided slices the streaming convolution copied (opt-3 OPT3-T05); exact, opt-in, auto = off.
    direct_node_feats: Toggle = "auto"
    # The folded last radial layer's [W, b] and its halves built once, not per force call (opt-3 OPT3-T05);
    # exact, opt-in, auto = off.
    folded_constants: Toggle = "auto"
    # cuEq Linear layers as per-l products with extracted weights (BLOCK_LINEAR).
    block_linear: Toggle = "auto"
    # Same-irrep input segments of one output segment in one product (GROUPED_BLOCK_LINEAR).
    grouped_linear: Toggle = "auto"
    # linear_1 and linear_res as one product per output segment (MERGED_RESIDUAL_LINEAR).
    merged_residual: Toggle = "auto"
    # Residual products, gate and linear_2 of MH-1-class interactions on per-segment rows, without the
    # joins and copies between them (opt-1 OPT-T02 stage A); auto = on with merged_residual (owner, 20 Oct 2026).
    gated_rows: Toggle = "auto"
    # linear_2's products in those chains written straight into its joined output by a gathered-row Triton
    # GEMM where the shared shape rule picks it (opt-1 OPT-T02 stage B); auto = on with gated_rows in float32,
    # not reproducible (owner, 20 Oct 2026).
    segment_gemm: Toggle = "auto"
    # A block linear's small per-l products (too few row tiles for the GPU by the shared shape rule) in one
    # Triton launch (opt-1 OPT-T03 candidate 2); opt-in until its A/B on both GPUs.
    grouped_launch: Toggle = "auto"
    # Species-only atom-level products (embeddings, the first interaction's skip_tp and linear_up) as
    # per-species tables gathered each step (opt-1 OPT-T08 C1); auto = on where the model takes it, in float32
    # plans that are not reproducible (owner decision, 21 October 2026).
    species_tables: Toggle = "auto"
    # The last readout's first linear folded into the last product's linear and skip_tp (OPT-T08 C2); auto = on
    # with block_linear where the model takes it, float32, not reproducible (owner decision, 21 October 2026).
    fold_readout_tail: Toggle = "auto"
    # products.0.linear folded into the linear consumers of layer 0's output, with the species carried next to
    # the contraction (OPT-T08 C3); auto = on with species_tables and block_linear (owner, 21 October 2026).
    fold_product_linear: Toggle = "auto"
    # Inductor's combo kernels, pointwise only, in the compiled model: independent adjacent pointwise kernels in
    # one launch; rounding (the energy moves by ~1e-9 relative), not adopted (opt-3 OPT3-T04,
    # fastmace.rewrites.inductor); opt-in, auto = off.
    combo_kernels: Toggle = "auto"
    # Every atom GEMM (block linears, segment_gemm, species skip, reassociated first interaction) and its input
    # gradients as 3xTF32 through Triton's tf32x3 (opt-4 OPT4-T03 stage 1, fastmace.rewrites.precision); rounding,
    # not IEEE-equivalent under cancellation; float32 on compute capability >= 8.0 only; opt-in, auto = off.
    tf32x3_gemms: Toggle = "auto"
    # Per-l products as in-place strided-batched GEMMs (INPLACE_BLOCK_LINEAR); opt-in, slower.
    inplace_block_linear: bool = False
    # MACE PR 1498 one-graph strain in the stress path (upstream_mace.use_single_graph_strain).
    single_graph_strain: Toggle = "auto"
    # PolarMACE: reassociated SparseUvuTensorProduct, MACE PR 1711 (polar_gpu.REASSOCIATE_UVU).
    polar_uvu: bool = True
    # Self-check results cached by checkpoint, sources, rewrite set, versions and device kind (2-T19):
    # "use" reads and writes the cache, "off" neither, "refresh" runs every check and rewrites it.
    # No effect until 2-T19.
    selfcheck_cache: Literal["use", "off", "refresh"] = "use"

    def __post_init__(self):
        _check_choices(self)


@dataclass(frozen=True)
class EngineConfig:
    """Backend, compilation and CUDA-graph engine options (SPEC §8.1, §9.3)."""

    # "hybrid": OEq convolutions with cuEq; "cueq": cuEq throughout (--backend).
    backend: Literal["auto", "hybrid", "cueq"] = "auto"
    # torch.compile mode of the model (--compile-mode).
    compile_mode: Literal["default", "none"] = "default"
    device: str = "cuda:0"
    # Force stage launched before the host has read the step's flags, validated while it runs.
    pipelined: Toggle = "auto"
    # Samples and frames copied behind their step and written while the next one runs.
    async_output: Toggle = "auto"
    # One speculative combined AC+B graph per step with late validation and rollback; NVT only.
    speculative_graph: bool = False
    # Pipelined steps: padded edge size = last validated count x (1 + margin) (PIPELINE_EDGE_MARGIN).
    pipeline_edge_margin: float = 0.0
    # Pipelined steps: rebuild early once an atom moved this fraction of skin / 2 (PIPELINE_REBUILD_FRACTION).
    pipeline_rebuild_fraction: float = 0.8
    # Plain cuEq: cuEquivariance's fused convolution on any CUDA device, MACE PR 1641.
    cueq_conv_fusion: Toggle = "auto"
    # Compiled cuEq models trace only the first derivative (FIRST_DERIVATIVE_ONLY; not with OEq).
    first_derivative_only: bool = True
    # Keep Inductor's mm + full-matrix add unfused (UNFUSE_FULL_ADDMM).
    unfuse_full_addmm: bool = True
    # torch._dynamo cache_size_limit; beyond it dynamo silently runs eager (DYNAMO_CACHE_SIZE_LIMIT).
    dynamo_cache_size_limit: int = 64
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before the first CUDA allocation (ALLOCATOR_SETTINGS).
    expandable_segments: Toggle = "auto"
    # Release cached free blocks between a stage's eager warm-up and its capture.
    empty_cache_before_capture: bool = True
    # Eager warm-ups after the first capture run inside the graph pool.
    warmup_in_graph_pool: bool = True
    # Evaluation plans compile with the edge dimension dynamic from the first call, so a changed edge
    # count does not recompile (2-T13). No effect until 2-T13.
    dynamic_edges: bool = False
    # Keep each conversion's model as a template and give later plans a private copy of it instead of
    # reloading and converting the checkpoint (CUDA cuEq and hybrid only; opt-1 OPT-T06).
    model_template_cache: bool = True
    # Pipelined fixed-cell steps: an early neighbour rebuild runs on a side stream beside the force graph
    # in flight, instead of waiting for it (the rebuild reads positions, which B does not write). Exact:
    # the same operations on the same positions. On by default (owner decision, 21 October 2026; OPT2-T04).
    overlap_rebuild: bool = True
    # Pipelined steps: one CUDA graph per step (ACP, the flags' copy to the host and an event inside the
    # graph, then B) instead of two replays. Exact: the same kernels in the same order. Opt-in (OPT2-T04).
    step_graph: bool = False
    # Fixed-cell Langevin on CUDA: the AC/ACP stage (half kick, drift, neighbour checks, compaction, saved
    # velocities and flags) as four Triton kernels around PyTorch's centre-of-mass sums and cumsum instead of
    # about 45 PyTorch kernels. Exact: the same float64 operations in the same order, the same noise, so it
    # stays on in reproducible plans. Anything else (NPT, CPU, no Triton) keeps the PyTorch stage. On by
    # default (owner decision, 21 October 2026; OPT3-T03).
    fused_integrator: bool = True
    # Re-captures of the graphs one run may make (bucket growth, overflow redos, events that drop graphs)
    # before the run warns, records it and the report recommends the compiled tier (SPEC §8.1; no mid-run
    # downgrade, phase 3 conductor decision). "auto": a rule from the workload (3-T04, 3-T05); 0: none.
    recapture_budget: int | Literal["auto"] = "auto"
    # Captured graph variants kept at once, oldest dropped first (SPEC §8.1: dynamo's limit, at most 64).
    graph_variant_cap: int = 64

    def __post_init__(self):
        _check_choices(self)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"EngineConfig.device must be a device string, got {self.device!r}")
        _nonnegative(self, "pipeline_edge_margin")
        _positive(self, "pipeline_rebuild_fraction", "dynamo_cache_size_limit")
        if self.pipeline_rebuild_fraction > 1:
            raise ValueError(f"EngineConfig.pipeline_rebuild_fraction must be at most 1, "
                             f"got {self.pipeline_rebuild_fraction!r}")
        if not isinstance(self.dynamo_cache_size_limit, int):
            raise ValueError(f"EngineConfig.dynamo_cache_size_limit must be an int, "
                             f"got {self.dynamo_cache_size_limit!r}")
        if self.recapture_budget != "auto":
            _count(self, "recapture_budget")
        _count(self, "graph_variant_cap", minimum=1)
        if self.graph_variant_cap > min(64, self.dynamo_cache_size_limit):
            raise ValueError(f"EngineConfig.graph_variant_cap must be at most 64 and at most dynamo_cache_size_limit "
                             f"({self.dynamo_cache_size_limit}), got {self.graph_variant_cap!r}")


@dataclass(frozen=True)
class PrecisionConfig:
    """Floating-point precision of the force field (SPEC §4, §7.3)."""

    # Model and force precision (--dtype).
    dtype: Literal["auto", "float32", "float64"] = "auto"
    # TF32 matrix products; off everywhere in the research engine (main.py: "TF32 off").
    tf32: bool = False
    # Potential energy accumulated in float64 (SPEC §7.3; a reference treatment, not a rewrite).
    # False reproduces phase 1a's float32 sum, for A/B against the research engine only (1b-T03).
    float64_energy: bool = True
    # Precision of MC energy evaluations (SPEC §10.4): the planner's resolution of
    # Workload.mc_energy_precision, which is what users set. "auto": the force precision if the
    # move-energy gate passes, float64 otherwise (3-T11); MD forces keep `dtype`.
    mc_energy: MCEnergyPrecision = "auto"

    def __post_init__(self):
        _check_choices(self)


@dataclass(frozen=True)
class NeighbourConfig:
    """Skin neighbour list and padded edge buckets (SPEC §8.1; fastmace.neighbors)."""

    # Neighbour-list skin in Angstrom (--skin). "auto": rule 11 from the workload (3-T05; until then
    # DEFAULT_SKIN_A, phase 1a's 0.3).
    skin: float | Literal["auto"] = "auto"
    # Order candidate edges by receiver at every rebuild (sort_by_receiver; --sort-receivers).
    sort_receivers: Toggle = "auto"
    # Build the list on the GPU instead of with matscipy (GPU_NEIGHBOUR_LIST).
    gpu_neighbour_list: Toggle = "auto"
    # Granularity of edge capacities and padded edge counts (EDGE_MULTIPLE).
    edge_multiple: int = 256
    # Step between padded edge counts as a fraction of the initial count (EDGE_BUCKET_FRACTION).
    edge_bucket_fraction: float = 0.05
    # Pair-buffer capacity relative to the first build (SkinNeighbourList headroom).
    capacity_headroom: float = 1.15
    # "fixed": capacity_headroom and edge_bucket_fraction as given; "auto": rule 11 sizes them from the
    # workload (3-T05; until then "fixed").
    headroom: Literal["auto", "fixed"] = "auto"
    # The D3 pair list's own skin in Angstrom (SPEC §11.2; 3b-T02). "auto": rule 11 from the workload and
    # the barostat when the force field has D3; without D3 it stays "auto" and nothing reads it.
    d3_skin: float | Literal["auto"] = "auto"

    def __post_init__(self):
        _check_choices(self)
        if self.skin != "auto":
            _positive(self, "skin")
        if self.d3_skin != "auto":
            _positive(self, "d3_skin")
        _positive(self, "edge_multiple", "capacity_headroom")
        _nonnegative(self, "edge_bucket_fraction")
        if not isinstance(self.edge_multiple, int):
            raise ValueError(f"NeighbourConfig.edge_multiple must be an int, got {self.edge_multiple!r}")
        if self.capacity_headroom < 1:
            raise ValueError(f"NeighbourConfig.capacity_headroom must be at least 1, got {self.capacity_headroom!r}")


@dataclass(frozen=True)
class PolarConfig:
    """PolarMACE long range (SPEC §11.1). No effect on other models.

    The two PR 1762 options change the model (SPEC §6.2 "opt-in only"); 0 and None reproduce
    the released model.
    """

    # MACE PR 1762: Fukui weights normalised by s / (s^2 + (R sum|f|)^2); 0 <= R < 1.
    fukui_regularisation: float = 0.0
    # MACE PR 1762: smooth cap of each atom's charge dipole, (onset, ceiling) in e Angstrom.
    dipole_cap: tuple[float, float] | None = None
    # Relative widening of the k-vector superset for a variable cell ("auto": by the integrator).
    kspace_headroom: float | Literal["auto"] = "auto"

    def __post_init__(self):
        _nonnegative(self, "fukui_regularisation")
        if self.fukui_regularisation >= 1:
            raise ValueError(f"PolarConfig.fukui_regularisation needs 0 <= R < 1, got {self.fukui_regularisation!r}")
        if self.dipole_cap is not None:
            cap = tuple(float(value) for value in self.dipole_cap)
            if len(cap) != 2 or not 0.0 < cap[0] < cap[1] or not math.isfinite(cap[1]):
                raise ValueError(f"PolarConfig.dipole_cap needs 0 < onset < ceiling, got {self.dipole_cap!r}")
            object.__setattr__(self, "dipole_cap", cap)
        if self.kspace_headroom != "auto":
            _nonnegative(self, "kspace_headroom")


@dataclass(frozen=True)
class IntegratorConfig:
    """The integrator settings (SPEC §10): main.py's Langevin NVT and Langevin-Hoover NPT (BAOAB), and from
    phase 4 every scheme of ``fastmace.integrators.registry``.

    Units as in main.py: femtoseconds, GPa, and the friction in inverse ASE time units. ``scheme`` names the
    registry record the planner builds; ``None`` resolves from the ensemble (``DEFAULT_SCHEMES``), so every
    earlier config means what it meant. Scheme-specific settings (time constants, chain lengths, the
    Berendsen mask) are ``parameters``, in the units of the public record (``fm.integrators``) that sets
    them; the fields below are shared or Langevin's.
    """

    # "nve": no thermostat; "nvt": a thermostat (phase 1a: ASE Langevin); "npt": a barostat, variable cell
    # (phase 1a: ASE LangevinBAOAB). The stress path and rule 10 follow "npt".
    ensemble: Literal["nve", "nvt", "npt"] = "nvt"
    # The registry scheme ("langevin", "baoab", "nve", "csvr", "nhc", "mtk", "berendsen"); None: from the
    # ensemble (DEFAULT_SCHEMES).
    scheme: str | None = None
    # Degrees of freedom of targets and temperatures (owner decision D2): "fastmace", 3N - 3 under fixcm and
    # 3(N - n_fixed) with FixAtoms; "ase", the scheme's ASE class; "auto", the planner's choice (D2's
    # default, registry.DOF_AUTO). Langevin and BAOAB keep their count.
    dof: Literal["auto", "fastmace", "ase"] = "auto"
    # Scheme parameters, (name, value) pairs sorted by name; a mapping is normalised. Values are None,
    # bools, numbers, strings or tuples of them.
    parameters: tuple[tuple[str, typing.Any], ...] = ()
    timestep_fs: float = 0.5
    # Langevin friction, inverse ASE time units; NPT uses T_tau = 1 / friction.
    friction: float = 0.02
    seed: int = 20261001
    # Remove the centre-of-mass motion (ASE Langevin fixcm).
    fixcm: bool = True
    # How fixcm projects the random terms (3-T22, docs/design/fixcm-projection.md). "ase": ASE's, bit for bit
    # (its noise stream; right for equal masses only): plain mean of the displacements, P/(N m_i) of the
    # velocities, both scaled by sqrt(N/(N-1)). "mass_weighted": the Gaussian conditioned on a fixed centre
    # of mass, mass-weighted means, no scale factor: canonical with 3N-3 degrees of freedom, per species too.
    # "auto" (3-T22b): the planner's rule, "mass_weighted" under NVT fixcm with MC moves declared (their
    # redraw_all holds (3N-3) kT), else "ase"; outside a plan it reads as "ase".
    fixcm_projection: Literal["auto", "ase", "mass_weighted"] = "auto"
    # Steps of random numbers drawn and copied to the GPU at once; same numbers and order (NOISE_BATCH).
    noise_batch: int = 256
    # NPT: external pressure (ASE externalstress = -pressure).
    pressure_GPa: float = 0.0
    # NPT: "hydrostatic" (shape fixed, volume free) or "full" (fully flexible cell).
    cell_mode: Literal["hydrostatic", "full"] = "hydrostatic"
    # NPT: barostat time constant P_tau; None = ASE's default 20 T_tau.
    barostat_tau_fs: float | None = None
    # NPT: factor on ASE's heuristic barostat mass.
    barostat_mass_factor: float = 1.0
    # NPT, full cell: ASE 3.27's trace term on all nine drag elements; for comparisons with ASE only
    # (ASE_TRACE_DRAG_ALL_ELEMENTS).
    ase_trace_drag_all_elements: bool = False
    # Thermostat target at the start, K. None: set by the caller through Integrator.set_temperature
    # (main.py's ramp, phase 1a); the public integrator records always set it.
    temperature_K: float | None = None
    # NPT: external stress in GPa, ASE's externalstress sign (positive = tensile): a scalar or six
    # Voigt components (xx, yy, zz, yz, xz, xy). None: -pressure_GPa on the diagonal. Setting both
    # this and a nonzero pressure_GPa is refused.
    external_stress_GPa: float | tuple[float, float, float, float, float, float] | None = None
    # NPT: no Langevin noise and friction on the cell degrees of freedom (ASE LangevinBAOAB).
    disable_cell_langevin: bool = False

    def __post_init__(self):
        _check_choices(self)
        if self.scheme is None:
            object.__setattr__(self, "scheme", DEFAULT_SCHEMES[self.ensemble])
        if not isinstance(self.scheme, str) or not self.scheme:
            raise ValueError(f"IntegratorConfig.scheme must be a scheme name or None, got {self.scheme!r}")
        object.__setattr__(self, "parameters", scheme_parameters(self.parameters, "IntegratorConfig.parameters"))
        _positive(self, "timestep_fs", "friction", "noise_batch", "barostat_mass_factor")
        if self.barostat_tau_fs is not None:
            _positive(self, "barostat_tau_fs")
        if isinstance(self.pressure_GPa, bool) or not math.isfinite(self.pressure_GPa):
            raise ValueError(f"IntegratorConfig.pressure_GPa must be finite, got {self.pressure_GPa!r}")
        for name in ("seed", "noise_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"IntegratorConfig.{name} must be a nonnegative int, got {value!r}")
        if self.temperature_K is not None:
            _nonnegative(self, "temperature_K")
        stress = external_stress(self.external_stress_GPa, "IntegratorConfig.external_stress_GPa")
        if stress is not None and self.pressure_GPa != 0:
            raise ValueError("IntegratorConfig: give either pressure_GPa or external_stress_GPa, not both "
                             f"(pressure_GPa={self.pressure_GPa!r}, external_stress_GPa={self.external_stress_GPa!r})")
        object.__setattr__(self, "external_stress_GPa", stress)

    def effective_external_stress_GPa(self) -> float | tuple[float, ...]:
        """The external stress in GPa, ASE sign: ``external_stress_GPa``, else ``-pressure_GPa``."""
        return -self.pressure_GPa if self.external_stress_GPa is None else self.external_stress_GPa

    def parameter(self, name: str, default: typing.Any = None) -> typing.Any:
        """The scheme parameter ``name``, ``default`` if it is not set."""
        return dict(self.parameters).get(name, default)

    @property
    def variable_cell(self) -> bool:
        """Whether the cell changes: the ``"npt"`` ensemble."""
        return self.ensemble == "npt"


# The scheme of a config without one, by ensemble: phase 1a's two integrators, and NVE (4-T02).
DEFAULT_SCHEMES = {"nvt": "langevin", "npt": "baoab", "nve": "nve"}


@dataclass(frozen=True)
class FallbackConfig:
    """The fallback of SPEC §9.4 on a failing self-check, gate, compile or capture (decision 29).

    No effect on a run where nothing fails.
    """

    # "auto": diagnose and fall back to a validated slower plan, recorded; "refuse": raise as in 1b,
    # with the record.
    fallback: Literal["auto", "refuse"] = "auto"
    # Reduced-gate runs (one rattled copy, the failing quantity) per fallback; full gate runs are
    # counted apart, one per candidate to activate (§9.4 step 2).
    diagnosis_budget: int = 64
    # The reference tier may end the terminal sequence, if it fits (§9.4 step 5).
    reference_last: bool = True

    def __post_init__(self):
        _check_choices(self)
        _positive(self, "diagnosis_budget")
        if not isinstance(self.diagnosis_budget, int):
            raise ValueError(f"FallbackConfig.diagnosis_budget must be an int, got {self.diagnosis_budget!r}")


@dataclass(frozen=True)
class GateConfig:
    """Where the gate's references may run (SPEC §12.3 "Resources"): GPU float64 on the production
    cell, then CPU float64 within a time limit, then a reduced cell. Whether the gate runs is
    ``check=`` (decision 3), not this record."""

    # CPU float64 is tried after GPU memory runs out only if its estimated time is within this limit, s.
    # 0 never tries the CPU.
    cpu_float64_time_limit_s: float = 300.0
    # "auto": the largest supercell of the same primitive cell smaller than the production cell whose
    # every height is at least reduced_cell_min_height_cutoffs x the cutoff (2-T05); "off": never.
    reduced_cell: Literal["auto", "off"] = "auto"
    # The reduced cell's minimum height in cutoffs; below 1 a periodic image would sit inside the cutoff.
    reduced_cell_min_height_cutoffs: float = 2.0

    def __post_init__(self):
        _check_choices(self)
        _nonnegative(self, "cpu_float64_time_limit_s")
        _positive(self, "reduced_cell_min_height_cutoffs")
        if self.reduced_cell_min_height_cutoffs < 1:
            raise ValueError(f"GateConfig.reduced_cell_min_height_cutoffs must be at least 1, "
                             f"got {self.reduced_cell_min_height_cutoffs!r}")


@dataclass(frozen=True)
class RunnerConfig:
    """The subprocess runner of SPEC §8.7: a plan that a process constraint forbids in this process
    runs in a child, handed over through a restart write (2-T07)."""

    # The child's Python; None: sys.executable when the runner starts (resolved_python()).
    python: str | None = None
    # Variables set in the child, over the parent's (a mapping is normalised to sorted pairs).
    environment: tuple[tuple[str, str], ...] = ()
    # The child starts from the parent's environment; False: from `environment` alone.
    inherit_environment: bool = True
    # Total limit of one child run, s; None: none.
    timeout_s: float | None = None
    # A child without progress (no committed step) for this long is killed, s.
    stall_timeout_s: float = 1800.0
    # One child serves every run() of the Simulation, keeping its compiled engine (OPT-T07); False: a
    # fresh child per run(), resuming the last restart write (2-T07's path; also on Windows).
    persistent: bool = True

    def __post_init__(self):
        _check_choices(self)
        if self.python is not None and (not isinstance(self.python, str) or not self.python):
            raise ValueError(f"RunnerConfig.python must be a path or None, got {self.python!r}")
        environment = self.environment
        if isinstance(environment, (str, bytes)):
            raise ValueError(f"RunnerConfig.environment must be a mapping or (name, value) pairs, got {environment!r}")
        try:
            pairs = tuple(sorted((environment.items() if hasattr(environment, "items") else environment)))
        except (TypeError, ValueError):
            raise ValueError(f"RunnerConfig.environment must be a mapping or (name, value) pairs, "
                             f"got {environment!r}") from None
        for pair in pairs:
            if len(pair) != 2 or not isinstance(pair[0], str) or not pair[0] or not isinstance(pair[1], str):
                raise ValueError(f"RunnerConfig.environment: names and values must be strings, got {pair!r}")
        object.__setattr__(self, "environment", tuple((name, value) for name, value in pairs))
        if self.timeout_s is not None:
            _positive(self, "timeout_s")
        _positive(self, "stall_timeout_s")
        if not isinstance(self.persistent, bool):
            raise ValueError(f"RunnerConfig.persistent must be a bool, got {self.persistent!r}")

    def resolved_python(self) -> str:
        """The child's interpreter: ``python``, else this process's ``sys.executable``."""
        return self.python if self.python is not None else sys.executable


@dataclass(frozen=True)
class MCConfig:
    """Limits of the MC driver (SPEC §10.4; 3-T09). The moves themselves are ``fm.mc`` records, and the
    streams' seeds derive from the simulation's ``seed`` (``mc_stream_seeds``)."""

    # Attempts of one move in one block; the captured attempt graph is replayed this often at most.
    max_attempts_per_block: int = 10_000
    # Move records attached at once (run(mc=(...)) with several moves).
    max_move_sets: int = 8

    def __post_init__(self):
        _count(self, "max_attempts_per_block", minimum=1)
        _count(self, "max_move_sets", minimum=1)


# The placeholder of 5-T01 for a field an open owner decision of plans/phase-5.md fixed; every such field has its
# decided default now (5-T07, 5-T09), and no record accepts it.
PENDING = "pending"
# Which monitor triggers a tier switch: "recapture-time", the re-capture budget exceeded and the time spent
# re-capturing above ``recapture_time_fraction`` (SPEC §18 decision 52, D2 (a)); "recapture-budget", the
# budget alone (D2 (c), not chosen); "none", no automatic switch.
ADAPTATION_TRIGGERS = ("recapture-time", "recapture-budget", "none")


def _count_or_none(instance, name, minimum):
    if getattr(instance, name) is not None:
        _count(instance, name, minimum)


@dataclass(frozen=True)
class AdaptationConfig:
    """Runtime adaptation (SPEC §9.6; plan tasks 5-T01, 5-T07; docs/design/planner-5.md): what the monitor may
    change at a canonical restart point, as a new plan revision (§9.5).

    The defaults are the owner's decisions 52 (trigger, D2 (a)) and 53 (limits, D3 (a)): the re-capture budget
    exceeded and more than ``recapture_time_fraction`` of the wall time re-capturing switch graph -> compiled,
    one way, at most once per run. The record can hold the options not chosen (D2 (b)-(d), D3 (b)-(d));
    ``Simulation`` refuses them (``unsupported``). Mid-run fallbacks are ``FallbackConfig``'s (§9.4 step 5).
    """

    # "auto": adapt as §9.6 and the fields below say; "off": the monitor records, nothing switches.
    adapt: Literal["auto", "off"] = "auto"
    # The trigger: one of ADAPTATION_TRIGGERS ("none": no automatic switch).
    trigger: Literal["recapture-time", "recapture-budget", "none"] = "recapture-time"
    # The share of wall time spent re-capturing, over the segments since the last switch, above which an
    # exceeded re-capture budget switches tiers (decision 52). Measured in the run, so no per-GPU number.
    recapture_time_fraction: float = 0.2
    # D2 (b), not chosen: rule 11 re-planned with the measured rebuild rate.
    skin_replan: bool = False
    # Tier switches one run may make, a resume counting those in its revision chain (decision 53: 1);
    # None: no limit (D3 (d), not chosen).
    max_switches: int | None = 1
    # D3 (b), not chosen: back to the graph tier after this many quiet segments; None: one way.
    switch_back_after: int | None = None
    # D3 (d), not chosen: canonical restart points between two switches; None: no minimum.
    min_points_between: int | None = None
    # D3 (c), not chosen: switches under reproducible=True.
    reproducible: bool = False

    def __post_init__(self):
        _check_choices(self)
        _positive(self, "recapture_time_fraction")
        if self.recapture_time_fraction >= 1:
            raise ValueError(f"AdaptationConfig.recapture_time_fraction must be below 1, "
                             f"got {self.recapture_time_fraction!r}")
        _count_or_none(self, "max_switches", 0)
        _count_or_none(self, "switch_back_after", 1)
        _count_or_none(self, "min_points_between", 1)

    def unsupported(self) -> tuple[str, ...]:
        """The fields set to an option decisions 52 and 53 did not choose, in field order (``Simulation`` refuses
        them, code ``adaptation-unsupported``)."""
        return tuple(name for name in ADAPTATION_DECISION_FIELDS
                     if getattr(self, name) not in ADAPTATION_SUPPORTED[name])


# The AdaptationConfig fields owner decisions 52 (D2) and 53 (D3) fixed, and the values implemented (5-T07):
# the trigger of decision 52 or none; at most one switch, one way, never under reproducible=True.
ADAPTATION_DECISION_FIELDS = ("trigger", "skin_replan", "max_switches", "switch_back_after", "min_points_between",
                              "reproducible")
ADAPTATION_SUPPORTED = {"trigger": ("recapture-time", "none"), "skin_replan": (False,), "max_switches": (0, 1),
                        "switch_back_after": (None,), "min_points_between": (None,), "reproducible": (False,)}


@dataclass(frozen=True)
class CalibrationConfig:
    """The optional calibration of SPEC §9.1 (plan tasks 5-T01, 5-T09; owner decision 55, D5 (a)): off by
    default, it measures the backends the rules rank equal on this run's structure (``planner.calibrate``), and
    nothing it measures is ever persisted or read by another run. Its warm-up and pair count are
    ``planner.calibrate``'s constants."""

    # Off by default (§9.1): True measures the tie, MD runs only.
    calibrate: bool = False
    # Measured steps per candidate, after warm-up (§9.1: about 50).
    steps: int = 50
    # Decision 55: the other candidate replaces the rules' choice only when faster by more than this share in
    # every pair (above the 3 % cross-process noise).
    margin: float = 0.05
    # True, the measurement may change the plan (decision 55, D5 (a)); False, recorded only (D5 (b), not chosen:
    # Simulation refuses it, code calibration-unsupported).
    choose: bool = True

    def __post_init__(self):
        _check_choices(self)
        _count(self, "steps", minimum=1)
        _positive(self, "margin")
        if self.margin >= 1:
            raise ValueError(f"CalibrationConfig.margin must be below 1, got {self.margin!r}")


@dataclass(frozen=True)
class OptimizerConfig:
    """A structure relaxation (SPEC §10.3; plan task 4-T01): the optimiser scheme of
    ``fastmace.integrators.registry`` with its parameters, the optional cell filter with its parameters,
    and ASE's convergence (``fmax``, ``steps``).

    Units of the public records (``fm.optimize``): Angstrom, fs, GPa; ASE's argument names otherwise.
    """

    # The registry scheme: "lbfgs" (4-T05) or "fire" (4-T06).
    scheme: str
    # The optimiser's parameters, (name, value) pairs sorted by name (``fm.optimize.LBFGS`` and ``FIRE``).
    parameters: tuple[tuple[str, typing.Any], ...] = ()
    # The cell filter: None (fixed cell) or "frechet" (ASE's FrechetCellFilter, 4-T07).
    cell_filter: str | None = None
    # The cell filter's parameters (``fm.optimize.FrechetCell``).
    cell_filter_parameters: tuple[tuple[str, typing.Any], ...] = ()
    # ASE's convergence: the largest (constraint-adjusted) per-row force norm below fmax, eV/Angstrom.
    fmax: float = 0.05
    # Most optimiser steps (ASE's run(steps=) default).
    steps: int = 100_000_000

    def __post_init__(self):
        if not isinstance(self.scheme, str) or not self.scheme:
            raise ValueError(f"OptimizerConfig.scheme must be a scheme name, got {self.scheme!r}")
        if self.cell_filter is not None and (not isinstance(self.cell_filter, str) or not self.cell_filter):
            raise ValueError(f"OptimizerConfig.cell_filter must be None or a filter name, got {self.cell_filter!r}")
        object.__setattr__(self, "parameters", scheme_parameters(self.parameters, "OptimizerConfig.parameters"))
        object.__setattr__(self, "cell_filter_parameters", scheme_parameters(
            self.cell_filter_parameters, "OptimizerConfig.cell_filter_parameters"))
        if self.cell_filter is None and self.cell_filter_parameters:
            raise ValueError("OptimizerConfig.cell_filter_parameters need a cell_filter")
        _positive(self, "fmax")
        _count(self, "steps", minimum=0)

    def parameter(self, name: str, default: typing.Any = None) -> typing.Any:
        """The optimiser parameter ``name``, ``default`` if it is not set."""
        return dict(self.parameters).get(name, default)

    def cell_filter_parameter(self, name: str, default: typing.Any = None) -> typing.Any:
        """The cell filter parameter ``name``, ``default`` if it is not set."""
        return dict(self.cell_filter_parameters).get(name, default)

    @property
    def variable_cell(self) -> bool:
        """Whether a cell filter moves the cell."""
        return self.cell_filter is not None


def _parameter_value(value, name):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_parameter_value(item, name) for item in value)
    raise ValueError(f"{name} must be None, a bool, a number, a string or a tuple of them, got {value!r}")


def scheme_parameters(value, name="parameters") -> tuple[tuple[str, typing.Any], ...]:
    """``value`` (a mapping or (name, value) pairs) as sorted, hashable (name, value) pairs; lists become
    tuples, so a JSON round trip gives the same pairs. ValueError for a duplicate or non-string name, or a
    value that is not None, a bool, a finite number, a string or a tuple of them."""
    if isinstance(value, typing.Mapping):
        pairs = list(value.items())
    else:
        try:
            pairs = [tuple(pair) for pair in value]
        except TypeError:
            raise ValueError(f"{name} must be a mapping or (name, value) pairs, got {value!r}") from None
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"{name} must be a mapping or (name, value) pairs, got {value!r}")
    names = [key for key, _ in pairs]
    if any(not isinstance(key, str) or not key for key in names) or len(set(names)) != len(names):
        raise ValueError(f"{name}: parameter names must be distinct non-empty strings, got {names!r}")
    return tuple(sorted((key, _parameter_value(item, f"{name}[{key!r}]")) for key, item in pairs))


def mc_stream_seeds(seed: int) -> dict[str, int]:
    """The seeds of the MC streams (``MC_STREAMS``) for a simulation ``seed``: fixed 63-bit values from
    SHA-256, the same in every process and on every machine, and distinct from the integrator's ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"the MC streams need a nonnegative int seed, got {seed!r}")
    seeds = {}
    for stream in MC_STREAMS:
        digest = hashlib.sha256(f"fastmace-mc:{seed}:{stream}".encode()).digest()
        seeds[stream] = int.from_bytes(digest[:8], "little") >> 1
    return seeds


def external_stress(value, name="external_stress_GPa"):
    """``value`` as None, a float or a tuple of six floats (Voigt order), all finite; else ValueError."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number or six Voigt components, got {value!r}")
    if isinstance(value, (int, float)):
        components = (float(value),)
    else:
        try:
            components = tuple(float(component) for component in value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number or six Voigt components, got {value!r}") from None
        if len(components) != 6 or any(isinstance(component, bool) for component in value):
            raise ValueError(f"{name} must be a number or six Voigt components (xx, yy, zz, yz, xz, xy), "
                             f"got {value!r}")
    if not all(math.isfinite(component) for component in components):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return components[0] if len(components) == 1 else components
