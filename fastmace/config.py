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

This module holds no state: no caches, no registries, no switches.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import typing
from dataclasses import dataclass
from typing import Literal

Toggle = Literal["on", "off", "auto"]
# Execution modes, the tiers of SPEC §8 (§9.3 "execution mode"): the planner's ``mode=``.
Mode = Literal["graph", "compiled", "eager", "reference"]
MODES = typing.get_args(Mode)


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

    def __post_init__(self):
        _check_choices(self)


@dataclass(frozen=True)
class NeighbourConfig:
    """Skin neighbour list and padded edge buckets (SPEC §8.1; fastmace.neighbors)."""

    # Neighbour-list skin in Angstrom (--skin).
    skin: float = 0.3
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

    def __post_init__(self):
        _check_choices(self)
        _positive(self, "skin", "edge_multiple", "capacity_headroom")
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
    """Langevin NVT and Langevin-Hoover NPT (BAOAB) settings of main.py (SPEC §10).

    Units as in main.py: femtoseconds, GPa, and the friction in inverse ASE time units.
    """

    # "nvt": ASE Langevin (GPULangevin); "npt": ASE LangevinBAOAB (GPULangevinBAOAB).
    ensemble: Literal["nvt", "npt"] = "nvt"
    timestep_fs: float = 0.5
    # Langevin friction, inverse ASE time units; NPT uses T_tau = 1 / friction.
    friction: float = 0.02
    seed: int = 20261001
    # Remove the centre-of-mass motion (ASE Langevin fixcm).
    fixcm: bool = True
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
