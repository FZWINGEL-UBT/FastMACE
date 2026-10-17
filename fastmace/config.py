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

This module holds no state: no caches, no registries, no switches.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass
from typing import Literal

Toggle = Literal["on", "off", "auto"]


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
    # Run-segmented Triton sum of the reassociated first interaction (SEGMENTED_FIRST_SUM).
    segmented_first_sum: Toggle = "auto"
    # Receiver-streaming OEq convolution, OEq PR 223 kernels (STREAMING_CONVOLUTION;
    # GENERIC_STREAMING for MP-0 and the density family).
    streaming_conv: Toggle = "auto"
    # Element-dependent skip connection restricted to the structure's species (GENERIC_SPECIES_SKIP).
    species_skip: Toggle = "auto"
    # Radial hidden block as one Triton kernel pair (FUSED_RADIAL_HIDDEN); opt-in, no measured gain.
    fused_radial_hidden: Toggle = "auto"
    # cuEq Linear layers as per-l products with extracted weights (BLOCK_LINEAR).
    block_linear: Toggle = "auto"
    # Same-irrep input segments of one output segment in one product (GROUPED_BLOCK_LINEAR).
    grouped_linear: Toggle = "auto"
    # linear_1 and linear_res as one product per output segment (MERGED_RESIDUAL_LINEAR).
    merged_residual: Toggle = "auto"
    # Per-l products as in-place strided-batched GEMMs (INPLACE_BLOCK_LINEAR); opt-in, slower.
    inplace_block_linear: bool = False
    # MACE PR 1498 one-graph strain in the stress path (upstream_mace.use_single_graph_strain).
    single_graph_strain: Toggle = "auto"
    # PolarMACE: reassociated SparseUvuTensorProduct, MACE PR 1711 (polar_gpu.REASSOCIATE_UVU).
    polar_uvu: bool = True

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
