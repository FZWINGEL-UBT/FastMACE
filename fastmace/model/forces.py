"""The ForceField protocol and its MACE implementation (SPEC §3.2, §7.1, §8.1).

The protocol is plan task 1a-T02's; ``MACEForceField`` (1a-T11) implements it from the research
engine's ``MACEForces`` (``langevin/gpu_langevin.py``). See docs/design/interfaces.md for who calls
what in a step, and ``fastmace.model.compile`` for model copies, compilation and the cache.

A force field owns the model (compiled and rewritten), its input batch, the skin neighbour list
with its edge buffers and device edge counter (so the streaming counter is per instance and
``bind_streaming_count`` disappears), and the fixed-address cell, volume and reciprocal cell
the model reads. It knows nothing about integrators or engines.

Two levels:

- ``evaluate(atoms, stress)``: the public one-off evaluation of §7.1, host in, host out;
- the graph-level methods the engine calls inside and between captured stages:
  ``__call__``, ``set_cell``, ``needs_rebuild`` and ``compact`` are graph-capturable (no host
  synchronisation, fixed addresses); ``rebuild`` and ``use_edges`` are host-side and run only
  between replays.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from fastmace.errors import ModelElementError, Reason, SpeciesNotInSuperset

if TYPE_CHECKING:
    import numpy as np
    import torch


class ForceOutput(NamedTuple):
    """One evaluation: ``energy`` ``(1,)`` eV, ``forces`` ``(N, 3)`` eV/Å, ``stress`` ``(3, 3)`` eV/Å³ or ``None``.

    Forces and stress in the force field's precision; the energy accumulated in float64 (SPEC
    §7.3), or in the force precision when the force field was built with ``float64_energy=False``.
    Owned by the model's output (copy them out before the next call). ``stress`` is MACE's: ASE
    sign convention, no kinetic part, ``None`` unless the force field was built with
    ``compute_stress``.
    """

    energy: torch.Tensor
    forces: torch.Tensor
    stress: torch.Tensor | None


class PerAtomEnergies(NamedTuple):
    """Per-atom float64 energies (eV) of the atom-resolved terms, and the rest of the total.

    ``energies.sum() + unresolved`` is the potential energy. ``terms`` names the terms each part
    holds: ``{"resolved": (...), "unresolved": (...)}`` (k-space electrostatics and the field term of
    PolarMACE are not atom-resolved).
    """

    energies: np.ndarray
    unresolved: float
    terms: dict


@runtime_checkable
class ForceField(Protocol):
    """Energy, forces and stress of one structure, with its own neighbour list.

    Built by the planner for one structure, species superset, precision and rewrite set; one
    instance per engine (two force fields from one compiled model step independently).
    """

    @property
    def device(self) -> torch.device:
        """Device of the model and every buffer."""
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """Model precision (float32 or float64); positions and cell come in as float64."""
        raise NotImplementedError

    @property
    def compute_stress(self) -> bool:
        """Whether ``__call__`` returns a stress (variable-cell integrators need it)."""
        raise NotImplementedError

    @property
    def padding_exact(self) -> bool:
        """Whether padded edges beyond the cutoff contribute exactly zero (graph tier needs True)."""
        raise NotImplementedError

    def evaluate(self, atoms: Any, stress: bool = False) -> tuple[float, np.ndarray, np.ndarray | None]:
        """One-off evaluation of an ASE ``Atoms`` (SPEC §7.1): energy (eV), forces, 3x3 stress or None.

        Not for the MD loop: it synchronises, rebuilds the neighbour list and returns host values.
        """
        raise NotImplementedError

    def __call__(self, positions: torch.Tensor) -> ForceOutput:
        """Energy, forces (and stress) at float64 ``positions`` ``(N, 3)`` on the current edges.

        Graph-capturable: copies the positions into the model's input leaf and runs the model;
        reads the edges that the last ``use_edges`` selected and the cell of the last ``set_cell``.
        This is stage B's force call.
        """
        raise NotImplementedError

    def set_cell(self, cell: torch.Tensor) -> None:
        """Copy a new float64 cell into the neighbour list and the model inputs (volume, reciprocal cell).

        Graph-capturable; only for ``compute_stress`` force fields (variable-cell runs).
        """
        raise NotImplementedError

    def needs_rebuild(self, positions: torch.Tensor, fraction: float = 1.0) -> torch.Tensor:
        """Device bool: has any atom moved more than ``fraction`` of half the skin since the last build?

        Deformation-aware in variable-cell runs. Graph-capturable. ``fraction < 1`` is the
        pipelined early-rebuild test.
        """
        raise NotImplementedError

    def compact(self, positions: torch.Tensor) -> None:
        """Stage C: write the candidate pairs within the cutoff to the front of the edge buffers.

        Graph-capturable; updates the device edge count (``edge_count``).
        """
        raise NotImplementedError

    def rebuild(self, positions: torch.Tensor) -> bool:
        """Rebuild the candidate pairs from float64 positions (host-side, between replays).

        Returns True if the edge capacity grew: buffers moved, so every captured graph is stale.
        """
        raise NotImplementedError

    def use_edges(self, num_edges: int) -> None:
        """Let the model read the first ``num_edges`` compacted edges (views, no copy; host-side)."""
        raise NotImplementedError

    @property
    def edge_count(self) -> torch.Tensor:
        """Device int64 scalar: edges within the cutoff after the last ``compact``."""
        raise NotImplementedError

    @property
    def edge_capacity(self) -> int:
        """Size of the edge buffers (host integer); padded edge sizes never exceed it."""
        raise NotImplementedError

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Persistent device buffers that the graph-level methods write, by name.

        Neighbour-list state (edge index, shifts, count, build cell, reference positions) and the
        model inputs the cell feeds (cell, volume, reciprocal cell). The engine saves and restores
        them around capture warm-ups; the runtime-state registry of SPEC §8.1 (1b) starts here.
        """
        raise NotImplementedError

    def info(self) -> Mapping[str, Any]:
        """Counters for provenance: neighbour rebuilds, edge capacity, applied rewrites."""
        raise NotImplementedError


# Rewrite records whose kernels read the force field's edge count (streaming_count): they leave out
# every edge at or after it, and they need receiver-sorted edges. With unsorted edges MH-1's energy
# was off by a factor of 2.5 while every self-check passed (1a-T07), because the self-checks build
# their own sorted graph.
STREAMING_RECORDS = ("streaming_conv", "generic_streaming")
SEGMENTED_RECORDS = ("segmented_first_sum", "generic_segmented_first_sum")  # edges grouped by sender species
# Workload constraint of the records whose rewritten model depends on the structure's species.
SPECIES_DEPENDENT = "fixed_species_superset"


def _numbers(species):
    """Symbols or atomic numbers as distinct atomic numbers (a set, sorted)."""
    from ase.data import atomic_numbers

    if isinstance(species, (str, int)):
        species = [species]
    return tuple(sorted({atomic_numbers[item] if isinstance(item, str) else int(item) for item in species}))


def _symbols(numbers):
    from ase.data import chemical_symbols

    return " ".join(chemical_symbols[z] if 0 < z < len(chemical_symbols) else str(z) for z in sorted(numbers))


def _backend(model, calc):
    """"hybrid", "cueq" or "e3nn" from the calculator's flags and the model's modules."""
    modules = [type(module).__module__ for module in model.modules()]
    if getattr(calc, "_enable_oeq", False) or any(name.startswith("openequivariance") for name in modules):
        return "hybrid"
    if getattr(calc, "_enable_cueq", False) or any(name.startswith("cuequivariance_torch") for name in modules):
        return "cueq"
    return "e3nn"


class MACEForceField:
    """Energy, forces and stress of one structure from a MACE model, without the ASE calculator.

    ``model`` is a ``fastmace.model.loading.LoadedModel`` or a single-model ``MACECalculator`` in eV
    and Angstrom, uncompiled; ``atoms`` the periodic structure. The input batch (species, head,
    charge and spin) is built once from ``atoms``; positions and edges are then replaced in place
    on every call. All options come in resolved, as the planner passes them:

    - ``rewrites``: a ``Resolution`` of ``registry`` (the planner's), or a ``RewriteConfig`` or
      ``{name: "on" | "off" | "auto"}`` that ``registry.resolve`` decides (``"auto"``: the record's
      default; an explicit "on" that cannot be honoured raises ``RewriteRefusedError``). None: no
      rewrites. The enabled records are applied to a private copy in the registry's order, each
      followed by its self-check against the source model (``check``); the structure's one-hot
      species reach them as ``Context.inputs["node_attrs"]``.
    - ``engine`` (``EngineConfig``): ``compile_mode`` and the process-wide compile settings; None
      is uncompiled. A compiled model comes from ``compile_cache`` (a
      ``fastmace.model.compile.CompiledModelCache``, owned by whoever creates force fields), so
      force fields given the same cache, source model and rewrites share it. Without a cache the
      force field compiles its own. OEq models take the force derivative outside the compiled
      graph (``training=False``); cuEquivariance and e3nn models inside it (``training=True``).
    - ``neighbours`` (``NeighbourConfig`` with ``sort_receivers`` and ``gpu_neighbour_list``
      resolved): the skin list. None: ``MACEForces``' defaults (matscipy order on the host).
      Streaming convolutions need ``sort_receivers="on"``.
    - ``polar`` (``PolarConfig``): PolarMACE's options; the force field enables the GPU-only
      long-range path on its own copy of the model (``fastmace.longrange.polar``).
    - ``compute_stress``: variable-cell runs; ``__call__`` then returns MACE's stress and
      ``set_cell`` takes the current cell.
    - ``charge``, ``spin_multiplicity``: OMOL-0's joint embedding (checked against it) or
      PolarMACE's total charge and spin (``validate_charge_spin``); a ``LoadedModel``'s values are
      the default.
    - ``float64_energy`` (``PrecisionConfig.float64_energy``): the energy is accumulated in float64
      (``fastmace.model.energy``, SPEC §7.3); forces and stress are MACE's either way. False
      reproduces 1a's energy, for A/B against the research engine only. ``atomic_energies``: the
      exact float64 E0 table; default the ``LoadedModel``'s (captured from the checkpoint before
      MACE converted the model), else the model's own table upcast.

    The streaming edge counter is this instance's neighbour-list count. Every call points the
    model's streaming interactions at it, so two force fields from one compiled model each read
    their own count; captured graphs keep the address they were captured with. The research
    engine's ``bind_streaming_count`` is gone.
    """

    def __init__(self, model, atoms, *, rewrites=None, engine=None, neighbours=None, polar=None,
                 compute_stress=False, compile_cache=None, registry=None, charge=None, spin_multiplicity=None,
                 check=True, float64_energy=True, atomic_energies=None, species=None):
        import torch

        from fastmace import _patches
        from fastmace.config import EngineConfig, NeighbourConfig, RewriteConfig
        from fastmace.longrange import polar as longrange
        from fastmace.model import compile as compiling
        from fastmace.model import energy as energy64
        from fastmace.model import loading
        from fastmace.model.capabilities import padding_is_exact
        from fastmace.neighbors.skin import SkinNeighbourList
        from fastmace.rewrites import engine as engine_rewrites
        from fastmace.rewrites import registry as reg

        loaded = model if hasattr(model, "calculator") else None
        calc = loaded.calculator if loaded is not None else model
        if len(calc.models) != 1:
            raise ValueError("Only single-model MACE calculators are supported")
        if getattr(calc, "use_compile", False):
            raise ValueError("Use an uncompiled calculator; the force field pads edges itself")
        if calc.energy_units_to_eV != 1 or calc.length_units_to_A != 1:
            raise ValueError("Expected a calculator in eV and Angstrom")
        if not all(atoms.pbc):
            raise ValueError("Periodic boundaries in all three directions are required")
        source = calc.models[0]
        self.source_model = source
        first = next(source.parameters())
        self._dtype, self._device = first.dtype, first.device
        self._compute_stress = bool(compute_stress)
        # The force field an MC energy copy follows (``follow``) and that copy's shift buffers.
        self._leader = self._unit_shifts = self._shifts64 = self._shifts = None
        # CUDA-graph edge padding relies on far edges contributing exactly zero (the engine checks).
        self._padding_exact = padding_is_exact(source)
        self.backend = loaded.backend if loaded is not None else _backend(source, calc)
        self.oeq = self.backend == "hybrid"
        engine = engine or EngineConfig(compile_mode="none")
        neighbours = neighbours or NeighbourConfig(sort_receivers="off", gpu_neighbour_list="off")
        for name in ("sort_receivers", "gpu_neighbour_list"):
            if getattr(neighbours, name) == "auto":
                raise ValueError(f"NeighbourConfig.{name} is 'auto'; the planner resolves it, so pass 'on' or 'off'")
        sort_receivers = neighbours.sort_receivers == "on"
        self.compile_mode = engine.compile_mode
        # Evaluation plans (2-T13): the compiled model sees the edge dimension as dynamic from the
        # first call, so a changed edge count does not compile again.
        self.dynamic_edges = bool(engine.dynamic_edges) and self.compile_mode != "none"
        problems = compiling.cpu_cueq_problems(source)
        if problems:
            raise loading.BackendError(
                "a cuEquivariance model on the CPU must use cuEq's naive kernels (convert it inside "
                "fastmace.model.compile.naive_cueq()): " + "; ".join(problems))
        self.is_polar = longrange.is_polar(source)
        if self.is_polar:
            longrange.require_route(self.backend, self.compile_mode)
        self.float64_energy = bool(float64_energy)
        # How the energy reaches float64 (2-T04, SPEC §18 decision 2): "patched" (the qualified
        # forwards of fastmace.model.energy / longrange.polar), "native" (a float64 model whose
        # forward is not qualified sums in float64 itself; the planner promotes the precision to
        # it), or "off". A float32 model with an unqualified forward is refused here.
        self.energy_treatment = "off"
        if self.float64_energy:
            self.energy_treatment = "patched"
            if not self.is_polar and self._dtype == torch.float64 and energy64.source_problems():
                self.energy_treatment = "native"
        # The float64 forward of a non-polar model (refused here if its source is not qualified).
        energy_forward = (energy64.float64_forward(source)
                          if self.energy_treatment == "patched" and not self.is_polar else None)

        # The input batch, built once (charge and spin as MACE's calculator reads them from atoms.info).
        if loaded is not None:
            charge = loaded.charge if charge is None else charge
            spin_multiplicity = loaded.spin_multiplicity if spin_multiplicity is None else spin_multiplicity
        atoms = atoms.copy()
        if self.is_polar:
            charge, spin_multiplicity = longrange.validate_charge_spin(
                atoms.numbers, 0 if charge is None else charge, 1 if spin_multiplicity is None else spin_multiplicity)
        elif charge is not None or spin_multiplicity is not None:
            loading.check_embedding(source, None, charge, spin_multiplicity)
        self.charge, self.spin_multiplicity = charge, spin_multiplicity
        info_keys = getattr(calc, "info_keys", {}) or {}
        if charge is not None:
            atoms.info[info_keys.get("total_charge", "charge")] = charge
        if spin_multiplicity is not None:
            atoms.info[info_keys.get("total_spin", "spin")] = spin_multiplicity
        self.numbers = atoms.numbers.copy()
        # The species superset (SPEC §8.5): every species-indexed buffer and rewrite is built for it
        # from the identity basis, never from the species present, so an atom can become any of them.
        z_table = [int(z) for z in calc.z_table.zs]
        unknown = sorted(set(_numbers(species)) - set(z_table)) if species is not None else []
        if unknown:
            raise ModelElementError(f"species {_symbols(unknown)} have no embedding in the model (it has "
                                    f"{_symbols(z_table)})", why="the model was not trained on them (SPEC §8.5)")
        wanted = set(_numbers(species) if species is not None else (int(z) for z in atoms.numbers))
        self.species_columns = tuple(sorted(z_table.index(z) for z in wanted if z in z_table))
        self.superset = tuple(z_table[column] for column in self.species_columns)
        self.check_species(atoms.numbers)
        batch = calc._atoms_to_batch(atoms).to_dict()
        node_attrs = batch["node_attrs"].to(self._device)
        if not bool((node_attrs[:, list(self.species_columns)].sum(dim=1) == 1).all()):
            raise SpeciesNotInSuperset([Reason("a one-hot species row of the structure does not sum to 1 over the "
                                               "superset's columns", rule=2, code="species-not-in-superset")])

        # Rewrites: resolved once, applied to a private copy (or inside the compiled copy).
        registry = registry or reg.default_registry()
        if rewrites is None:
            rewrites = {name: "off" for name in registry.names()}
        resolution = rewrites if isinstance(rewrites, reg.Resolution) else registry.resolve(source, rewrites)
        self.resolution = resolution
        enabled = resolution.enabled
        streaming = [name for name in enabled if name in STREAMING_RECORDS]
        if streaming and not sort_receivers:
            raise ValueError(f"{', '.join(streaming)} needs receiver-sorted edges (NeighbourConfig.sort_receivers="
                             "'on'); in any other order the streaming kernels give wrong results")
        species_dependent = any(SPECIES_DEPENDENT in registry[name].workload_constraints
                                for name in enabled if name in registry)
        specs = getattr(source, "embedding_specs", None) or {}
        joint_features = {name: batch[name].to(self._device, self._dtype) for name in specs if name in batch} or None
        context = reg.Context(config=rewrites if isinstance(rewrites, RewriteConfig) else RewriteConfig(),
                              inputs={"node_attrs": node_attrs, "species_columns": self.species_columns,
                                      "joint_features": joint_features})
        model_records = [name for name in enabled if name not in engine_rewrites.RECORDS]
        # The strain path, bound per model (2-T08b, docs/design/strain-per-model.md): the one-graph
        # function with the record on, else MACE's own, so that neither another instance nor the
        # reference calculators' process patch can change it. None (left as MACE has it) when the
        # strain sources do not hash or the forward cannot be rebound; "on" then fails loudly.
        strain = "single_graph_strain" in enabled
        if not strain and (_patches.patch_problems("single_graph_strain") or (
                not self.is_polar and engine_rewrites.strain_problems(type(source).forward))):
            strain = None
        self.applied = ()
        self.split_forces = False
        self.dynamo_reset = False
        self._polar_model = None
        self.polar_info = None
        cell = atoms.cell.array

        def prepare(private):
            if self.is_polar:
                self.polar_info = longrange.enable(private, cell, polar, variable_cell=self._compute_stress)
                self._polar_model = private
            self.applied = registry.apply(private, resolution, context, original=source, check=check)
            # After the self-checks, which compare against MACE's own energy.
            if self.energy_treatment == "patched":
                if self.is_polar:
                    longrange.set_float64_energy(private)
                else:
                    energy64.install(private)
            if strain is not None:  # last: it rebinds whichever forward is in force
                engine_rewrites.bind_strain_path(private, strain)
            return private

        if self.compile_mode != "none":
            accelerated = bool(getattr(calc, "_uses_accelerated_backend", False)) or self.backend != "e3nn"
            cache = compile_cache if compile_cache is not None else compiling.CompiledModelCache()
            # A PolarMACE copy carries its structure's k-space state, so it is never shared.
            key = None if self.is_polar else (
                id(source), self.compile_mode, self.backend, self.oeq, accelerated, tuple(enabled),
                self.species_columns if species_dependent else None, engine.first_derivative_only,
                engine.unfuse_full_addmm, self.energy_treatment, self.dynamic_edges,
                torch.are_deterministic_algorithms_enabled())  # reproducible plans compile apart (2-T16)
            entry = cache.get(key, source) if key is not None else None
            if entry is None:
                if self.is_polar:
                    # Process constraint of fastmace.longrange.polar: code compiled for a previous
                    # structure kept that call's autograd graph alive and broke the next capture.
                    torch._dynamo.reset()
                    self.dynamo_reset = True
                compiled = compiling.compile_model(source, self.compile_mode, accelerated=accelerated, oeq=self.oeq,
                                                   prepare=prepare, engine=engine)
                entry = (cache.put(key, source, compiled, self.applied) if key is not None
                         else compiling.CompiledEntry(source, compiled, self.applied))
            self.model, self.applied = entry.model, entry.applied
            # OEq custom operators are opaque to AOTAutograd: MACE's OEq compile path
            # differentiates outside the compiled model.
            self.training = not self.oeq
            # Compiled OEq models are called with forces and stress off, and MACE's gradient step runs after the
            # call, outside the compiled frame (setup-across-compositions F2; _split_outputs). With it inside, Dynamo
            # traced the frame up to torch.autograd.grad, restarted at the graph break and traced it again: 2.2-3.5 s
            # per build. ScaleShiftMACE differentiates its interaction energy, which its output dict carries.
            self.split_forces = (self.oeq and not self.is_polar
                                 and energy64.class_name(source) == "ScaleShiftMACE")
        elif model_records or self.is_polar:
            self.model = prepare(compiling.private_copy(source))
            self.training = False
        else:
            self.model = source
            if enabled:  # process-wide records only (fastmace.rewrites.engine): the model is not touched
                self.applied = registry.apply(source, resolution, context, original=source, check=check)
            self.training = False

        self._streaming = [module for module in self.model.modules()
                           if getattr(module, "streaming_kernels", None) is not None]
        if self._streaming and not sort_receivers:
            raise ValueError("the model has streaming convolutions, which need receiver-sorted edges "
                             "(NeighbourConfig.sort_receivers='on')")

        self.data = {key: (value.to(self._device, self._dtype) if value.is_floating_point() else value.to(self._device))
                     if torch.is_tensor(value) else value for key, value in batch.items()}
        # The calculator's own model (eager, no private copy) is never changed: its float64 forward
        # is called as a function.
        self._model_call = self.model
        if self.model is source:
            forward = energy_forward
            if strain is not None:
                forward = engine_rewrites.strain_forward(forward or type(source).forward, strain)
            if forward is not None:
                self._model_call = functools.partial(forward, self.model)
            self.strain_path = None if strain is None else ("one_graph" if strain else "mace")
        else:
            self.strain_path = engine_rewrites.strain_path(getattr(self.model, "_orig_mod", self.model))
        self.atomic_energies = None
        if self.energy_treatment == "patched":
            if atomic_energies is None:
                atomic_energies = getattr(loaded, "atomic_energies", None)
            if atomic_energies is None:
                atomic_energies = energy64.model_atomic_energies(source)
            atomic_energies = torch.as_tensor(atomic_energies, dtype=torch.float64).cpu().clone()
            expected = tuple(source.atomic_energies_fn.atomic_energies.shape)
            if tuple(atomic_energies.shape) != expected:
                raise ValueError(f"atomic_energies has shape {tuple(atomic_energies.shape)}, the model's E0 table "
                                 f"{expected}")
            self.atomic_energies = atomic_energies
            head = int(batch["head"].reshape(-1)[0]) if "head" in batch else 0
            self.data[energy64.NODE_E0_KEY] = energy64.node_e0_float64(atomic_energies, node_attrs, head)
            self._e0_row = torch.atleast_2d(atomic_energies)[head].to(self._device)
        self._setup_species(atoms, z_table)
        segmented = any(name in SEGMENTED_RECORDS for name in enabled)
        self.neighbours = SkinNeighbourList(
            cell, calc.r_max, self._device, self._dtype, neighbours,
            sender_groups=self._sender_groups if segmented and sort_receivers else None,
            group_count=len(self.superset), variable_cell=self._compute_stress)
        self._setup_tables()
        if self._compute_stress:
            # Fixed-address cell that MACE's stress path reads (set_cell refills it).
            self.data["cell"] = torch.tensor(cell, dtype=self._dtype, device=self._device)
            # The strain the stress differentiates by, as an input leaf. MACE otherwise makes it
            # inside the model; compiled, it is then a sibling output of the energy instead of an
            # input, its gradient is lost and the stress comes out as exactly zero.
            self.data["displacement"] = torch.zeros((1, 3, 3), dtype=self._dtype, device=self._device,
                                                    requires_grad=True)
        # Leaf tensor that the model differentiates; refilled before every call.
        self.positions = torch.zeros((len(atoms), 3), dtype=self._dtype, device=self._device, requires_grad=True)
        self.data["positions"] = self.positions
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=self._device)
        self.rebuild(positions)
        self.compact(positions)
        self.use_edges(int(self.neighbours.count))
        self._bind_streaming_count()

    # ------------------------------------------------------------------ properties

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def compute_stress(self):
        return self._compute_stress

    @property
    def padding_exact(self):
        return self._padding_exact

    @property
    def edge_count(self):
        return self.neighbours.count

    @property
    def edge_capacity(self):
        return self.neighbours.capacity

    # ------------------------------------------------------------------ species superset (SPEC §8.5)

    def _setup_species(self, atoms, z_table):
        """The species and charge buffers an event writes, and the derived buffers built from them.

        ``atomic_numbers`` (int64, (N,)) and ``charges`` (float64, (N,), the per-atom formal charges)
        are what a transaction writes (3-T03); every other species- or charge-indexed buffer is
        derived from them by a refresh function (``register_derived``). The lookup tables cover every
        atomic number, so a refresh is a few device gathers at fixed addresses.
        """
        import torch

        device, n = self._device, len(atoms)
        size = max(119, max(z_table) + 1)
        column_of = torch.zeros(size, dtype=torch.int64)
        group_of = torch.zeros(size, dtype=torch.int64)
        for group, column in enumerate(self.species_columns):
            column_of[z_table[column]] = column
            group_of[z_table[column]] = group
        self._column_of, self._group_of = column_of.to(device), group_of.to(device)
        self.atomic_numbers = torch.as_tensor(atoms.numbers, dtype=torch.int64).to(device)
        self.charges = torch.as_tensor(atoms.get_initial_charges(), dtype=torch.float64).to(device)
        # The atoms' node_attrs columns, as every species refresh leaves them (a fresh construction equals a
        # refreshed one in every tensor, 3-T03's registry check).
        self._species_index = torch.index_select(self._column_of, 0, self.atomic_numbers)
        self._ones = torch.ones((n, 1), dtype=self.data["node_attrs"].dtype, device=device)
        self._sender_groups = torch.index_select(self._group_of, 0, self.atomic_numbers)
        # Total charge = its value at construction + the change of the summed per-atom charges, so
        # that it is exactly the given total until a charge changes.
        # Total charge = offset + sum of the per-atom charges, offset = the given total - their sum at
        # construction: a transmutation changes the per-atom charges only, and the total follows on the
        # device. Integer charges keep it exact; a fresh construction has the same offset and scratch.
        self._has_total = {name: torch.is_tensor(self.data.get(name)) for name in ("total_charge", "total_spin")}
        self._charge_sum = torch.zeros(1, dtype=torch.float64, device=device)  # scratch: the total charge
        self._charge_offset = torch.zeros(1, dtype=torch.float64, device=device)
        self._spin_base = None
        if self._has_total["total_charge"]:
            self._charge_offset.copy_(self.data["total_charge"].detach().to(torch.float64).reshape(1))
            self._charge_offset.sub_(self.charges.sum(dim=0, keepdim=True))
            torch.sum(self.charges, dim=0, keepdim=True, out=self._charge_sum)
            self._charge_sum.add_(self._charge_offset)
        if self._has_total["total_spin"]:
            self._spin_base = self.data["total_spin"].detach().clone()
        self._tables = []  # (FirstInteractionReassociation, its [S, C] table, its refresh)

    def _setup_tables(self):
        """Each reassociated interaction's sender-feature table, from the identity basis of the superset."""
        import torch

        from fastmace.rewrites.mh1 import modules as m

        model = getattr(self.model, "_orig_mod", self.model)
        reassociated = [(index, interaction) for index, interaction in enumerate(getattr(model, "interactions", ()))
                        if getattr(interaction, "first_reassociation", None) is not None]
        if not reassociated:
            return
        parameter = next(model.parameters())
        n = len(self.numbers)
        attrs = m.identity_attrs(self.species_columns, self.data["node_attrs"].shape[1], n, parameter.dtype,
                                 parameter.device)
        joint = None
        if hasattr(model, "joint_embedding"):
            batch = torch.zeros(attrs.shape[0], dtype=torch.int64, device=parameter.device)
            joint = (batch, {name: self.data[name] for name in model.embedding_specs})
        count = len(self.species_columns)
        for index, interaction in reassociated:
            table = m.identity_species_feats(model, interaction, attrs, count, joint).detach().clone()

            def refresh(interaction=interaction, table=table):
                table.copy_(m.identity_species_feats(model, interaction, attrs, count, joint))

            self._tables.append((index, interaction.first_reassociation, table, refresh))
        self._bind_streaming_count()

    def check_species(self, numbers):
        """Raise ``SpeciesNotInSuperset`` unless every atomic number of ``numbers`` ((N,)) is in the superset.

        Host only, nothing is written: a transaction calls it before it applies a species change
        (SPEC §8.5 "membership is checked explicitly at setup and on every host-side mutation").
        """
        import numpy as np

        numbers = np.asarray(numbers)
        if numbers.shape != (len(self.numbers),):
            raise ValueError(f"species of {len(self.numbers)} atoms expected, got shape {numbers.shape}")
        outside = sorted({int(z) for z in numbers} - set(self.superset))
        if outside:
            raise SpeciesNotInSuperset([Reason(
                f"species {_symbols(outside)} are not in the ForceField's declared superset ({_symbols(self.superset)})"
                " (SPEC §8.5, §9.3 rule 2)", rule=2, code="species-not-in-superset")])

    def _refresh_node_attrs(self):
        import torch

        torch.index_select(self._column_of, 0, self.atomic_numbers, out=self._species_index)
        node_attrs = self.data["node_attrs"]
        node_attrs.zero_()
        node_attrs.scatter_(1, self._species_index.unsqueeze(1), self._ones)

    def _refresh_node_e0(self):
        import torch

        from fastmace.model import energy as energy64

        # One table entry per atom: exactly the one-hot product of node_e0_float64.
        torch.index_select(self._column_of, 0, self.atomic_numbers, out=self._species_index)
        torch.index_select(self._e0_row, 0, self._species_index, out=self.data[energy64.NODE_E0_KEY])

    def _refresh_sender_groups(self):
        import torch

        torch.index_select(self._group_of, 0, self.atomic_numbers, out=self._sender_groups)

    def _refresh_total_charge(self):
        import torch

        torch.sum(self.charges, dim=0, keepdim=True, out=self._charge_sum)
        self._charge_sum.add_(self._charge_offset)
        self.data["total_charge"].copy_(self._charge_sum.reshape(self.data["total_charge"].shape))

    def _refresh_total_spin(self):
        # The spin multiplicity stays: a transmutation is allowed only with ΔZ − Δq even (SPEC §10.4).
        self.data["total_spin"].copy_(self._spin_base)

    def _derived(self):
        """(name, buffer, refresh, depends_on) of every derived buffer, dependencies first."""
        from fastmace.model import energy as energy64

        out = [("node_attrs", self.data["node_attrs"], self._refresh_node_attrs, ("species",))]
        if energy64.NODE_E0_KEY in self.data:
            out.append(("node_e0", self.data[energy64.NODE_E0_KEY], self._refresh_node_e0, ("species",)))
        # Registered whether or not this list sorts by sender group, so that it never goes stale.
        out.append(("sender_groups", self._sender_groups, self._refresh_sender_groups, ("species",)))
        if self._has_total["total_charge"]:
            out.append(("total_charge", self.data["total_charge"], self._refresh_total_charge, ("charges",)))
        if self._has_total["total_spin"]:
            out.append(("total_spin", self.data["total_spin"], self._refresh_total_spin, ("charges",)))
        out += [(f"species_feats.interactions.{index}", table, refresh, ("charges",))
                for index, _, table, refresh in self._tables]
        return out

    def derived_buffers(self):
        """The derived device buffers by name (without the ``force_field.`` prefix of ``register_derived``)."""
        return {name: buffer for name, buffer, _, _ in self._derived()}

    def register_derived(self, registry, state=None, prefix="force_field"):
        """Register every derived buffer with its refresh function in ``registry`` (a
        ``fastmace.events.DerivedStateRegistry``) as ``<prefix>.<name>``.

        The refresh functions recompute their buffer in place on the device from ``atomic_numbers``
        and ``charges`` (fixed addresses, no host synchronisation; CUDA-graph capturable):
        ``node_attrs`` (one-hot), ``node_e0`` (float64 per-atom E0), ``sender_groups`` (the
        segmented first sum's sender species, a fixed count of superset-size groups), the total
        charge and spin, and each reassociated interaction's sender-feature table (identity basis;
        OMOL-0's joint embedding reads the total charge and spin, so it follows them). The
        neighbour list reads the sender groups at its next build.

        ``state`` (a ``SimulationState``; the engine passes it, 3-T03): the refreshes then read the
        committed ``state.species`` and, when the model takes them, ``state.charges`` instead of the
        force field's own buffers (``use_sources``), so a transaction writes the state only.
        """
        if state is not None:
            self.use_sources(state.species, getattr(state, "charges", None))
        return [registry.register(f"{prefix}.{name}", buffer, refresh, depends)
                for name, buffer, refresh, depends in self._derived()]

    def use_sources(self, species, charges=None):
        """Read the species ((N,) int64 atomic numbers) and per-atom charges ((N,) float64) from these
        device tensors from now on (aliases, not copies). Their values must be this force field's
        current ones."""
        import torch

        if species.shape != self.atomic_numbers.shape or species.dtype != torch.int64:
            raise ValueError(f"species source must be int64 {tuple(self.atomic_numbers.shape)}, got "
                             f"{species.dtype} {tuple(species.shape)}")
        if not torch.equal(species.to(self.atomic_numbers.device), self.atomic_numbers):
            raise ValueError("the species source differs from the force field's current species")
        self.atomic_numbers = species
        if charges is not None:
            if charges.shape != self.charges.shape:
                raise ValueError(f"charges source must have shape {tuple(self.charges.shape)}, got "
                                 f"{tuple(charges.shape)}")
            if not torch.equal(charges.to(self.charges), self.charges):
                raise ValueError("the charges source differs from the force field's current charges")
            self.charges = charges

    def set_total_charge(self, total_charge):
        """A host event's new total charge (3-T03's ``set_charges(total_charge=...)``): the device baseline
        offset is set so that the next ``total_charge`` refresh gives ``total_charge`` for the current
        per-atom charges (exactly, for integer charges). Host side, between replays; transmutations
        change the per-atom charges only, and their refresh follows on the device."""
        if not self._has_total["total_charge"]:
            raise ValueError("this model takes no total charge")
        self._charge_offset.fill_(float(total_charge))
        self._charge_offset.sub_(self.charges.sum(dim=0, keepdim=True))

    # ------------------------------------------------------------------ graph level

    def _bind_streaming_count(self):
        """Point the streaming interactions at this force field's edge count, and the reassociated
        first interaction at its sender-feature table (host-side pointer swaps).

        The model may be shared with other force fields (one compiled model per cache key), so each
        call binds its own counter and table before the model runs; a captured graph keeps the
        address it was captured with. Nothing outside the force field ever rebinds.
        """
        # A copy that follows another force field (``follow``) reads that force field's edges and count.
        count = (self.neighbours if self._leader is None else self._leader.neighbours).count
        for module in self._streaming:
            if module._buffers.get("streaming_count") is not count:
                module.register_buffer("streaming_count", count, persistent=False)
        for _, reassociation, table, _ in self._tables:
            reassociation.bind(table)

    def __call__(self, positions):
        """Energy [1], forces [N, 3] (and stress [3, 3]) at float64 ``positions`` on the current edges.

        The stress is MACE's (ASE convention, eV/Å^3, no kinetic part): the energy derivative with
        respect to a symmetric strain of positions and cell, taken in the same backward pass as
        the forces.
        """
        out = self._outputs(positions)
        stress = out["stress"].reshape(3, 3) if self._compute_stress else None
        return ForceOutput(out["energy"], out["forces"], stress)

    def _outputs(self, positions):
        """The model's output dict at float64 ``positions`` on the current edges."""
        import torch

        self._bind_streaming_count()
        with torch.no_grad():
            self.positions.copy_(positions)
        if self.split_forces:
            return self._split_outputs(self._compute_stress)
        if not self._compute_stress:
            return self._model_call(self.data, training=self.training, compute_force=True, compute_stress=False)
        # MACE's stress path replaces "positions" and "shifts" in the dict it gets (strained
        # copies); a shallow copy keeps ours bound to the buffers.
        return self._model_call(dict(self.data), training=self.training, compute_force=True, compute_stress=True)

    def _split_outputs(self, stress):
        """The force call with MACE's gradient step after the compiled model instead of inside it (F2).

        The compiled model runs with forces and stress off, so its frame holds no ``torch.autograd.grad``: Dynamo
        traces it once, without the graph break and the restart. Then MACE's own ``get_outputs`` (with the
        process's patched force functions) differentiates the scalar MACE differentiates (``ScaleShiftMACE``: the
        interaction energy) with respect to the same input leaves, the positions and, with stress, the strain.
        MACE's OEq compile path already ran this gradient outside the compiled graph, after the break.
        """
        from mace.modules import utils

        # MACE's stress path replaces "positions" and "shifts" in the dict it gets (strained copies).
        data = dict(self.data) if stress else self.data
        out = self._model_call(data, training=self.training, compute_force=False, compute_stress=False,
                               compute_displacement=stress)
        forces, virials, stress_out, _, _ = utils.get_outputs(
            energy=out["interaction_energy"], positions=self.positions,
            displacement=self.data["displacement"] if stress else None, cell=self.data["cell"],
            training=self.training, compute_force=True, compute_virials=False, compute_stress=stress)
        return {**out, "forces": forces, "virials": virials, "stress": stress_out}

    def energy_only(self, positions):
        """The float64 energy [1] at float64 ``positions`` on the current edges: a forward pass only (SPEC §8.1).

        Under ``torch.no_grad()``, with forces, stress and ``training`` off, so no autograd graph is
        built and the output holds no ``grad_fn``. Calling the force path with forces switched off
        while the positions require gradients would run AOTAutograd's joint graph, keep activations
        nobody reads and leave ``grad_fn`` references that break later captures (the failure noted
        in the research engine's ``polar_gpu._Call``). A compiled model compiles this call as its
        own variant (grad mode and ``training`` are guarded). The sum is the same float64 sum as the
        force path's (SPEC §7.3). A copy that follows another force field (``follow``) first reads
        that force field's compacted edges and cell.
        """
        import contextlib

        import torch

        from fastmace.rewrites import engine as engine_rewrites

        self._bind_streaming_count()
        # A copy in another precision runs under its own default dtype: MACE (and the capture-safe
        # mask_head) make tensors in torch's process-wide default dtype, which the force field it
        # follows set when its model was loaded (host side only; captured graphs do not read it).
        scope = (engine_rewrites._default_dtype(self._dtype)
                 if self._leader is not None and torch.get_default_dtype() != self._dtype else contextlib.nullcontext())
        with torch.no_grad(), scope:
            if self._leader is not None:
                self._read_leader()
            self.positions.copy_(positions)
            out = self._model_call(self.data, training=False, compute_force=False, compute_stress=False)
            energy = out["energy"]
            return energy if energy.dtype == torch.float64 else energy.to(torch.float64)

    # ------------------------------------------------------------------ a copy in another precision (3-T07)

    def follow(self, leader):
        """Evaluate on ``leader``'s compacted edges and cell from now on: the MC energy copy in another
        precision (SPEC §10.4 "Energy precision"; ``fastmace.mc.evaluator.float64_copy``).

        ``leader`` is the engine's force field, of the same structure and superset. This copy then reads
        the leader's edge index as a view, rebuilds the image shifts in its own precision from the
        leader's integer unit shifts and float64 cell (exactly as a force field of this precision
        computes them), and shares the leader's species, charge and total-charge sources, so the
        derived-state refreshes of both force fields follow one transaction. Its own neighbour list
        is not used again. ``use_edges`` takes the leader's padded edge count.
        """
        import torch

        if self._compute_stress != leader.compute_stress:
            raise ValueError("the copy and the force field it follows must both be variable-cell or both fixed-cell")
        if tuple(self.superset) != tuple(leader.superset) or len(self.numbers) != len(leader.numbers):
            raise ValueError(f"the copy has species {_symbols(self.superset)} and {len(self.numbers)} atoms, the "
                             f"force field it follows {_symbols(leader.superset)} and {len(leader.numbers)}: build it "
                             "from the same structure and superset")
        if self.is_polar != leader.is_polar or self.data["node_attrs"].shape != leader.data["node_attrs"].shape:
            raise ValueError("the copy and the force field it follows must be the same model (species columns "
                             f"{tuple(self.data['node_attrs'].shape)} and {tuple(leader.data['node_attrs'].shape)})")
        self.use_sources(leader.atomic_numbers, leader.charges)
        self._charge_offset = leader._charge_offset  # a host total-charge event reaches both
        self._leader = leader
        if not self._compute_stress:
            self._leader_cell = torch.as_tensor(leader.neighbours.cell, dtype=torch.float64).to(self._device)
        self.use_edges(int(leader.data["edge_index"].shape[1]))

    def _read_leader(self):
        """Copy the leader's unit shifts and cell into this copy's fixed-address inputs (capturable)."""
        import torch

        leader = self._leader.neighbours
        self._unit_shifts.copy_(leader.unit_shifts)
        if self._compute_stress:
            self.set_cell(leader.cell_t)
            cell = leader.cell_t
        else:
            cell = self._leader_cell
        from fastmace.neighbors.skin import times_3x3

        self._shifts64.copy_(times_3x3(self._unit_shifts.to(torch.float64), cell))  # no FP64 GEMM in the step
        self._shifts.copy_(self._shifts64)

    def set_cell(self, cell):
        """compute_stress: the current float64 cell for the neighbour list and the model inputs; graph safe."""
        if not self._compute_stress:
            raise RuntimeError("set_cell needs a force field built with compute_stress=True")
        self.neighbours.set_cell(cell)
        self._write_cell_inputs(cell)

    def reset_cell(self, cell) -> None:
        """A new cell between replays, fixed or variable cell (4-T05's ``Simulation.update(cell=)``): the
        neighbour list's host cell (the caller rebuilds it) and the model's cell inputs in place, and a follower
        copy's leader cell. The captured graphs read these at fixed addresses, so they stay valid."""
        import torch

        cell = torch.as_tensor(cell, dtype=torch.float64).to(self._device)
        self.neighbours.reset_cell(cell.cpu().numpy())
        if getattr(self, "_leader_cell", None) is not None:
            self._leader_cell.copy_(cell)
        self._write_cell_inputs(cell)

    def _write_cell_inputs(self, cell):
        """The model inputs the cell feeds (cell, volume, reciprocal cell), in place; graph safe."""
        import torch

        with torch.no_grad():
            if "cell" in self.data:
                self.data["cell"].copy_(cell.reshape(self.data["cell"].shape))
            # PolarMACE's k-space sums also read the batch's volume and reciprocal cell
            # (mace.data.atomic_data: det(cell), 2 pi inv(cell^T)); other models ignore them.
            # Cross products instead of linalg.det/inv, which check for errors on the host.
            a, b, c = cell[0], cell[1], cell[2]
            crosses = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a), torch.linalg.cross(a, b)))
            volume = torch.dot(a, crosses[0])
            if "volume" in self.data:
                self.data["volume"].copy_(volume.reshape(self.data["volume"].shape))
            if "rcell" in self.data:
                self.data["rcell"].copy_((2 * torch.pi * crosses / volume).reshape(self.data["rcell"].shape))

    def needs_rebuild(self, positions, fraction=1.0):
        return self.neighbours.needs_rebuild(positions, fraction)

    def compact(self, positions):
        self.neighbours.compact(positions)

    def rebuild(self, positions):
        """Rebuild the candidate pairs from NumPy or float64 device positions; True if the capacity grew.

        On a CUDA device with ``gpu_neighbour_list`` the pairs are found on the device, otherwise
        with matscipy on the host. Call ``use_edges`` afterwards: grown buffers have moved.
        """
        return self.neighbours.build(positions)

    def use_edges(self, num_edges):
        """Let the model read the first ``num_edges`` compacted edges (views, no copy).

        A copy that follows another force field (``follow``) reads that force field's edge index and
        its own shift buffers, sized to that force field's capacity (moved when it grows).
        """
        if self._leader is not None:
            import torch

            leader = self._leader.neighbours
            capacity = leader.capacity
            if self._unit_shifts is None or self._unit_shifts.shape[0] != capacity:
                self._unit_shifts = torch.zeros((capacity, 3), dtype=leader.unit_shifts.dtype, device=self._device)
                self._shifts64 = torch.zeros((capacity, 3), dtype=torch.float64, device=self._device)
                self._shifts = torch.zeros((capacity, 3), dtype=self._dtype, device=self._device)
            self.data["edge_index"] = leader.edge_index[:, :num_edges]
            self.data["shifts"], self.data["unit_shifts"] = self._shifts[:num_edges], self._unit_shifts[:num_edges]
        else:
            self.data["edge_index"], self.data["shifts"], self.data["unit_shifts"] = self.neighbours.edges(num_edges)
        if self.dynamic_edges:
            from fastmace.model import compile as compiling

            compiling.mark_edges_dynamic(self.data)
            compiling.mark_atoms_dynamic(self.data, self.positions.shape[0])

    # ------------------------------------------------------------------ one-off evaluation

    def evaluate(self, atoms, stress=False):
        """Energy (eV), forces (eV/Å, float64) and the 3x3 stress (eV/Å^3) or None, of ``atoms``.

        ``atoms`` must have this force field's species in the same order (the input batch is fixed
        at construction); a fixed-cell force field takes only its own cell. Synchronises and
        rebuilds the neighbour list: not for the MD loop.
        """
        import torch

        if stress and not self._compute_stress:
            raise ValueError("stress=True needs a force field built with compute_stress=True")
        out = self(self._prepare_one_off(atoms))
        energy = float(out.energy.detach().reshape(-1)[0])
        forces = out.forces.detach().to(torch.float64).cpu().numpy()
        values = out.stress.detach().to(torch.float64).cpu().numpy() if stress else None
        return energy, forces, values

    def per_atom_energies(self, atoms):
        """Per-atom float64 energies of ``atoms`` and what they omit (``PerAtomEnergies``; SPEC §7.3).

        Same conditions as ``evaluate`` (synchronises, rebuilds the list); needs ``float64_energy``.
        """
        from fastmace.model import energy as energy64

        if not self.float64_energy:
            raise ValueError("per-atom energies need a force field built with float64_energy=True")
        if self.energy_treatment != "patched":
            raise ValueError("per-atom energies need the qualified float64 energy forward (MACE 0.3.16); this force "
                             "field sums a float64 model natively (SPEC §18 decision 2)")
        out = self._outputs(self._prepare_one_off(atoms))
        return PerAtomEnergies(out["node_energy_float64"].detach().cpu().numpy().copy(),
                               float(out["energy_unresolved_float64"].detach().reshape(-1)[0]),
                               energy64.terms(self.source_model))

    def _prepare_one_off(self, atoms):
        """Check ``atoms``, set its cell, rebuild and compact the list; its float64 device positions."""
        import numpy as np
        import torch

        current = self.atomic_numbers.cpu().numpy()  # after any species event
        if len(atoms) != len(current) or not np.array_equal(atoms.numbers, current):
            raise ValueError("evaluate needs the force field's current species in the same order (the input batch "
                             "changes only through events)")
        if not all(atoms.pbc):
            raise ValueError("Periodic boundaries in all three directions are required")
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if self._compute_stress:
            self.set_cell(torch.tensor(cell, dtype=torch.float64, device=self._device))
        elif not np.allclose(cell, self.neighbours.cell, rtol=0, atol=1e-12):
            raise ValueError("a fixed-cell force field evaluates its own cell only; build it with "
                             "compute_stress=True for other cells")
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=self._device)
        self.rebuild(positions)
        self.compact(positions)
        self.use_edges(int(self.neighbours.count))
        return positions

    # ------------------------------------------------------------------ state and provenance

    def _polar_state(self):
        if self._polar_model is None:
            return None
        from fastmace.longrange import polar

        return polar.get_state(self._polar_model)

    def reset_polar(self, cell) -> tuple[bool, bool]:
        """Canonical restart point (1b-T14, SPEC §10.6): the polar state a fresh build at ``cell`` has.

        The k-vector set is rebuilt from the (float64 host) ``cell`` when it is not the one it was
        built for, which happens only for a variable cell; it is refilled in place when its shapes
        stay. The sticky overflow flag is cleared (registry: ``reset``). Returns ``(moved,
        overflowed)``: whether the polar buffers moved (captured graphs are stale) and whether the flag
        was set. ``(False, False)`` without a polar state.
        """
        import numpy as np
        import torch

        from fastmace.longrange import polar

        state = self._polar_state()
        if state is None:
            return False, False
        overflowed = bool(state.overflow.item())
        state.overflow.zero_()
        cell = np.asarray(cell, dtype=np.float64)
        if np.array_equal(state.cell, cell):
            return False, overflowed
        fresh = polar.PolarState(self._polar_model, cell, state.headroom)
        old, new = state.buffers(), fresh.buffers()
        if any(old[name].shape != new[name].shape for name in old):
            polar.set_state(self._polar_model, fresh)
            self.polar_info = fresh.info()
            return True, overflowed
        with torch.no_grad():
            for name in old:
                old[name].copy_(new[name])
        state.bounds, state.cell, state.superset_size = fresh.bounds, fresh.cell, fresh.superset_size
        self.polar_info = state.info()
        return False, overflowed

    def buffers(self):
        """Persistent device buffers the graph-level methods write (and the polar state they read), by name.

        The engine registers them as ``force_field.<name>`` (``fastmace.engine.registry.POLICIES``):
        the neighbour list, the positions leaf the model differentiates (refilled before every
        call), the float64 per-atom E0 (from the species), the cell inputs and the polar state.
        """
        from fastmace.model import energy as energy64

        out = {f"neighbours.{name}": tensor for name, tensor in self.neighbours.buffers().items()}
        out["positions"] = self.positions
        if energy64.NODE_E0_KEY in self.data:
            out["node_e0"] = self.data[energy64.NODE_E0_KEY]
        if self._compute_stress:
            for name in ("cell", "volume", "rcell"):
                if name in self.data:
                    out[name] = self.data[name]
        state = self._polar_state()
        if state is not None:
            out.update({f"polar.{name}": tensor for name, tensor in state.buffers().items()})
        return out

    def selfcheck_records(self):
        """The per-record self-check results of this force field's rewrites (SPEC §6.4), as plain mappings."""
        from fastmace.rewrites import registry as reg

        return reg.selfcheck_records(self.applied)

    def info(self):
        """Counters and choices for provenance."""
        state = self._polar_state()
        return {
            "backend": self.backend, "dtype": str(self._dtype).removeprefix("torch."), "device": str(self._device),
            "compile_mode": self.compile_mode, "training": self.training, "compute_stress": self._compute_stress,
            "padding_exact": self._padding_exact, "rewrites": [entry.name for entry in self.applied],
            "decisions": self.resolution.reasons(),
            "selfchecks": {entry.name: None if entry.result is None else entry.result.max_relative_error
                           for entry in self.applied},
            "streaming_interactions": len(self._streaming), "sort_receivers": self.neighbours.sort_receivers,
            "gpu_neighbour_list": self.neighbours.gpu_build, "neighbour_rebuilds": self.neighbours.rebuilds,
            "edge_capacity": self.neighbours.capacity, "candidates": self.neighbours.num_candidates,
            "species_columns": list(self.species_columns), "superset": list(self.superset), "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity, "dynamo_reset": self.dynamo_reset,
            "polar": None if state is None else state.info(), "float64_energy": self.float64_energy,
            "energy_treatment": self.energy_treatment, "strain_path": self.strain_path,
            "follows": None if self._leader is None else str(self._leader.dtype).removeprefix("torch."),
        }
