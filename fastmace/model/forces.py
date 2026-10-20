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
                 check=True, float64_energy=True, atomic_energies=None):
        import torch

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
        batch = calc._atoms_to_batch(atoms).to_dict()
        node_attrs = batch["node_attrs"].to(self._device)
        self.species_columns = tuple(torch.nonzero(node_attrs.sum(dim=0) > 0).flatten().tolist())

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
        context = reg.Context(config=rewrites if isinstance(rewrites, RewriteConfig) else RewriteConfig(),
                              inputs={"node_attrs": node_attrs})
        model_records = [name for name in enabled if name not in engine_rewrites.RECORDS]
        self.applied = ()
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
        if energy_forward is not None and self.model is source:
            self._model_call = functools.partial(energy_forward, self.model)
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
        segmented = any(name in SEGMENTED_RECORDS for name in enabled)
        self.neighbours = SkinNeighbourList(
            cell, calc.r_max, self._device, self._dtype, neighbours,
            sender_groups=(node_attrs[:, list(self.species_columns)].argmax(dim=1).cpu().numpy()
                           if segmented and sort_receivers else None),
            variable_cell=self._compute_stress)
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

    # ------------------------------------------------------------------ graph level

    def _bind_streaming_count(self):
        """Point the streaming interactions at this force field's edge count (a host-side pointer swap).

        The model may be shared with other force fields (one compiled model per cache key), so each
        call binds its own counter before the model runs; a captured graph keeps the address it was
        captured with. Nothing outside the force field ever rebinds.
        """
        count = self.neighbours.count
        for module in self._streaming:
            if module._buffers.get("streaming_count") is not count:
                module.register_buffer("streaming_count", count, persistent=False)

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
        if not self._compute_stress:
            return self._model_call(self.data, training=self.training, compute_force=True, compute_stress=False)
        # MACE's stress path replaces "positions" and "shifts" in the dict it gets (strained
        # copies); a shallow copy keeps ours bound to the buffers.
        return self._model_call(dict(self.data), training=self.training, compute_force=True, compute_stress=True)

    def set_cell(self, cell):
        """compute_stress: the current float64 cell for the neighbour list and the model inputs; graph safe."""
        import torch

        if not self._compute_stress:
            raise RuntimeError("set_cell needs a force field built with compute_stress=True")
        self.neighbours.set_cell(cell)
        with torch.no_grad():
            self.data["cell"].copy_(cell)
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
        """Let the model read the first ``num_edges`` compacted edges (views, no copy)."""
        self.data["edge_index"], self.data["shifts"], self.data["unit_shifts"] = self.neighbours.edges(num_edges)
        if self.dynamic_edges:
            from fastmace.model import compile as compiling

            compiling.mark_edges_dynamic(self.data)

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

        if len(atoms) != len(self.numbers) or not np.array_equal(atoms.numbers, self.numbers):
            raise ValueError("evaluate needs the force field's species in the same order (the input batch is fixed)")
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
            "species_columns": list(self.species_columns), "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity, "dynamo_reset": self.dynamo_reset,
            "polar": None if state is None else state.info(), "float64_energy": self.float64_energy,
            "energy_treatment": self.energy_treatment,
        }
