"""The CUDA-graph engine: captured stages, pipelined and speculative steps, asynchronous output (SPEC §8.1).

Copied from the capture, replay and pipelining half of the research engine's ``GPULangevin`` and
``GPULangevinBAOAB`` (``langevin/gpu_langevin.py``, ``langevin/gpu_npt.py``) and refactored along
the integrator–engine cut of docs/design/interfaces.md: the integrator's half steps and the force
field's graph-level calls are issued in the baseline's order, so every graph records the
baseline's kernels.

One step is split into stages:

- ``A``: ``integrator.first_half`` (kick, drift; BAOAB: barostat and cell), then, for a variable
  cell, ``force_field.set_cell``, and the neighbour check ``force_field.needs_rebuild``;
- ``C``: ``force_field.compact``, the edges within the cutoff moved to the front;
- ``B``: the force call on a padded prefix of the compacted edges, its results copied into
  ``state.forces``, ``energy`` (and ``stress``), then ``integrator.second_half``.

A and C run as one graph (``"AC"``), after which the host reads the rebuild flag and the edge count
with one transfer; on the rare rebuild steps the force field rebuilds its candidates and C runs
again. B is captured once per padded edge size (``("B", edges)``): the edge count rounded up to the
edge bucket, about ``NeighbourConfig.edge_bucket_fraction`` of the initial count; the padded edges
lie beyond the cutoff and contribute exactly zero (``ForceField.padding_exact``).

**Pipelined** (``EngineConfig.pipelined="on"``, graph ``"ACP"``): B is launched right after AC with
the padded size of the last validated count, before the host has read the flags; the step is
validated while B runs, one step later. A step whose neighbour list needed a rebuild, or whose
count outgrew that size, restores the integrator's mid-step tensors and redoes B (after a rebuild
and C when needed); after a count redo the larger size is kept until the next rebuild. The list is
rebuilt early once an atom has moved ``pipeline_rebuild_fraction`` of the hard limit, so rebuild
redos stay rare.

**Speculative** (``speculative_graph``, graph ``("ACB", edges, slot)``, NVT only): AC and B in one
graph, validated when the next step is launched (``SPECULATIVE_SLOTS`` in flight), with the same
restore and redo of B.

**Asynchronous output**: ``snapshot`` validates the step in flight, queues pinned copies of
``integrator.output_values`` behind it and records an event (``fastmace.io.trajectory``).

Capture: three eager warm-ups on a side stream (inside the graph pool after the first capture of a
stage kind, ``warmup_in_graph_pool``), then the capture. Warm-ups snapshot and restore every
captured buffer of the runtime-state registry (``registry``: the state, ``integrator.buffers``,
``force_field.buffers`` and the engine's own flags, host mirrors and mid-step copies, sticky flags
included), which is more than the baseline's ``_graph_state``: SPEC §8.1 lists what that missed.

Seams for 1b and phase 2 (SPEC §8.1, kept at today's behaviour here): every re-capture goes through
``_clear_graphs`` and ``_capture`` (the re-capture policy and its log); every host decision on a
step's flags is in ``_step_checked``, ``_validate_pipelined`` and ``_validate_oldest_speculative``
(overflow before commit); the registry (``fastmace.engine.registry``) holds the policies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from fastmace.neighbors.buckets import edge_bucket, round_up_edges

from . import constraints as process
from .base import Engine, Snapshot, _record_production_engine
from .registry import Registry

# Speculative combined-graph steps kept in flight: one, validated when the next step is launched
# (its noise load already queued behind it). The speculative path is designed around one slot.
SPECULATIVE_SLOTS = 1
# Eager runs of a stage before its capture (library workspaces, allocator warm-up).
WARMUP_RUNS = 3


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class GraphEngine(Engine):
    """Steps ``state`` with ``integrator`` on ``force_field``, as captured CUDA graphs.

    The planner (1a-T15) builds it from resolved configs: ``config.pipelined`` must be "on" or "off"
    (its "auto" is the planner's), ``neighbours`` is the force field's ``NeighbourConfig`` (edge
    multiple and bucket fraction; default the force field's own). Building it checks the process
    constraints (``fastmace.engine.constraints``; ``constraints`` is the process record, default the
    process's), then does the baseline's set-up: ``set_cell`` for a variable cell, compaction,
    ``use_edges``, the initial force evaluation into the state, ``integrator.start``. Graphs are
    captured on first use.

    ``capture`` (default: on CUDA) replays captured graphs; without it the stages run eagerly.
    ``pad_edges`` (default: ``capture``) rounds the edges up to the bucket; eager steps on the
    exact count are the baseline's ``cuda_graph=False`` path.
    """

    def __init__(self, config, force_field, integrator, state, *, neighbours=None, capture=None, pad_edges=None,
                 constraints=None):
        from fastmace.config import NeighbourConfig

        self.config, self.force_field, self.integrator, self.state = config, force_field, integrator, state
        device = state.positions.device
        self.device = device
        if _device(config.device) != device or force_field.device != device:
            raise ValueError(f"EngineConfig.device {config.device!r}, the force field ({force_field.device}) and the "
                             f"state ({device}) must be one device")
        self.cuda = device.type == "cuda"
        self.capture = self.cuda if capture is None else bool(capture)
        self.pad_edges = self.capture if pad_edges is None else bool(pad_edges)
        if config.pipelined == "auto":
            raise ValueError("EngineConfig.pipelined is 'auto'; the planner resolves it, so pass 'on' or 'off'")
        self.variable_cell = bool(integrator.variable_cell)
        if config.speculative_graph:
            if self.variable_cell:
                raise ValueError("speculative combined graphs are not available for NPT (variable cell)")
            if config.pipelined == "on":
                raise ValueError("speculative_graph excludes pipelined='on'")
            if not self.capture:
                raise ValueError("speculative_graph needs captured graphs")
        if self.pad_edges and not force_field.padding_exact:
            raise ValueError("This MACE model gives edges beyond the cutoff nonzero weights (apply_cutoff=True with "
                             "biased or species-dependent radial networks), so the CUDA-graph edge padding would "
                             "change its forces; use another engine for it")
        if self.variable_cell and not (force_field.compute_stress and state.stress is not None):
            raise ValueError("a variable-cell integrator needs a force field and a state with stress")
        self.speculative = bool(config.speculative_graph)
        self.pipelined = config.pipelined == "on"
        if neighbours is None:
            neighbours = getattr(getattr(force_field, "neighbours", None), "config", None) or NeighbourConfig()
        self.neighbour_config = neighbours
        self.process = process.check_engine(config, force_field, record=constraints, owner=type(self).__name__,
                                            segments=torch.cuda.memory_snapshot() if self.cuda else None)
        _record_production_engine(type(self).__name__)

        # Initial compaction and forces (GPULangevin._init_runtime; BAOAB sets the cell first).
        if self.variable_cell:
            force_field.set_cell(state.cell)
        force_field.compact(state.positions)
        initial_edges = int(force_field.edge_count)
        force_field.use_edges(initial_edges)
        self.initial_edges = initial_edges
        self.edge_bucket = edge_bucket(initial_edges, neighbours.edge_bucket_fraction, neighbours.edge_multiple)
        self._initial_forces()

        pinned = dict(dtype=torch.int64, pin_memory=self.cuda)
        self._graphs = {}
        self._pool = None
        self._pool_captures = 0
        self._captured_kinds = set()
        self._side_stream = None
        self.capture_memory = []
        self.pool_resets = 0
        self.edge_sizes_used = set()
        self.rebuilds = 0
        self.rebuild_needed = torch.zeros((), dtype=torch.bool, device=device)
        # Rebuild flag and edge count, read by the host with one transfer per step.
        self._flags = torch.zeros(2, dtype=torch.int64, device=device)
        self._flags_host = torch.zeros(2, **pinned)
        # Speculative steps: per slot [rebuild, count], copied behind the step.
        self._spec_flags = torch.zeros((SPECULATIVE_SLOTS, 2), dtype=torch.int64, device=device)
        self._spec_flags_host = torch.zeros((SPECULATIVE_SLOTS, 2), **pinned)
        self._spec_edges = self._padded(initial_edges)
        self._pending = []
        self.speculative_launches = 0
        self.speculative_rollbacks = 0
        self.speculative_rebuild_rollbacks = 0
        self.speculative_bucket_rollbacks = 0
        # Pipelined steps: flags [hard rebuild, count, early rebuild].
        self._pipe_flags = torch.zeros(3, dtype=torch.int64, device=device)
        self._pipe_flags_host = torch.zeros(3, **pinned)
        self._pipe_flags_view = self._pipe_flags_host.numpy()
        self._pipe_pending = None
        self._pipe_floor = 0  # padded size kept after a count redo, until the next rebuild
        self._pipe_edges = self._padded(initial_edges)
        self.pipelined_redos = 0
        self.pipelined_rebuild_redos = 0
        self.pipelined_early_rebuilds = 0
        self.canonical_points = 0
        self.polar_overflow_steps = []  # steps whose canonical point found the sticky polar flag set
        # What stage A produces and stage B changes (Langevin v; BAOAB v, p_eps), saved inside the
        # pipelined and speculative graphs so that a failed step can redo stage B alone.
        self._mids = tuple(torch.zeros_like(tensor) for tensor in integrator.mid_step_tensors(state))
        self.registry = Registry(self._registered_buffers)
        self.registry.entries()  # every buffer has a policy (RegistryError otherwise)
        integrator.start(state)  # LangevinBAOAB.__init__: the cell force of the starting forces

    # ------------------------------------------------------------------ public API

    @property
    def mode(self) -> str:
        return "speculative" if self.speculative else "pipelined" if self.pipelined else "checked"

    def step(self) -> None:
        if self.speculative:
            self._step_speculative()
        elif self.pipelined:
            self._step_pipelined()
        else:
            self._step_checked()

    def flush(self) -> None:
        """Validate every speculative or pipelined step still in flight (no-op otherwise)."""
        while self._pending:
            self._validate_oldest_speculative()
        self._validate_pipelined()

    def snapshot(self) -> Snapshot:
        """The current step's output values, copied to pinned host memory behind it without draining the GPU.

        Validates the step still in flight first (exactly what the next step would do). The copies do
        not change the MD state; ``Snapshot.wait()`` gives the values.
        """
        from fastmace.io.trajectory import copy_to_host

        self.flush()
        values, event = copy_to_host(self.integrator.output_values(self.state))
        return Snapshot(step=self.state.step, values=values, event=event)

    def sync(self) -> Mapping[str, Any]:
        """Flush and return the output values as NumPy arrays (synchronising)."""
        self.flush()
        return {key: value.detach().cpu().numpy() for key, value in self.integrator.output_values(self.state).items()}

    def potential_energy(self) -> float:
        self.flush()
        return float(self.state.energy[0])

    def info(self) -> Mapping[str, Any]:
        return {
            "engine": "graph", "mode": self.mode, "capture": self.capture, "pad_edges": self.pad_edges,
            "variable_cell": self.variable_cell, "initial_edges": self.initial_edges, "edge_bucket": self.edge_bucket,
            "edge_sizes_used": sorted(self.edge_sizes_used), "neighbour_rebuilds": self.rebuilds,
            "graphs": [str(key) for key in self._graphs], "pool_resets": self.pool_resets,
            "capture_memory": list(self.capture_memory), "speculative_launches": self.speculative_launches,
            "speculative_rollbacks": self.speculative_rollbacks,
            "speculative_rebuild_rollbacks": self.speculative_rebuild_rollbacks,
            "speculative_bucket_rollbacks": self.speculative_bucket_rollbacks, "pipelined_redos": self.pipelined_redos,
            "pipelined_rebuild_redos": self.pipelined_rebuild_redos,
            "pipelined_early_rebuilds": self.pipelined_early_rebuilds, "canonical_points": self.canonical_points,
            "polar_overflow_steps": list(self.polar_overflow_steps), "process": dict(self.process),
        }

    # ------------------------------------------------------------------ canonical restart points

    def canonicalise(self, *, reevaluate: bool = False) -> None:
        """Put the engine into the state a fresh build at the current positions, cell and species has (SPEC §10.6).

        Called by every restart write, in interrupted and uninterrupted runs alike, so that a resumed
        run continues bit for bit as an uninterrupted run with the same restart cadence. It quiesces
        the pipeline (``flush``), rebuilds the neighbour list from the current positions and cell (h0
        becomes the current cell; a variable-cell PolarMACE k-vector set is rebuilt too), compacts it
        and resets the registry's other ``reset`` buffers: the device flags, their host mirrors, the
        mid-step copies, the sticky polar overflow flag (its steps are recorded in
        ``polar_overflow_steps``) and the pipeline's padded-size floor and next sizes. The ``save``
        buffers are not touched. A rebuild that moved buffers drops the captured graphs.

        ``reevaluate=True`` (host writes of positions or momenta, owner decision D4) also evaluates the
        forces at the current positions and calls ``integrator.start``, as building does.
        """
        self.flush()
        state, field = self.state, self.force_field
        if self.variable_cell:
            field.set_cell(state.cell)
        moved = False
        reset_polar = getattr(field, "reset_polar", None)
        if reset_polar is not None:
            polar_moved, overflowed = reset_polar(state.cell.detach().cpu().numpy())
            moved |= polar_moved
            if overflowed:
                self.polar_overflow_steps.append(state.step)
        moved |= bool(field.rebuild(state.positions))
        if moved:
            self._clear_graphs()
        field.compact(state.positions)  # as the build does: eager, then the exact count
        count = int(field.edge_count)  # host sync
        field.use_edges(count)
        with torch.no_grad():
            for tensor in (self.rebuild_needed, self._flags, self._pipe_flags, self._spec_flags, self._flags_host,
                           self._pipe_flags_host, self._spec_flags_host, *self._mids):
                tensor.zero_()
        self._pipe_floor = 0
        self._pipe_edges = self._padded(count)
        self._spec_edges = self._padded(count)
        if reevaluate:
            self._force_stage()
            self.integrator.start(state)
        self.canonical_points += 1

    def restart_buckets(self) -> dict[str, int]:
        """The edge bucket and neighbour capacity for the restart's plan record (SPEC §10.6, ``Plan.to_restart``)."""
        return {"edge_bucket": int(self.edge_bucket), "neighbour_capacity": int(self.force_field.edge_capacity)}

    def restore_restart_point(self, *, edge_bucket: int | None = None, neighbour_capacity: int | None = None) -> None:
        """Resume: take a restart's edge bucket and neighbour capacity, then ``canonicalise``.

        Call it on a freshly built engine after the restart's state is loaded
        (``fastmace.io.restart.resume_engine``). ``None`` keeps the fresh build's value (files
        without a plan record).
        """
        if edge_bucket is not None:
            self.edge_bucket = int(edge_bucket)
        if neighbour_capacity is not None:
            neighbours = getattr(self.force_field, "neighbours", None)
            if not callable(getattr(neighbours, "set_capacity", None)):
                raise TypeError(f"{type(self.force_field).__name__} has no neighbour list whose capacity can be set")
            if neighbours.set_capacity(neighbour_capacity):
                self._clear_graphs()
        self.canonicalise()

    def _registered_buffers(self) -> dict[str, torch.Tensor]:
        """The registry's sources: every tensor a stage writes and that lives across steps, by name.

        The state first (its positions and barostat state are the integrator's too), then ``integrator.buffers`` and
        its temperature target, ``force_field.buffers``, the engine's flags, their host mirrors and
        the mid-step copies. ``fastmace.engine.registry.POLICIES`` has the policy of each.
        """
        state = self.state
        named = {f"state.{name}": getattr(state, name) for name in (
            "positions", "velocities", "cell", "species", "masses", "charges", "forces", "energy", "stress",
            "p_eps", "force_eps") if getattr(state, name, None) is not None}
        if state.rng.batch is not None:
            named["state.rng.batch"] = state.rng.batch
        named.update({f"integrator.{name}": tensor for name, tensor in self.integrator.buffers(state).items()})
        # Set by set_temperature between steps and read by every step; not in buffers() (1a).
        sqrt_kT = getattr(self.integrator, "sqrt_kT", None)
        if isinstance(sqrt_kT, torch.Tensor):
            named.setdefault("integrator.sqrt_kT", sqrt_kT)
        named.update({f"force_field.{name}": tensor for name, tensor in self.force_field.buffers().items()})
        named.update({"engine.rebuild_needed": self.rebuild_needed, "engine.flags": self._flags,
                      "engine.pipe_flags": self._pipe_flags, "engine.spec_flags": self._spec_flags,
                      "engine.flags_host": self._flags_host, "engine.pipe_flags_host": self._pipe_flags_host,
                      "engine.spec_flags_host": self._spec_flags_host})
        named.update({f"engine.mid.{index}": tensor for index, tensor in enumerate(self._mids)})
        return named

    # ------------------------------------------------------------------ steps

    def _padded(self, count: int) -> int:
        if self.pad_edges:
            return round_up_edges(count, self.force_field.edge_capacity, self.edge_bucket)
        return count

    def _use_edges(self, edges: int) -> None:
        self.force_field.use_edges(edges)
        self.edge_sizes_used.add(edges)

    def _rebuild(self) -> None:
        """Host-side rebuild of the candidates; a grown capacity moved the buffers, so every graph is stale."""
        self.rebuilds += 1
        if self.force_field.rebuild(self.state.positions):
            self._clear_graphs()

    def _clear_graphs(self) -> None:
        """Capacity grew: tensors moved, recapture all stages in a new pool (SPEC §8.1 re-capture seam)."""
        self._graphs.clear()
        self._pool = None
        self._pool_captures = 0
        self._captured_kinds.clear()

    def _step_checked(self) -> None:
        # Same draws, same order as ASE (drawn noise_batch steps ahead).
        self.integrator.prepare_step(self.state)
        # A and C as one stage, then one host transfer for the rebuild flag and the edge count. On
        # a rebuild step C runs again on the new candidates.
        self._run("AC")
        self._flags_host.copy_(self._flags, non_blocking=True)
        if self._flags_host.is_pinned():
            torch.cuda.current_stream().synchronize()  # host sync (the only one on most steps)
        rebuild, count = (int(value) for value in self._flags_host)
        if rebuild:
            self._rebuild()
            self._run("C")
            count = int(self.force_field.edge_count)  # host sync (rebuild steps only)
        edges = self._padded(count)
        self._use_edges(edges)
        self._run(("B", edges))
        self.state.step += 1

    def _step_pipelined(self) -> None:
        self._validate_pipelined()
        self.integrator.prepare_step(self.state)
        self._run("ACP")
        event = None
        self._pipe_flags_host.copy_(self._pipe_flags, non_blocking=True)
        if self._pipe_flags_host.is_pinned():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        edges = self._pipe_edges
        self._use_edges(edges)
        self._run(("B", edges))
        self._pipe_pending = (event, edges)
        self.state.step += 1

    def _validate_pipelined(self) -> None:
        if self._pipe_pending is None:
            return
        event, edges = self._pipe_pending
        self._pipe_pending = None
        if event is not None:
            event.synchronize()  # the flags of AC; B usually still runs
        flags = self._pipe_flags_view
        rebuild, count, early = bool(flags[0]), int(flags[1]), bool(flags[2])
        capacity = self.force_field.edge_capacity
        if rebuild or count > edges:
            # Stage A was exact; only B used a stale or truncated edge set.
            self.pipelined_redos += 1
            self._restore_mid()
            if rebuild:
                self.pipelined_rebuild_redos += 1
                self._rebuild()
                self._run("C")
                count = int(self.force_field.edge_count)  # host sync (redo only)
                capacity = self.force_field.edge_capacity
            edges = self._padded(count)
            self._use_edges(edges)
            self._run(("B", edges))
            self._pipe_floor = 0 if rebuild else edges
        elif early:
            # Rebuild before the hard limit; the next AC compacts the new candidates.
            self.pipelined_early_rebuilds += 1
            self._pipe_floor = 0
            self._rebuild()
            capacity = self.force_field.edge_capacity
        wanted = int(math.ceil(count * (1.0 + self.config.pipeline_edge_margin)))
        edges = self._padded(wanted) if self.pad_edges else min(wanted, capacity)
        self._pipe_edges = min(max(edges, self._pipe_floor), capacity)

    def _step_speculative(self) -> None:
        # Load the next noise first (stream-ordered behind the pending graph; a correction below
        # reads rnd_vel, not xi/eta), then validate the pending step: one step in flight, so a
        # failed speculation wastes only its own force stage, corrected in validation.
        self.integrator.prepare_step(self.state)
        if self._pending:
            self._validate_oldest_speculative()
        spec = self._spec_edges
        self.force_field.use_edges(spec)
        entry = {"slot": 0, "edges": spec, "event": None}
        self._pending.append(entry)
        self._run(("ACB", spec, 0))
        self.speculative_launches += 1
        self.edge_sizes_used.add(spec)
        self._spec_flags_host[0].copy_(self._spec_flags[0], non_blocking=True)
        if self.cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            entry["event"] = event

    def _validate_oldest_speculative(self) -> None:
        if not self._pending:
            return
        entry = self._pending.pop(0)
        if entry["event"] is not None:
            entry["event"].synchronize()
        rebuild = int(self._spec_flags_host[entry["slot"]][0])
        count = int(self._spec_flags_host[entry["slot"]][1])
        if not rebuild and count <= entry["edges"]:
            self.state.step += 1
            self._spec_edges = self._padded(count)
            return
        self.speculative_rollbacks += 1
        if rebuild:
            self.speculative_rebuild_rollbacks += 1
        else:
            self.speculative_bucket_rollbacks += 1
        # Stage A of the speculative step was exact (positions, random displacements); only its
        # force stage used a stale or truncated edge set. Restore what stage A produced and redo
        # just that stage, rebuilding the neighbour list first when the flags ask for it.
        self._restore_mid()
        if rebuild:
            self._rebuild()
            self._run("C")
            count = int(self.force_field.edge_count)
        edges = self._padded(count)
        self._use_edges(edges)
        self._spec_edges = edges
        self._run(("B", edges))
        self.state.step += 1

    # ------------------------------------------------------------------ stages

    def _initial_forces(self) -> None:
        self._force_stage()

    def _force_stage(self) -> None:
        """The force call on the current edges, its results copied into the state."""
        out = self.force_field(self.state.positions)
        self.state.forces.copy_(out.forces.detach())
        self.state.energy.copy_(out.energy.detach())
        if self.state.stress is not None and out.stress is not None:
            self.state.stress.copy_(out.stress.detach())

    def _stage_a(self) -> None:
        state = self.state
        self.integrator.first_half(state)
        if self.variable_cell:
            self.force_field.set_cell(state.cell)
        self.rebuild_needed.copy_(self.force_field.needs_rebuild(state.positions))

    def _stage_c(self) -> None:
        self.force_field.compact(self.state.positions)

    def _stage_ac(self) -> None:
        self._stage_a()
        self._stage_c()
        self._flags[0].copy_(self.rebuild_needed)
        self._flags[1].copy_(self.force_field.edge_count)

    def _stage_acp(self) -> None:
        self._stage_ac()
        self._save_mid()
        self._pipe_flags[0].copy_(self.rebuild_needed)
        self._pipe_flags[1].copy_(self.force_field.edge_count)
        self._pipe_flags[2].copy_(self.force_field.needs_rebuild(self.state.positions,
                                                                 self.config.pipeline_rebuild_fraction))

    def _stage_b(self) -> None:
        self._force_stage()
        self.integrator.second_half(self.state)

    def _save_mid(self) -> None:
        """Keep what stage A produced that stage B changes (redo of B after a failed step)."""
        for saved, tensor in zip(self._mids, self.integrator.mid_step_tensors(self.state), strict=True):
            saved.copy_(tensor)

    def _restore_mid(self) -> None:
        for saved, tensor in zip(self._mids, self.integrator.mid_step_tensors(self.state), strict=True):
            tensor.copy_(saved)

    def _stage(self, key):
        if isinstance(key, tuple) and key[0] == "ACB":
            slot = key[2]

            def combined():
                self._stage_ac()
                self._save_mid()
                self._spec_flags[slot].copy_(self._flags)
                self._stage_b()

            return combined
        return {"C": self._stage_c, "AC": self._stage_ac, "ACP": self._stage_acp}.get(key, self._stage_b)

    # ------------------------------------------------------------------ capture and replay

    def _run(self, key) -> None:
        if not self.capture:
            self._stage(key)()
            return
        if key not in self._graphs:
            self._capture(key)
        self._graphs[key].replay()

    def _warm_up(self, key) -> None:
        """The warm-up runs of a capture without the capture: run ``key`` eagerly, then restore the state."""
        stage = self._stage(key)
        saved = self.registry.snapshot()
        for _ in range(WARMUP_RUNS):
            stage()
        self.registry.restore(saved)

    def _capture(self, key) -> None:
        """Capture one stage as a CUDA graph without changing the MD state."""
        from fastmace import _patches

        if not self.cuda:
            raise RuntimeError("CUDA graphs need a CUDA device")
        _patches.set_capture_safe_mask_head(True)
        stage = self._stage(key)
        saved = self.registry.snapshot()
        if self._side_stream is None:  # one stream: library workspaces are made once
            self._side_stream = torch.cuda.Stream()
        side = self._side_stream
        side.wait_stream(torch.cuda.current_stream())
        # Only stage kinds captured before (e.g. B at a new edge size): the first capture of a
        # kind creates library workspaces (cuBLAS: 32-64 MiB), which must not live in the pool.
        kind = key[0] if isinstance(key, tuple) else key
        begin_pool = getattr(torch._C, "_cuda_beginAllocateCurrentStreamToPool", None)
        # Renamed between torch 2.5 (end...CurrentStreamToPool) and 2.14 (endAllocateToPool).
        end_pool = (getattr(torch._C, "_cuda_endAllocateToPool", None)
                    or getattr(torch._C, "_cuda_endAllocateCurrentStreamToPool", None))
        in_pool = (self.config.warmup_in_graph_pool and kind in self._captured_kinds and self._pool is not None
                   and begin_pool is not None and end_pool is not None)
        device = torch.cuda.current_device()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        with torch.cuda.stream(side):  # warm-up allocations and library handles
            if in_pool:
                # The pool keeps the freed intermediates of earlier graphs (about one force
                # evaluation); a warm-up outside it needs the same amount again. The stream-based
                # redirection of PyTorch's CUDA-graph trees includes the backward pass (autograd
                # thread, same stream).
                begin_pool(device, self._pool)
            try:
                for _ in range(WARMUP_RUNS):
                    stage()
            finally:
                if in_pool:
                    end_pool(device, self._pool)
                    torch._C._cuda_releasePool(device, self._pool)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        record = dict(key=str(key), warmup_in_pool=in_pool,
                      max_reserved_after_warmup_MiB=torch.cuda.max_memory_reserved() / 2**20)
        if in_pool and torch.cuda.memory_allocated() > allocated:
            # Something the warm-up keeps alive now lives in the pool, where replays of the existing
            # graphs could overwrite it. Retire the pool: drop all graphs (recaptured on demand in a
            # new pool); the allocation stays valid.
            self._clear_graphs()
            self.pool_resets += 1
        self.registry.restore(saved)
        del saved
        if self.config.empty_cache_before_capture:
            # The capture allocates in the graph pool and cannot reuse blocks cached by the warm-up,
            # so otherwise both count towards the peak.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # Capture on the warm-up stream: the allocator reuses free blocks per stream, so pool
        # blocks freed by earlier captures serve later warm-ups.
        with torch.cuda.graph(graph, pool=self._pool, stream=side):
            stage()
        torch.cuda.synchronize()
        record["max_reserved_after_capture_MiB"] = torch.cuda.max_memory_reserved() / 2**20
        self.capture_memory.append(record)
        self._graphs[key] = graph
        self._pool_captures += 1
        self._captured_kinds.add(kind)


__all__ = ["SPECULATIVE_SLOTS", "WARMUP_RUNS", "GraphEngine"]
