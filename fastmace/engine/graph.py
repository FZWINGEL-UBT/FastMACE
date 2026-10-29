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

**Overflow before commit** (SPEC §8.1, plan task 3-T04): a step is committed only after the host has
checked every flag its evaluation can set. The edge count is checked against the padded edges the force
stage read (pipelined and speculative steps; a checked step reads the count first, so it cannot
overflow). With PolarMACE and a variable cell the sticky k-vector flag is read after every force stage
(one host sync per step, only in those plans; ``overflow_check``). An overflowed step restores what
stage A produced (the mid-step tensors), grows what overflowed (the neighbour list, or the k-vector set
at the step's cell) and redoes the force stage with the same noise: stage A is exact, so this is the
step redone from the state before it. Nothing observable comes from the overflowed evaluation:
``snapshot``, ``sync``, ``canonicalise`` and restart writes validate first, and ``uncommitted()`` is
true while a step is unvalidated. MC attempts get the same restore-and-redo with their own random-number
cursors through ``redo_on_overflow``; their evaluations set the device word ``overflow``
(``flag_edge_overflow``). A pipelined step adds to ``state.step`` at its launch and is committed at its
validation.

**Re-capture policy** (SPEC §8.1): every capture goes through ``_captured`` and is logged
(``recapture_log``, and a ``logging`` line). The first capture of a key is not counted; a capture that
replaces or adds a variant of a key captured before (a new edge size, a capture after
``invalidate_graphs``, an eviction or a pool reset) is a re-capture and counts against
``EngineConfig.recapture_budget``. Past the budget the engine warns once (``RecaptureBudgetWarning``),
records the step and recommends the compiled tier (``recapture_warning``). The engine itself never
downgrades; runtime adaptation (SPEC §9.6, ``AdaptationConfig``) may switch the run to the compiled tier at
a committed step. At most ``EngineConfig.graph_variant_cap`` graphs are kept:
the least recently replayed one is evicted, and its later capture counts. The registry
(``fastmace.engine.registry``) holds the buffer policies.

**Capture time** (5-T06): each capture's host time, from the synchronisation that ends the step's queued
work to the one that ends the capture (both are the capture's own, so the step gains none), adds to
``capture_seconds``, and a counted one to ``recapture_seconds``; ``runtime_counters()`` holds what the
runtime monitor (``fastmace.planner.adapt``) reads.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from time import perf_counter as _clock
from typing import Any

import torch

from fastmace.events import DerivedStateRegistry
from fastmace.neighbors.buckets import edge_bucket, round_up_edges

from . import constraints as process
from .base import Engine, Snapshot, _record_production_engine
from .registry import Registry

# Speculative combined-graph steps kept in flight: one, validated when the next step is launched
# (its noise load already queued behind it). The speculative path is designed around one slot.
SPECULATIVE_SLOTS = 1
# Eager runs of a stage before its capture (library workspaces, allocator warm-up).
WARMUP_RUNS = 3
# The ``fastmace_stage`` tag of an error raised while capturing (planner.fallback.stage_of; 2-T15b).
CAPTURE_STAGE = "capture"
# The energy-only graph's key (SPEC §8.1, 3-T07), and the stage keys the engine itself captures, which
# captured(key, fn) refuses for other modules' graphs (3-T04).
ENERGY_KEY = "E"
ENGINE_KEYS = ("A", "AC", "ACB", "ACP", "ACPB", "B", "C", ENERGY_KEY)
# Bits of the device overflow word (``GraphEngine.overflow``, ``overflowed()``) and their names in the logs.
EDGE_OVERFLOW = 1
KVECTOR_OVERFLOW = 2
OVERFLOW_KINDS = ((EDGE_OVERFLOW, "edges"), (KVECTOR_OVERFLOW, "k-vectors"))
# Redos of one evaluation before ``CapacityOverflow``: one grows what overflowed, so a second is a bug.
MAX_REDOS = 3
# The reason logged for the graphs a retired pool drops (``_capture``).
POOL_RESET = "graph pool retired"
# Entries kept in ``recapture_log`` and ``overflow_log`` (the counters keep counting past it).
LOG_LIMIT = 256

logger = logging.getLogger(__name__)


class CapacityOverflow(RuntimeError):
    """An evaluation still overflowed after ``MAX_REDOS`` redos (internal, SPEC §7.6: overflows are not public
    errors). ``word`` holds the ``EDGE_OVERFLOW`` / ``KVECTOR_OVERFLOW`` bits."""

    def __init__(self, word: int, redos: int):
        self.word = int(word)
        super().__init__(f"the evaluation still overflowed ({', '.join(_kinds(word))}) after {redos} redos")


class RecaptureBudgetWarning(UserWarning):
    """The run re-captured more graphs than ``EngineConfig.recapture_budget`` (SPEC §8.1)."""


def _kinds(word: int) -> list[str]:
    return [name for bit, name in OVERFLOW_KINDS if word & bit]


def _kind(key) -> Any:
    """A graph key's family: a stage key's first element (``("B", edges)`` -> ``"B"``), else the key."""
    return key[0] if isinstance(key, tuple) else key


def _precision(field) -> str:
    """``"float32"`` or ``"float64"``: a force field's precision."""
    return str(field.dtype).removeprefix("torch.")


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
        # An optimiser in the integrator's place (fastmace.optimize.driver.OptimizerStepper, 4-T05): checked
        # steps only, whose one host transfer also carries its status.
        self.optimizer = getattr(integrator, "optimizer", None)
        if self.optimizer is not None and (config.speculative_graph or config.pipelined == "on"):
            raise ValueError("an optimiser runs checked steps only: its converged flag decides each step before "
                             "the force call (pipelined='off', speculative_graph=False)")
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
        self._graphs = OrderedDict()  # least recently replayed first (the variant cap evicts from the front)
        self._pool = None
        self._pool_captures = 0
        self._captured_kinds = set()
        self._side_stream = None
        self._rebuild_stream = None  # EngineConfig.overlap_rebuild (OPT2-T04)
        self.rebuilds_beside = 0
        self.capture_memory = []
        self.pool_resets = 0
        self.edge_sizes_used = set()
        self.rebuilds = 0
        self.rebuild_needed = torch.zeros((), dtype=torch.bool, device=device)
        # Rebuild flag and edge count, read by the host with one transfer per step.
        self._flags = torch.zeros(2, dtype=torch.int64, device=device)
        self._flags_host = torch.zeros(2, **pinned)
        # An optimiser's [converged, max force, energy], read with the flags; halted: the last step stopped at it.
        self._status_host = torch.zeros(3, dtype=torch.float64, pin_memory=self.cuda)
        self._status_view = self._status_host.numpy()
        self.halted = False
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
        # EngineConfig.step_graph (OPT2-T04): the event inside the one-per-step graph that the next
        # step's validation waits on; external, so that it is recorded as a node of the graph.
        self._step_event = None
        if self.config.step_graph and self.cuda and self._pipe_flags_host.is_pinned():
            self._step_event = torch.cuda.Event(external=True)
        self._pipe_pending = None
        self._pipe_floor = 0  # padded size kept after a count redo, until the next rebuild
        self._pipe_edges = self._padded(initial_edges)
        self.pipelined_redos = 0
        self.pipelined_rebuild_redos = 0
        self.pipelined_early_rebuilds = 0
        self.canonical_points = 0
        self.polar_overflow_steps = []  # steps whose evaluation (or canonical point) found the polar flag set
        # Overflow before commit (3-T04): the device word MC evaluations set (EDGE_OVERFLOW, KVECTOR_OVERFLOW),
        # the per-step k-vector check of PolarMACE with a variable cell (one host sync after B), the redos.
        self.overflow = torch.zeros((), dtype=torch.int64, device=device)
        self.polar_check = self.variable_cell and getattr(force_field, "_polar_state", lambda: None)() is not None
        self.overflows_redone = 0
        self.overflow_log = []
        self._redoing = False
        # Re-capture policy (3-T04): the log, the budget ("auto" without a plan: the variant cap, as the
        # planner resolves it), evictions, invalidations, and why a dropped key's next capture counts.
        budget = config.recapture_budget
        self.recapture_budget = config.graph_variant_cap if budget == "auto" else int(budget)
        self.recaptures = 0
        self.capture_count = 0
        self.capture_seconds = 0.0  # host time of every capture (5-T06)
        self.recapture_seconds = 0.0  # ... of the counted ones
        self._capture_clock = None  # the capture's start: reset by _capture once the queued work is done
        self.recapture_log = []
        self.recapture_budget_exceeded_at = None
        self.graph_evictions = 0
        self.invalidations = []
        self._dropped = {}  # key -> why it was dropped (its next capture is a re-capture)
        self._kinds_seen = set()  # key families captured at least once in this engine
        self._external = {}  # captured(key, fn): other modules' stage functions
        self.trial_rebuilds = 0  # MC displacement trials built around their geometry (trial_rebuild, 3-T10)
        # The energy-only graph (3-T07): the force field per MC energy precision (the engine's own in its
        # precision, a float64 copy beside it, add_energy_field) and each precision's output buffer.
        self._energy_fields = {_precision(force_field): force_field}
        self._energy_out = {}
        # What stage A produces and stage B changes (Langevin v; BAOAB v, p_eps), saved inside the
        # pipelined and speculative graphs so that a failed step can redo stage B alone.
        self._mids = tuple(torch.zeros_like(tensor) for tensor in integrator.mid_step_tensors(state))
        # EngineConfig.fused_integrator (OPT3-T03): the AC/ACP stage in Triton kernels where it applies.
        self._fused = None
        self.fused_integrator_reason = None
        if config.fused_integrator:
            from fastmace.kernels.triton.fused_integrator import FusedIntegrator, applicable

            self.fused_integrator_reason = applicable(self)
            if self.fused_integrator_reason is None:
                self._fused = FusedIntegrator(self)
        # The derived-state registry (SPEC §8.5, 3-T03): the force field's species tables first, then the
        # integrator's mass and friction coefficients and its cell force, then the fused constants, so that a
        # refresh in registration order reads refreshed inputs. Events (fastmace.events) refresh it.
        self.derived = DerivedStateRegistry()
        register = getattr(force_field, "register_derived", None)
        if register is not None:
            register(self.derived, state)
        integrator.register_derived(self.derived, state)
        if self._fused is not None:
            self._fused.register_derived(self.derived)
        self.transactions_committed = 0  # committed events; with the step, the state version of caches (3-T07)
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

    def prepare(self) -> None:
        """Capture the stages the first step replays, before it (plan task 2-T15b; SPEC §8.1, §9.4).

        Capture is otherwise lazy, at a stage's first replay; ``Simulation`` calls this after the
        build and before the gate, so that a capture failure comes before the first step, where the
        fallback acts. The keys are the first step's: checked AC and B at the padded initial edge
        count, pipelined ACP and B at its first edge size, speculative ACB. A capture leaves the MD
        state as it was (the runtime-state registry restores it), so this only moves the captures
        earlier. A no-op without capture.

        Each stage is captured with the edge views its replay will read, exactly as the step sets
        them: a stage that evaluates forces (B, ACB) reads the first ``edges`` compacted edges of its
        key (``force_field.use_edges``), and a CUDA graph records those views. Captured with the
        build's exact count instead, a padded B graph read only the initial edges for the whole run,
        and every edge beyond them was dropped from the forces (2-T21d).
        """
        if not self.capture:
            return
        if self.speculative:
            keys = [("ACB", self._spec_edges, 0)]
        elif self.pipelined and self.config.step_graph:
            keys = [("ACPB", self._pipe_edges)]
        elif self.pipelined:
            keys = ["ACP", ("B", self._pipe_edges)]
        else:
            keys = ["AC", ("B", self._padded(self.initial_edges))]
        for key in keys:
            if key in self._graphs:
                continue
            edges = self._key_edges(key)
            if edges is not None:  # as _step_checked, _step_pipelined and _step_speculative do
                self.force_field.use_edges(edges)
            self._captured(key)

    @staticmethod
    def _key_edges(key) -> int | None:
        """The edge count a stage key's replay reads (B, ACB, ACPB and E keys), or None (AC, ACP, C)."""
        if isinstance(key, tuple) and key[0] in ("B", "ACB", "ACPB"):
            return int(key[1])
        if isinstance(key, tuple) and key[0] == ENERGY_KEY:
            return int(key[2])
        return None

    def _active_edges(self) -> int | None:
        """The number of edges the force field's model reads now (its edge views), or None if it has none."""
        data = getattr(self.force_field, "data", None)
        index = data.get("edge_index") if isinstance(data, dict) else None
        return None if index is None else int(index.shape[-1])

    def flush(self) -> None:
        """Validate every speculative or pipelined step still in flight (no-op otherwise)."""
        while self._pending:
            self._validate_oldest_speculative()
        self._validate_pipelined()

    def quiesce(self) -> None:
        """An event's first step (SPEC §8.5): finish the pipelined and speculative work, so that the state is
        a committed step. The output queue is the ``Simulation``'s, which drains it first."""
        self.flush()

    @property
    def state_version(self) -> tuple[int, int]:
        """``(state.step, transactions_committed)``: changes with every committed step and every committed
        event, for caches of the committed state (3-T07's energy). Not restart state."""
        return int(self.state.step), int(self.transactions_committed)

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

    def optimizer_status(self) -> tuple[bool, float, float]:
        """``(converged, max force, energy)`` as the last step's host transfer read them: the committed state
        before that step (the state itself when it halted). An optimiser engine only; no sync."""
        converged, force, energy = (float(value) for value in self._status_view)
        return bool(converged), force, energy

    def read_optimizer_status(self) -> tuple[bool, float, float]:
        """``(converged, max force, energy)`` of the committed state now (a host sync)."""
        if self.optimizer is None:
            raise TypeError("this engine steps an integrator, not an optimiser")
        converged, force, energy = (float(value) for value in self.integrator.status.detach().cpu())
        return bool(converged), force, energy

    def energy(self, *, precision: str | None = None) -> torch.Tensor:
        """The potential energy of the committed state from the forward-only energy graph (key ``ENERGY_KEY``,
        under ``torch.no_grad()``, float64 accumulation; SPEC §8.1), in ``precision`` (``"float32"``,
        ``"float64"``; ``None``: the force precision). A device scalar; no forces, no ``grad_fn``.

        The graph reads the current positions and the edges the force field reads now; its key is
        ``(ENERGY_KEY, precision, edges)``, one variant per edge size, captured at the first call like the
        force stage (warm-up, registry restore, re-capture policy). A precision other than the force
        precision needs its force field first (``add_energy_field``). The returned buffer has a fixed
        address and is overwritten by the next call. The evaluation flags an edge count beyond the active
        edges (``flag_edge_overflow``; PolarMACE's k-vector flag is set by its forward), so a caller decides
        nothing from it before ``overflowed()`` or ``redo_on_overflow`` (SPEC §8.1 "Overflow before commit").
        Without capture (CPU, eager) the stage runs directly, with the same result.
        """
        if self._pending or self._pipe_pending is not None:
            self.flush()  # the committed state: no step in flight
        key = self._energy_key(precision)
        self._run(key)
        return self._energy_out[key[1]]

    def energy_stage(self, *, precision: str | None = None) -> tuple[Callable[[], None], torch.Tensor]:
        """``(stage, out)``: the energy-only evaluation as a plain function, for another module's captured graph
        (3-T09's attempt: propose, apply, refresh, energy, accept). ``stage()`` writes the float64 energy at
        the current positions and active edges into ``out`` and flags an edge overflow, without a capture
        of its own; it reads the edges active now, so the caller captures one stage per edge size."""
        key = self._energy_key(precision)
        return self._stage_energy(key), self._energy_out[key[1]]

    def _energy_key(self, precision):
        """``(ENERGY_KEY, precision, edges)`` for the edges active now; binds the copy's edges and the output."""
        precision = _precision(self.force_field) if precision is None else precision
        field = self._energy_fields.get(precision)
        if field is None:
            raise ValueError(f"no {precision} force field for the energy-only graph (have "
                             f"{', '.join(self._energy_fields)}); add one with add_energy_field")
        edges = self._active_edges()
        if field is not self.force_field:
            field.use_edges(edges)
        if precision not in self._energy_out:
            self._energy_out[precision] = torch.zeros(1, dtype=torch.float64, device=self.device)
        return ENERGY_KEY, precision, edges

    @property
    def energy_fields(self) -> Mapping[str, Any]:
        """The energy-only graph's force field per precision (a copy of the mapping)."""
        return dict(self._energy_fields)

    def add_energy_field(self, field) -> None:
        """Give the energy-only graph a force field in another precision: the MC energy copy that follows this
        engine's force field (``fastmace.mc.evaluator.float64_copy``; SPEC §10.4 "Energy precision").

        Its species- and charge-derived buffers join the derived-state registry (``energy.<precision>.*``),
        so every transaction refreshes them with the force field's own (3-T03).
        """
        precision = _precision(field)
        if precision in self._energy_fields:
            raise ValueError(f"the energy-only graph already has a {precision} force field")
        if getattr(field, "_leader", None) is not self.force_field:
            raise ValueError("the energy force field must follow this engine's force field (MACEForceField.follow)")
        field.register_derived(self.derived, self.state, prefix=f"energy.{precision}")
        self._energy_fields[precision] = field

    def _stage_energy(self, key):
        _, precision, edges = key
        field, out = self._energy_fields[precision], self._energy_out[precision]

        def stage():
            out.copy_(field.energy_only(self.state.positions))
            self.flag_edge_overflow(self.force_field.edge_count, edges)

        return stage

    def captured(self, key: str, fn: Callable[[], Any]) -> Callable[[], None]:
        """Capture ``fn`` (no arguments, fixed shapes and addresses) as a CUDA graph under ``key`` with the
        engine's warm-up, registry restore and re-capture policy (variant cap, budget, log), and return
        its replay. For other modules' graphs (the MC attempt, 3-T09); ``ENGINE_KEYS`` are refused.

        The graph is captured at the first replay and again after an invalidation or an eviction, so the
        replay stays valid for the engine's life. Without capture (CPU, eager) the replay calls ``fn``.
        A tuple key's first element is its family: ``("mc.swap", bucket)`` keys are variants of one key.
        """
        if _kind(key) in ENGINE_KEYS:
            raise ValueError(f"{key!r} is an engine stage key ({', '.join(ENGINE_KEYS)}); other modules' graphs "
                             "need their own keys")
        if not callable(fn):
            raise TypeError(f"captured({key!r}, fn): fn must be callable, got {type(fn).__name__}")
        self._external[key] = fn

        def replay() -> None:
            self._run(key)

        return replay

    def invalidate_graphs(self, reason: str, keys: Iterable | None = None) -> None:
        """Drop the captured graphs (all, or those of ``keys``) and log ``reason``; their next capture is a
        re-capture (counted against the budget). Dropping all of them also retires the graph pool.

        For changes that captured graphs bake in: a grown capacity that moved buffers, a friction change
        (``c1``/``c2`` are Python floats in the integrator's kernels; 3-T03).
        """
        if keys is None:
            dropped = list(self._graphs)
            self._graphs.clear()
            self._pool = None
            self._pool_captures = 0
            self._captured_kinds.clear()
        else:
            dropped = [key for key in keys if key in self._graphs]
            for key in dropped:
                del self._graphs[key]
        for key in dropped:
            self._dropped[key] = str(reason)
        self.invalidations.append({"step": int(self.state.step), "reason": str(reason), "graphs": len(dropped)})
        del self.invalidations[:-LOG_LIMIT]
        logger.info("graph engine: dropped %d captured graphs at step %d (%s)", len(dropped), self.state.step, reason)

    def uncommitted(self) -> bool:
        """True while a step is unvalidated (pipelined or speculative, in flight) or an overflowed evaluation
        is being redone: nothing may be written or decided from the state now (3-T08's restart guard)."""
        return bool(self._pending) or self._pipe_pending is not None or self._redoing

    def overflowed(self) -> int:
        """The overflow bits set since the last clear: the device word and the polar flag (a host sync)."""
        word = int(self.overflow)
        flag = self._polar_flag()
        if flag is not None and bool(flag):
            word |= KVECTOR_OVERFLOW
        return word

    def flag_edge_overflow(self, count: torch.Tensor, edges) -> None:
        """Set ``EDGE_OVERFLOW`` in the device word when ``count`` exceeds ``edges`` (device ops only, so an MC
        evaluation can call it inside its captured graph; 3-T07, 3-T09, 3-T10)."""
        with torch.no_grad():
            count = torch.as_tensor(count, device=self.device)
            # A host number is compared as a scalar: as_tensor(int, device=cuda) is a host-to-device copy,
            # which a CUDA graph capture refuses (3-T07).
            limit = edges if isinstance(edges, (int, float)) else torch.as_tensor(edges, device=self.device)
            exceeded = count > limit
            self.overflow.bitwise_or_(exceeded.to(torch.int64) * EDGE_OVERFLOW)

    def clear_overflow(self) -> None:
        """Clear the device word and the polar flag (a host decision, after the redo or at a restart point)."""
        with torch.no_grad():
            self.overflow.zero_()
            flag = self._polar_flag()
            if flag is not None:
                flag.zero_()

    def redo_on_overflow(self, attempt: Callable[[], Any], *, keep: Iterable[torch.Tensor] = (),
                         max_redos: int = MAX_REDOS) -> int:
        """Run ``attempt`` (an MC attempt or block: proposal, refresh, evaluation) and redo it while it overflows.

        Before it: quiesce (``flush``), then snapshot the registered state and the tensors of ``keep`` (the
        caller's random-number cursors, e.g. an MC stream's device offset). After it the host reads
        ``overflowed()`` (one sync). On an overflow it restores both snapshots exactly, grows what
        overflowed (``_grow``), and runs ``attempt`` again with the same numbers; no decision is taken
        from the overflowed evaluation. Returns the number of redos; ``CapacityOverflow`` (internal) after
        ``max_redos``. The engine draws no random numbers itself.
        """
        self.flush()
        keep = tuple(keep)
        saved = self.registry.snapshot()
        kept = [tensor.detach().clone() for tensor in keep]
        self.clear_overflow()
        redos = 0
        self._redoing = True
        try:
            while True:
                attempt()
                word = self.overflowed()
                if not word:
                    return redos
                if redos >= max_redos:
                    raise CapacityOverflow(word, redos)
                redos += 1
                self.registry.restore(saved)
                with torch.no_grad():
                    for tensor, value in zip(keep, kept, strict=True):
                        tensor.copy_(value)
                self._grow(word)
                self._record_overflow(word, int(self.state.step), "mc")
                saved = self.registry.snapshot()  # a grown list has new buffers; the rest is as restored
        finally:
            self._redoing = False
            del saved

    # ------------------------------------------------------------------ MC trial lists (3-T10)

    def trial_rebuild(self) -> bool:
        """Rebuild the neighbour list around the current (trial) positions, eagerly, between replays: a
        displacement beyond the skin test (SPEC §8.5 step 4, §10.4 "Graph capture"). The GPU build reads
        its pair count on the host and may reallocate, which is why it runs outside the captured attempt.

        The list is built in place, so the captured graphs that read it stay valid. The caller saved the
        committed list first (``SkinNeighbourList.save_list``) and restores it on a rejection. Returns True
        if the capacity grew: the buffers moved and the captured graphs are dropped, and the caller
        restores the attempt and redoes it with the same numbers (an overflow, not an outcome).
        """
        self.trial_rebuilds += 1
        grew = bool(self.force_field.rebuild(self.state.positions))
        if grew:
            self.invalidate_graphs("a trial neighbour build grew the capacity")
        return grew

    def use_trial_edges(self) -> int:
        """Compact at the current positions and let the model read the padded edge count (one host sync, between
        replays); returns the padded size, which an MC attempt's graph key names."""
        field = self.force_field
        field.compact(self.state.positions)
        edges = self._padded(int(field.edge_count))  # host sync
        self._use_edges(edges)
        return edges

    def adopt_trial_list(self) -> None:
        """An accepted displacement keeps the list built around its geometry: the pipeline's padded sizes follow
        it as after a rebuild (its floor is dropped). The block still ends at a canonical point."""
        edges = self._padded(int(self.force_field.edge_count))  # host sync
        self._pipe_floor = 0
        self._pipe_edges = edges
        self._spec_edges = edges

    def record_mc_overflow(self, word: int) -> None:
        """An MC attempt overflowed (``word``) and its caller redoes it with the same numbers after growing what
        overflowed: count it (``overflows_redone``, the log) and clear the flags."""
        self._record_overflow(int(word), int(self.state.step), "mc")
        self.clear_overflow()

    @property
    def recapture_warning(self) -> str | None:
        """The report's line once the re-capture budget is exceeded (None before): it recommends the compiled tier."""
        if self.recapture_budget_exceeded_at is None:
            return None
        return (f"{self.recaptures} graph re-captures passed the budget of {self.recapture_budget} at step "
                f"{self.recapture_budget_exceeded_at}: frequent re-captures cost more than they save; plan the "
                'compiled tier (mode="compiled") for this workload')

    def info(self) -> Mapping[str, Any]:
        extra = {"rebuilds_beside": self.rebuilds_beside} if self.config.overlap_rebuild else {}
        if self.config.fused_integrator:  # True, or why the PyTorch stage was kept
            extra["fused_integrator"] = self.fused_integrator_reason or True
        if self.optimizer is not None:
            extra["optimizer"] = self.integrator.config.scheme
        return {**extra,
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
            "energy_graphs": {precision: {"copy": field is not self.force_field,
                                          "backend": getattr(field, "backend", None),
                                          "compile_mode": getattr(field, "compile_mode", None)}
                              for precision, field in self._energy_fields.items()},
            "overflow_check": "per step" if self.polar_check else "edges", "overflows_redone": self.overflows_redone,
            "overflow_log": [dict(entry) for entry in self.overflow_log], "recaptures": self.recaptures,
            "recapture_budget": self.recapture_budget,
            "recapture_budget_exceeded_at": self.recapture_budget_exceeded_at,
            "graph_variant_cap": self.config.graph_variant_cap, "graph_evictions": self.graph_evictions,
            "recapture_log": [dict(entry) for entry in self.recapture_log],
            "invalidations": [dict(entry) for entry in self.invalidations],
            "captures": self.capture_count, "capture_seconds": self.capture_seconds,
            "recapture_seconds": self.recapture_seconds,
        }

    def runtime_counters(self) -> Mapping[str, Any]:
        """The cumulative counters the runtime monitor reads at a canonical point (``planner.adapt``; 5-T06):
        neighbour rebuilds, captures and re-captures with their host time, overflows redone, and the step the
        re-capture budget was exceeded at. Cheaper than ``info()``, and part of it."""
        return {"neighbour_rebuilds": self.rebuilds, "captures": self.capture_count,
                "capture_seconds": self.capture_seconds, "recaptures": self.recaptures,
                "recapture_seconds": self.recapture_seconds, "overflows_redone": self.overflows_redone,
                "recapture_budget_exceeded_at": self.recapture_budget_exceeded_at}

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
        # Every list, D3's too, whatever its own test says (DispersedForceField.rebuild_all; 3b-T04).
        moved |= bool(getattr(field, "rebuild_all", field.rebuild)(state.positions))
        if moved:
            self.invalidate_graphs("the canonical rebuild moved buffers")
        field.compact(state.positions)  # as the build does: eager, then the exact count
        count = int(field.edge_count)  # host sync
        field.use_edges(count)
        with torch.no_grad():
            for tensor in (self.rebuild_needed, self._flags, self._pipe_flags, self._spec_flags, self._flags_host,
                           self._pipe_flags_host, self._spec_flags_host, self.overflow, *self._mids):
                tensor.zero_()
        self._pipe_floor = 0
        self._pipe_edges = self._padded(count)
        self._spec_edges = self._padded(count)
        if reevaluate:
            self._force_stage()
            self.integrator.start(state)
        self.canonical_points += 1

    def reset_cell(self, cell) -> None:
        """A host write of the cell at fixed cell (4-T05, for 4-T09's quench frames on one engine): ``state.cell``,
        the force field's lists and cell inputs, and the energy-graph copies, all in place. The caller ends at a
        canonical point (``canonicalise(reevaluate=True)``; ``Simulation.update`` does), whose rebuild writes
        the new shifts into the buffers the captured graphs read: only a grown capacity or a new padded edge
        size re-captures. A variable-cell engine is refused (its barostat state belongs to the cell)."""
        if self.variable_cell:
            raise ValueError("a host write of the cell is for fixed-cell engines; a variable-cell run moves its cell "
                             "itself")
        fields = [self.force_field] + [field for field in self._energy_fields.values() if field is not self.force_field]
        missing = [type(field).__name__ for field in fields if getattr(field, "reset_cell", None) is None]
        if missing:
            raise TypeError(f"{', '.join(missing)} cannot take a new cell (reset_cell)")
        if isinstance(cell, torch.Tensor):
            cell = cell.detach().to(self.device, torch.float64)
        else:  # a copy: read-only (frozen) arrays too
            cell = torch.tensor([[float(value) for value in row] for row in cell], dtype=torch.float64,
                                device=self.device)
        self.flush()
        with torch.no_grad():
            self.state.cell.copy_(cell)
        for field in fields:
            field.reset_cell(cell)

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
                self.invalidate_graphs("the restart's neighbour capacity")
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
        if self.optimizer is not None:  # 4-T05: the optimiser's history and scalars, the DOF view's own buffers
            named.update({f"optimizer.{name}": tensor for name, tensor in self.optimizer.buffers().items()})
            named.update({f"dof.{name}": tensor for name, tensor in self.optimizer.dof.buffers().items()})
        # Set by set_temperature between steps and read by every step; not in buffers() (1a).
        sqrt_kT = getattr(self.integrator, "sqrt_kT", None)
        if isinstance(sqrt_kT, torch.Tensor):
            named.setdefault("integrator.sqrt_kT", sqrt_kT)
        externalstress = getattr(self.integrator, "externalstress", None)  # the pressure target (3b-T05)
        if isinstance(externalstress, torch.Tensor):
            named.setdefault("integrator.externalstress", externalstress)
        named.update({f"force_field.{name}": tensor for name, tensor in self.force_field.buffers().items()})
        named.update({"engine.rebuild_needed": self.rebuild_needed, "engine.flags": self._flags,
                      "engine.pipe_flags": self._pipe_flags, "engine.spec_flags": self._spec_flags,
                      "engine.flags_host": self._flags_host, "engine.pipe_flags_host": self._pipe_flags_host,
                      "engine.spec_flags_host": self._spec_flags_host, "engine.overflow": self.overflow,
                      "engine.status_host": self._status_host})
        named.update({f"engine.mid.{index}": tensor for index, tensor in enumerate(self._mids)})
        # Derived buffers not registered above under their owner's name (a tensor is registered once).
        named.update({f"derived.{name}": self.derived[name].buffer for name in self.derived.names()
                      if isinstance(self.derived[name].buffer, torch.Tensor)})
        return named

    # ------------------------------------------------------------------ steps

    def _padded(self, count: int) -> int:
        if self.pad_edges:
            return round_up_edges(count, self.force_field.edge_capacity, self.edge_bucket)
        return count

    def _use_edges(self, edges: int) -> None:
        self.force_field.use_edges(edges)
        self.edge_sizes_used.add(edges)

    def _rebuild(self, beside: bool = False) -> None:
        """Host-side rebuild of the candidates; a grown capacity moved the buffers, so every graph is stale.

        ``beside`` (an early rebuild of a pipelined step, with ``EngineConfig.overlap_rebuild``): the rebuild
        runs on a side stream while the force graph in flight finishes on the current stream, and the
        current stream waits for it before the next step (OPT2-T04). Its host syncs then wait for the
        rebuild alone, not for the force call. Only for fixed-cell CUDA engines: the rebuild reads the
        positions, which the second half of a Langevin step does not write.
        """
        self.rebuilds += 1
        if beside and self.config.overlap_rebuild and self.cuda and not self.variable_cell:
            main = torch.cuda.current_stream()
            if self._rebuild_stream is None:
                self._rebuild_stream = torch.cuda.Stream()
            neighbours = getattr(self.force_field, "neighbours", None)
            if neighbours is not None:
                neighbours.reader_stream = main  # a growth waits for the force call and allocates on main
            try:
                with torch.cuda.stream(self._rebuild_stream):
                    grew = self.force_field.rebuild(self.state.positions)
            finally:
                if neighbours is not None:
                    neighbours.reader_stream = None
            main.wait_stream(self._rebuild_stream)  # the next AC compacts the new candidates
            self.rebuilds_beside += 1
        else:
            grew = self.force_field.rebuild(self.state.positions)
        if grew:
            self.invalidate_graphs("the neighbour capacity grew")

    def _clear_graphs(self) -> None:
        """Capacity grew: tensors moved, recapture all stages in a new pool (``invalidate_graphs``)."""
        self.invalidate_graphs("capacity grew")

    # ------------------------------------------------------------------ overflow before commit

    def _polar_flag(self) -> torch.Tensor | None:
        """The sticky PolarMACE k-vector overflow flag (a device bool), or None without a polar state."""
        state = getattr(self.force_field, "_polar_state", lambda: None)()
        return None if state is None else state.overflow

    def _grow(self, word: int) -> None:
        """Make what overflowed big enough for the current state, then clear the flags (a host decision).

        K-vectors: the set is rebuilt for the current cell (``reset_polar``; the step's cell, which stage A
        wrote). Edges: the neighbour list is rebuilt at the current positions and compacted. Either may
        move buffers, which drops the captured graphs (``invalidate_graphs``).
        """
        field, state = self.force_field, self.state
        if word & KVECTOR_OVERFLOW:
            moved, _ = field.reset_polar(state.cell.detach().cpu().numpy())
            if moved:
                self.invalidate_graphs("the k-vector set grew")
        if word & EDGE_OVERFLOW:
            if field.rebuild(state.positions):
                self.invalidate_graphs("the neighbour capacity grew")
            field.compact(state.positions)
        self.clear_overflow()

    def _record_overflow(self, word: int, step: int, mode: str | None = None) -> None:
        self.overflows_redone += 1
        if word & KVECTOR_OVERFLOW:
            self.polar_overflow_steps.append(step)
        self.overflow_log.append({"step": step, "kinds": _kinds(word), "mode": mode or self.mode})
        del self.overflow_log[:-LOG_LIMIT]
        logger.info("graph engine: step %d overflowed (%s) and was redone", step, ", ".join(_kinds(word)))

    def _polar_word(self) -> int:
        """``KVECTOR_OVERFLOW`` if the force stage just run set the polar flag (the per-step host sync)."""
        if not self.polar_check:
            return 0
        flag = self._polar_flag()
        return KVECTOR_OVERFLOW if flag is not None and bool(flag) else 0  # host sync (polar NPT plans only)

    def _redo_force_stage(self, word: int, step: int) -> None:
        """An overflowed force stage: restore stage A's output, grow, redo B on the current edges, check again."""
        for _ in range(MAX_REDOS):
            self._restore_mid()
            self._grow(word)
            self._record_overflow(word, step)
            edges = self._padded(int(self.force_field.edge_count))  # host sync (redo only)
            self._use_edges(edges)
            self._run(("B", edges))
            word = self._polar_word()
            if not word:
                return
        raise CapacityOverflow(word, MAX_REDOS)

    def _step_checked(self) -> None:
        # Same draws, same order as ASE (drawn noise_batch steps ahead).
        self.integrator.prepare_step(self.state)
        # A and C as one stage, then one host transfer for the rebuild flag and the edge count. On
        # a rebuild step C runs again on the new candidates.
        self._run("AC")
        self._flags_host.copy_(self._flags, non_blocking=True)
        if self.optimizer is not None:
            self._status_host.copy_(self.integrator.status, non_blocking=True)
        if self._flags_host.is_pinned():
            torch.cuda.current_stream().synchronize()  # host sync (the only one on most steps)
        if self.optimizer is not None:
            # A converged optimiser left the positions unchanged in stage A: the committed state stays, and
            # the step neither evaluates forces nor counts (4-T05).
            self.halted = bool(self._status_view[0])
            if self.halted:
                return
        rebuild, count = (int(value) for value in self._flags_host)
        if rebuild:
            self._rebuild()
            self._run("C")
            count = int(self.force_field.edge_count)  # host sync (rebuild steps only)
        edges = self._padded(count)
        self._use_edges(edges)
        if self.polar_check:
            self._save_mid()  # a k-vector overflow redoes B from here (the pipelined graphs save it inside)
        self._run(("B", edges))
        word = self._polar_word()
        if word:
            self._redo_force_stage(word, self.state.step + 1)
        self.state.step += 1  # committed: every flag of the step checked

    def _step_pipelined(self) -> None:
        self._validate_pipelined()
        self.integrator.prepare_step(self.state)
        if self.config.step_graph:
            # One replay per step (OPT2-T04): ACP, the flags' copy to the host, an event the host waits
            # on (inside the graph), then B. The kernels and their order are those of ACP and B.
            edges = self._pipe_edges
            self._use_edges(edges)  # before ACP, which reads no edge views
            self._run(("ACPB", edges))
            self._pipe_pending = (self._step_event, edges)
            self.state.step += 1
            return
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
        if self.polar_check and self.cuda:
            torch.cuda.current_stream().synchronize()  # the k-vector flag of B (polar NPT plans only)
        word = self._polar_word() | (EDGE_OVERFLOW if count > edges and not rebuild else 0)
        if word:
            self._record_overflow(word, self.state.step)
        if word & KVECTOR_OVERFLOW:
            self._grow(KVECTOR_OVERFLOW)  # the k-vector set for the step's cell, before B is redone below
        if rebuild or word:
            # Stage A was exact; only B used a stale or truncated edge set, or too few k-vectors.
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
            if self._polar_word():  # the redo itself cannot overflow: the set covers the step's cell
                raise CapacityOverflow(KVECTOR_OVERFLOW, 1)
            if rebuild or word & EDGE_OVERFLOW:  # a k-vector redo alone keeps the floor
                self._pipe_floor = 0 if rebuild else edges
        elif early:
            # Rebuild before the hard limit; the next AC compacts the new candidates.
            self.pipelined_early_rebuilds += 1
            self._pipe_floor = 0
            self._rebuild(beside=event is not None)  # beside the force graph still in flight (OPT2-T04)
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
            self._record_overflow(EDGE_OVERFLOW, self.state.step + 1)
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
        if self._fused is not None:
            self._fused.stage(pipelined=False)
            return
        self._stage_a()
        self._stage_c()
        self._flags[0].copy_(self.rebuild_needed)
        self._flags[1].copy_(self.force_field.edge_count)

    def _stage_acp(self) -> None:
        if self._fused is not None:
            self._fused.stage(pipelined=True)
            return
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
        external = self._external.get(key)
        if external is not None:
            return external
        if _kind(key) == ENERGY_KEY:
            return self._stage_energy(key)
        if isinstance(key, tuple) and key[0] == "ACB":
            slot = key[2]

            def combined():
                self._stage_ac()
                self._save_mid()
                self._spec_flags[slot].copy_(self._flags)
                self._stage_b()

            return combined
        if isinstance(key, tuple) and key[0] == "ACPB":
            return self._stage_acpb
        return {"C": self._stage_c, "AC": self._stage_ac, "ACP": self._stage_acp}.get(key, self._stage_b)

    def _stage_acpb(self) -> None:
        """ACP, the flags to the host, the event the next step's validation waits on, then B (step_graph)."""
        self._stage_acp()
        self._pipe_flags_host.copy_(self._pipe_flags, non_blocking=True)
        if self._step_event is not None:
            self._step_event.record(torch.cuda.current_stream())
        self._stage_b()

    # ------------------------------------------------------------------ capture and replay

    def _run(self, key) -> None:
        if not self.capture:
            self._stage(key)()
            return
        graph = self._graphs.get(key)
        if graph is None:
            self._captured(key)
            graph = self._graphs[key]
        else:
            self._graphs.move_to_end(key)  # most recently replayed last (the variant cap evicts the first)
        graph.replay()

    def _captured(self, key) -> None:
        """``_capture(key)``; an error it raises is tagged with the stage ``"capture"`` (``CAPTURE_STAGE``)
        and raised again unchanged, so the fallback can tell a capture failure from others (2-T15b).

        A force stage (B, ACB) must be captured reading its key's edges: the graph keeps the edge
        views active now for every replay. Anything else is a bug in the caller and raises before the
        capture (2-T21d), instead of a graph that drops edges from the forces for the rest of the run.
        """
        edges, active = self._key_edges(key), self._active_edges()
        if edges is not None and active is not None and active != edges:
            raise RuntimeError(f"capturing {key!r} with {active} active edges, not its {edges}: the graph would "
                               "read the wrong edges for the rest of the run (call force_field.use_edges first)")
        counted, reason = self._capture_reason(key)
        while len(self._graphs) >= self.config.graph_variant_cap:
            evicted, _ = self._graphs.popitem(last=False)
            self._dropped[evicted] = "evicted by the variant cap"
            self.graph_evictions += 1
            logger.info("graph engine: evicted %s (variant cap %d)", evicted, self.config.graph_variant_cap)
        self._capture_clock = _clock()
        try:
            self._capture(key)
        except Exception as error:
            try:
                error.fastmace_stage = CAPTURE_STAGE
            except AttributeError:  # pragma: no cover - an exception type without a __dict__
                pass
            raise
        seconds = _clock() - self._capture_clock
        self._kinds_seen.add(_kind(key))
        self._record_capture(key, counted, reason, seconds)

    def _capture_reason(self, key) -> tuple[bool, str]:
        """Whether capturing ``key`` now counts against the budget, and why (conductor answer 3 of 3-T04)."""
        if key in self._dropped:
            return True, self._dropped.pop(key)
        if _kind(key) in self._kinds_seen:
            return True, f"new variant of {_kind(key)}"
        return False, "first capture"

    def _record_capture(self, key, counted: bool, reason: str, seconds: float) -> None:
        step = int(self.state.step)
        self.recapture_log.append({"step": step, "key": str(key), "reason": reason, "counted": counted,
                                   "s": seconds})
        del self.recapture_log[:-LOG_LIMIT]
        self.capture_count += 1
        self.capture_seconds += seconds
        if counted:
            self.recapture_seconds += seconds
        if not counted:
            logger.info("graph engine: captured %s at step %d (%s)", key, step, reason)
            return
        self.recaptures += 1
        logger.info("graph engine: re-captured %s at step %d (%s; %d of a budget of %d)", key, step, reason,
                    self.recaptures, self.recapture_budget)
        if self.recaptures > self.recapture_budget and self.recapture_budget_exceeded_at is None:
            self.recapture_budget_exceeded_at = step
            warnings.warn(self.recapture_warning, RecaptureBudgetWarning, stacklevel=4)

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
        self._capture_clock = _clock()  # the step's queued work is done: the capture's time starts here
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
            self.invalidate_graphs(POOL_RESET)
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


__all__ = ["CAPTURE_STAGE", "SPECULATIVE_SLOTS", "WARMUP_RUNS", "GraphEngine"]
