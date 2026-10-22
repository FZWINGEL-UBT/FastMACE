"""The runtime-state registry: every persistent buffer, its owner and its policies (SPEC §8.1, §10.6).

Plan task 1b-T01 fixed the record and its policies; plan task 1b-T04 built the registry here and
the snapshot/restore that capture warm-ups use (it replaced phase 1a's ``GraphEngine.runtime_state``);
1b-T14 calls the ``reset`` functions at canonical restart points; the gate (1b-T16) snapshots and
restores the state around its replay. ``docs/design/contracts-1b.md`` gives the policy of every
buffer.

Two independent questions per buffer:

- **restart**: ``"save"`` (in the restart file), ``"reset"`` (recomputed at a canonical restart
  point by the function ``recompute`` names), or ``"derived"`` (recomputed from saved state when
  the engine is built or resumed);
- **capture**: snapshotted and restored around capture warm-ups (every registered buffer is, unless
  it is written before it is read in every stage); **sticky**: a flag that only a host decision
  clears, such as an overflow flag, which a warm-up must neither set nor clear.

The owners name their buffers (``Integrator.buffers``, ``ForceField.buffers``, the engine's own);
the policies live here, in ``POLICIES``, keyed by attribute path, so a buffer without a policy is an
error (``RegistryError``) rather than a silent omission. ``unregistered_mutations`` is the CI check
of SPEC §12.2: it runs an action on an eager engine and reports every tensor reachable from the
engine whose version counter moved but which the registry does not hold.
"""

from __future__ import annotations

import fnmatch
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from fastmace.errors import FastMACEError

if TYPE_CHECKING:
    import torch

RestartPolicy = Literal["save", "reset", "derived"]
RESTART_POLICIES = ("save", "reset", "derived")


class RegistryError(FastMACEError, RuntimeError):
    """A persistent buffer without a policy, or a restore whose buffer moved (SPEC §8.1)."""

    default_next_step = ("This is a FastMACE bug: report it with the buffer name. The policy table is "
                         "fastmace.engine.registry.POLICIES.")


@dataclass(frozen=True)
class RegisteredBuffer:
    """One persistent device buffer: where it lives, who owns it, and its policies.

    - ``name``: the attribute path from the engine, e.g. ``"force_field.polar.overflow"``;
    - ``owner``: the object that allocates and writes it (``"state"``, ``"integrator"``,
      ``"force_field"``, ``"neighbours"``, ``"polar"``, ``"engine"``);
    - ``restart``: one of ``RESTART_POLICIES``;
    - ``recompute``: for ``"reset"`` (required) and ``"derived"`` (optional), the name of the
      function that recomputes it; ``None`` for ``"save"``;
    - ``capture``: restored after capture warm-ups;
    - ``sticky``: a flag cleared only by a host decision.
    """

    name: str
    owner: str
    restart: RestartPolicy
    recompute: str | None = None
    capture: bool = True
    sticky: bool = False

    def __post_init__(self):
        for field in ("name", "owner"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"RegisteredBuffer.{field} must be a non-empty string, got {value!r}")
        if self.restart not in RESTART_POLICIES:
            raise ValueError(f"RegisteredBuffer {self.name}: restart policy {self.restart!r} is not one of "
                             f"{RESTART_POLICIES}")
        if self.recompute is not None and (not isinstance(self.recompute, str) or not self.recompute):
            raise ValueError(f"RegisteredBuffer {self.name}: recompute must name a function, got {self.recompute!r}")
        if self.restart == "reset" and self.recompute is None:
            raise ValueError(f"RegisteredBuffer {self.name}: a 'reset' buffer names the function that recomputes "
                             "it at the canonical restart point (recompute=...)")
        if self.restart == "save" and self.recompute is not None:
            raise ValueError(f"RegisteredBuffer {self.name}: a 'save' buffer comes from the restart file and is "
                             f"not recomputed (recompute={self.recompute!r})")
        for field in ("capture", "sticky"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"RegisteredBuffer {self.name}: {field} must be a bool, got {getattr(self, field)!r}")


def _policy(name, owner, restart, recompute=None, *, capture=True, sticky=False):
    return RegisteredBuffer(name, owner, restart, recompute, capture, sticky)


# The policy of every persistent buffer, by attribute path from the engine; ``fnmatch`` patterns, the
# first match wins (specific names before the patterns that cover them). docs/design/contracts-1b.md
# has the same table.
POLICIES = (
    # The physical state of the committed step (T05 writes it to the restart file).
    *(_policy(f"state.{name}", "state", "save") for name in (
        "positions", "velocities", "cell", "species", "masses", "charges", "forces", "energy", "stress")),
    _policy("state.rng.batch", "state", "save"),  # drawn-ahead noise; generator and cursor are host state
    # BAOAB's barostat momentum and cell force (in the state since 1b-T05; the integrator aliases them).
    *(_policy(f"state.{name}", "state", "save") for name in ("p_eps", "force_eps")),
    # Integrator scratch, written before it is read in every step.
    *(_policy(f"integrator.{name}", "integrator", "derived") for name in (
        "x_old", "rnd_pos", "rnd_vel", "xi", "eta", "zeta")),
    _policy("integrator.sqrt_kT", "integrator", "derived", "Integrator.set_temperature"),
    _policy("integrator.externalstress", "integrator", "derived", "Integrator.set_external_stress"),
    # BAOAB's copy of the target for the step in flight: stage A writes it from the target above.
    _policy("integrator.externalstress_step", "integrator", "derived", "LangevinBAOAB.restore_externalstress"),
    # Neighbour list: the copy of the current cell, then everything a rebuild makes.
    _policy("force_field.neighbours.cell", "neighbours", "derived", "ForceField.set_cell"),
    _policy("force_field.neighbours.*", "neighbours", "reset", "ForceField.rebuild"),
    # Model inputs the cell and positions feed.
    *(_policy(f"force_field.{name}", "force_field", "derived", "ForceField.set_cell")
      for name in ("cell", "volume", "rcell")),
    _policy("force_field.positions", "force_field", "derived"),  # refilled before every force call
    # Float64 per-atom E0 (1b-T03), from the species; changes only with a species event (phase 3).
    _policy("force_field.node_e0", "force_field", "derived", "fastmace.model.energy.node_e0_float64"),
    # PolarMACE k-space state: the overflow flag is sticky (``|=``), the k-vector set comes from the cell.
    _policy("force_field.polar.overflow", "polar", "reset", "fastmace.longrange.polar.reset_overflow",
            sticky=True),
    _policy("force_field.polar.*", "polar", "reset", "fastmace.longrange.polar.enable"),
    # The engine's device flags and mid-step copies: quiesced (validated) at a restart point.
    *(_policy(f"engine.{name}", "engine", "reset", "GraphEngine.flush", sticky=True)
      for name in ("rebuild_needed", "flags", "pipe_flags", "spec_flags", "overflow")),
    _policy("engine.mid.*", "engine", "reset", "GraphEngine.flush"),
    # Host mirrors of the flags: overwritten by the copy behind every step before the host reads them.
    *(_policy(f"engine.{name}", "engine", "derived", capture=False)
      for name in ("flags_host", "pipe_flags_host", "spec_flags_host")),
    # The derived-state registry's buffers (SPEC §8.5, 3-T03; fastmace.events): recomputed from the species,
    # masses, charges, friction and the force results by their refresh functions. Captured, so that a
    # warm-up of a graph that refreshes them (3-T09's attempts) leaves no trace.
    _policy("derived.*", "derived", "derived", "fastmace.events.DerivedStateRegistry.refresh"),
)


def policy_for(name: str) -> RegisteredBuffer:
    """The policy of the buffer at attribute path ``name``; ``RegistryError`` when it has none."""
    for entry in POLICIES:
        if entry.name == name or fnmatch.fnmatchcase(name, entry.name):
            return entry if entry.name == name else replace(entry, name=name)
    raise RegistryError(f"the runtime buffer {name} has no registry policy",
                        why="Every buffer a stage writes and that lives across steps needs a restart and a "
                            "capture policy (SPEC §8.1), or capture, restart and the gate would miss it.")


def _identity(tensor) -> tuple:
    return tensor.device, tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


@dataclass(frozen=True)
class RegistrySnapshot:
    """Copies of the captured buffers (``capture=True``), by name, with the identity they were taken from."""

    values: Mapping[str, tuple[tuple, Any]]


class Registry:
    """The registered buffers of one engine, resolved from ``sources`` on every call.

    ``sources()`` returns name -> tensor in priority order: a tensor that two owners name (the
    state's positions are the integrator's too) is registered once, under its first name. Resolving
    on every call keeps the registry right after a rebuild that moved the neighbour buffers.
    """

    def __init__(self, sources: Callable[[], Mapping[str, torch.Tensor]]):
        self._sources = sources

    def entries(self) -> dict[str, tuple[RegisteredBuffer, torch.Tensor]]:
        out, seen = {}, set()
        for name, tensor in self._sources().items():
            key = _identity(tensor)
            if key not in seen:
                seen.add(key)
                out[name] = (policy_for(name), tensor)
        return out

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: tensor for name, (_, tensor) in self.entries().items()}

    def snapshot(self) -> RegistrySnapshot:
        """Copies of every captured buffer, sticky flags included."""
        return RegistrySnapshot({name: (_identity(tensor), tensor.detach().clone())
                                 for name, (policy, tensor) in self.entries().items() if policy.capture})

    def restore(self, snapshot: RegistrySnapshot) -> None:
        """Copy ``snapshot`` back in place; ``RegistryError`` if a buffer moved or vanished since."""
        current = self.tensors()
        for name, (identity, _) in snapshot.values.items():
            if name not in current or _identity(current[name]) != identity:
                raise RegistryError(f"cannot restore the runtime buffer {name}: it moved since the snapshot",
                                    why="A neighbour rebuild that grew the capacity reallocates its buffers; a "
                                        "snapshot restores values in place, not addresses.")
        import torch

        with torch.no_grad():  # the force field's positions are a leaf that requires grad
            for name, (_, value) in snapshot.values.items():
                current[name].copy_(value)


# --------------------------------------------------------------------------- the CI check (SPEC §12.2)

_LEAVES = (str, bytes, int, float, complex, bool, type(None), type, types.ModuleType, types.FunctionType,
           types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)


def _skip(obj) -> bool:
    if isinstance(obj, _LEAVES):
        return True
    module = type(obj).__module__ or ""
    # NumPy arrays and generators, torch dtypes, devices, streams, events and graphs hold no tensors.
    return module.startswith("numpy") or module in ("torch", "torch.cuda", "torch.cuda.graphs",
                                                    "torch.cuda.streams")


def reachable_tensors(root, name: str = "engine") -> dict[int, tuple[str, torch.Tensor]]:
    """Every tensor reachable from ``root`` through attributes, dicts and sequences: id -> (path, tensor)."""
    import torch

    out, seen, stack = {}, set(), [(name, root)]
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            out[id(obj)] = (path, obj)
            continue
        if _skip(obj):
            continue
        if isinstance(obj, Mapping):
            children = [(f"{path}[{key!r}]", value) for key, value in obj.items()]
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = [(f"{path}[{index}]", value) for index, value in enumerate(obj)]
        else:
            fields = dict(getattr(obj, "__dict__", {}))
            for slot in getattr(type(obj), "__slots__", ()):
                if hasattr(obj, slot):
                    fields[slot] = getattr(obj, slot)
            children = [(f"{path}.{key}", value) for key, value in fields.items()]
        stack.extend(reversed(children))
    return out


def _version(tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:  # inference tensors have no version counter
        return None


def _storage(tensor):
    try:
        return tensor.device, tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None


def unregistered_mutations(engine, action: Callable[[], Any]) -> list[str]:
    """Run ``action`` and return the attribute paths of mutated tensors the registry does not hold.

    A tensor counts as mutated when its version counter moved (one that existed before ``action``)
    or is nonzero (one ``action`` created and kept). A tensor sharing storage with a registered buffer
    (a view of it) is covered. Needs an eager engine: captured replays do not move version counters.
    """
    # The tensors themselves are kept, so that no id is reused while ``action`` runs.
    before = {key: (tensor, _version(tensor)) for key, (_, tensor) in reachable_tensors(engine).items()}
    registered = {_storage(tensor) for tensor in engine.registry.tensors().values()}
    action()
    registered |= {_storage(tensor) for tensor in engine.registry.tensors().values()}
    found = []
    for key, (path, tensor) in reachable_tensors(engine).items():
        version = _version(tensor)
        if version is None or tensor.numel() == 0:
            continue
        old = before.get(key)
        moved = version != old[1] if old is not None and old[0] is tensor else version > 0
        if moved and _storage(tensor) not in registered:
            found.append(path)
    return sorted(found)


__all__ = ["POLICIES", "RESTART_POLICIES", "RegisteredBuffer", "Registry", "RegistryError", "RegistrySnapshot",
           "policy_for", "reachable_tensors", "unregistered_mutations"]
