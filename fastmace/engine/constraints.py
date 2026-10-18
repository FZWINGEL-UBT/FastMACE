"""Process constraints of the CUDA-graph engine, checked when an engine is built (SPEC §3.2, §8.7).

Some of what an engine relies on is process-wide state of torch or MACE, not of the engine:

- **Allocator first.** ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` acts only if it is set
  before the first CUDA allocation (``fastmace.model.compile.configure_allocator``). Later it is
  silently ignored, so the check looks at the allocator's segments, not at the variable.
- **Dynamo cache limit.** ``torch._dynamo.config.cache_size_limit`` counts every edge size and
  model against one per-function limit; beyond it dynamo silently runs eager. It is raised to
  ``EngineConfig.dynamo_cache_size_limit`` and never lowered.
- **One compile family per process.** MACE's autograd-compile policy
  (``mace.tools.compile.configure_autograd_for_compile``) is process-wide: OEq (hybrid) models are
  compiled with ``torch.autograd.grad`` disallowed in the graph, cuEquivariance and e3nn models
  with it allowed. Dynamo recompiles for every new edge size, under the policy in force at that
  moment, so a compiled hybrid and a compiled plain-cuEq engine cannot share a process. The second
  family is refused; SPEC §8.7's subprocess runner is the way to mix them.
- **PolarMACE resets dynamo per engine.** A compiled PolarMACE copy carries its structure's
  k-space state; code compiled for a previous structure kept that call's autograd graph alive and
  broke the next capture (research engine, 16 October 2026). ``MACEForceField`` resets dynamo
  before compiling a polar copy; the engine checks that it did.
- **Streaming convolutions need receiver-sorted edges** (1a-T07): with unsorted edges MH-1's
  energy was off by a factor of 2.5 while every self-check passed. The force field refuses the
  configuration; the engine also checks the actual edge order once, which costs nothing per step.

The record of compile families is the one piece of module-level state here (``PROCESS``): what it
protects is process-wide itself. It is a record of facts, never a switch, and tests pass their own
``ProcessRecord``. The 1b runtime-state registry and the planner's subprocess runner build on it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastmace.errors import FastMACEError


class ProcessConstraintError(FastMACEError, RuntimeError):
    """An engine would break a process-wide constraint (SPEC §8.7); joins fastmace.errors in 1b."""


# Compile families by MACE's autograd-compile policy: the force derivative traced into the
# compiled graph ("autograd-in-graph": cuEquivariance, e3nn) or taken outside it ("oeq": hybrid).
OEQ_FAMILY = "oeq"
AUTOGRAD_FAMILY = "autograd-in-graph"
EXPANDABLE = "expandable_segments:True"


def compile_family(force_field) -> str | None:
    """The force field's compile family, or None when its model is not compiled."""
    if getattr(force_field, "compile_mode", "none") == "none":
        return None
    return OEQ_FAMILY if getattr(force_field, "oeq", False) else AUTOGRAD_FAMILY


@dataclass
class ProcessRecord:
    """What engines built so far in this process have fixed: the compile family and who fixed it."""

    family: str | None = None
    family_owner: str | None = None
    engines: int = 0
    notes: list[str] = field(default_factory=list)

    def claim_family(self, family: str | None, owner: str) -> None:
        """Record ``family`` (None: nothing compiled, always allowed); refuse a second family."""
        if family is None:
            return
        if self.family is None:
            self.family, self.family_owner = family, owner
            return
        if family != self.family:
            raise ProcessConstraintError(
                f"this process already compiled a {self.family} model ({self.family_owner}); MACE's "
                f"autograd-compile policy is process-wide, so a compiled {family} model ({owner}) cannot "
                "run in the same process. Run it in a separate process (SPEC §8.7)")


PROCESS = ProcessRecord()


def wants_expandable_segments(config, environ: Mapping[str, str] | None = None,
                              torch_version: str | None = None) -> bool:
    """Whether ``EngineConfig.expandable_segments`` asks for the expandable-segments allocator.

    "on": yes. "auto" (rule ``on_for_torch_2_14_unless_alloc_conf_set``): from torch 2.14, unless the
    user set ``PYTORCH_CUDA_ALLOC_CONF`` to something else. "off": no.
    """
    from fastmace.model.compile import ALLOCATOR_MIN_TORCH

    setting = config.expandable_segments
    if setting == "off":
        return False
    if setting == "on":
        return True
    environ = os.environ if environ is None else environ
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    version = tuple(int(part) for part in torch_version.split("+")[0].split(".")[:2])
    current = environ.get("PYTORCH_CUDA_ALLOC_CONF")
    return version >= ALLOCATOR_MIN_TORCH and (current is None or EXPANDABLE in current)


def check_allocator(config, segments: list[Mapping[str, Any]] | None, environ: Mapping[str, str] | None = None,
                    torch_version: str | None = None) -> str:
    """Refuse an engine whose allocator setting did not take effect; returns what is in effect.

    ``segments`` is ``torch.cuda.memory_snapshot()`` (None on the CPU: nothing to check). Every
    segment of an allocator with expandable segments is expandable; one that is not means the
    variable was set after the first CUDA allocation (or not at all).
    """
    if segments is None:
        return "cpu"
    environ = os.environ if environ is None else environ
    if not wants_expandable_segments(config, environ, torch_version):
        return "default"
    if EXPANDABLE not in (environ.get("PYTORCH_CUDA_ALLOC_CONF") or ""):
        raise ProcessConstraintError(
            f"EngineConfig.expandable_segments={config.expandable_segments!r} needs "
            f"PYTORCH_CUDA_ALLOC_CONF={EXPANDABLE} before the first CUDA allocation "
            "(fastmace.model.compile.configure_allocator), and it is not set")
    fixed = [segment for segment in segments if not segment.get("is_expandable", False)]
    if fixed:
        raise ProcessConstraintError(
            f"PYTORCH_CUDA_ALLOC_CONF={EXPANDABLE} was set after the first CUDA allocation: {len(fixed)} of "
            f"{len(segments)} allocator segments are not expandable. Call configure_allocator before "
            "anything touches CUDA")
    return EXPANDABLE


def ensure_dynamo_cache_limit(config, dynamo_config=None) -> int:
    """Raise dynamo's cache-size limit to ``config.dynamo_cache_size_limit`` (never lower it); returns it."""
    if dynamo_config is None:
        import torch._dynamo

        dynamo_config = torch._dynamo.config
    dynamo_config.cache_size_limit = max(dynamo_config.cache_size_limit, config.dynamo_cache_size_limit)
    return dynamo_config.cache_size_limit


def check_polar_reset(force_field) -> None:
    """A compiled PolarMACE force field must have reset dynamo before compiling its copy."""
    if not getattr(force_field, "is_polar", False) or compile_family(force_field) is None:
        return
    if not getattr(force_field, "dynamo_reset", False):
        raise ProcessConstraintError(
            "a compiled PolarMACE force field must reset dynamo before compiling its own copy (code compiled "
            "for another structure keeps that call's autograd graph alive and breaks the next capture)")


def check_receiver_sorting(force_field) -> None:
    """Streaming convolutions read receiver-sorted edges: check the flag and the compacted edges.

    Host-side, once at engine construction (one transfer of the edge index).
    """
    info = force_field.info()
    if not info.get("streaming_interactions"):
        return
    if not info.get("sort_receivers"):
        raise ProcessConstraintError(
            "the model has streaming convolutions, but its neighbour list does not sort edges by receiver; "
            "the streaming kernels then give wrong results while every self-check passes (1a-T07)")
    count = int(force_field.edge_count)
    edge_index = force_field.buffers().get("neighbours.edge_index")
    if edge_index is None or count < 2:
        return
    receivers = edge_index[1, :count]
    if bool((receivers[1:] < receivers[:-1]).any()):
        raise ProcessConstraintError("the compacted edges of a streaming model are not sorted by receiver")


def check_engine(config, force_field, *, record: ProcessRecord | None = None, owner: str = "engine",
                 segments: list[Mapping[str, Any]] | None = None, dynamo_config=None) -> dict[str, Any]:
    """Every process constraint of a new engine, before its first force evaluation; returns the facts.

    ``segments``: ``torch.cuda.memory_snapshot()`` on CUDA, None on the CPU. The compile family is
    claimed last, after every other check has passed.
    """
    record = PROCESS if record is None else record
    facts = {"allocator": check_allocator(config, segments),
             "dynamo_cache_size_limit": ensure_dynamo_cache_limit(config, dynamo_config)}
    check_polar_reset(force_field)
    check_receiver_sorting(force_field)
    family = compile_family(force_field)
    record.claim_family(family, owner)
    record.engines += 1
    facts["compile_family"] = family
    return facts


__all__ = [
    "AUTOGRAD_FAMILY", "EXPANDABLE", "OEQ_FAMILY", "PROCESS", "ProcessConstraintError", "ProcessRecord",
    "check_allocator", "check_engine", "check_polar_reset", "check_receiver_sorting", "compile_family",
    "ensure_dynamo_cache_limit", "wants_expandable_segments",
]
