"""The compiled MD tier (SPEC §8.2): ``torch.compile`` without CUDA graphs (plan task 3-T06).

The graph engine's stages, integrators, output, restart and events, run without capture on a model
compiled in ``default`` mode. With ``dynamic_edges`` (the default) the stages read the exact edge
count and the compiled model sees the edge dimension as dynamic from the first call, so a new edge
count after a neighbour rebuild does not compile again (the evaluation path of 2-T13); that is what
lets a model whose edge padding is inexact run here (rule 4). Without it the edges are padded to the
graph tier's buckets, which needs exact padding. The planner chooses this tier for inexact padding
(rule 4), frequent host callbacks (rule 9) and as the fallback's second mode (graph → compiled →
eager → reference). Only the planner instantiates it (SPEC §5).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .graph import GraphEngine

MODE = "compiled"


class CompiledEngine(GraphEngine):
    """``CompiledEngine(config, force_field, integrator, state, neighbours=None, constraints=None,
    dynamic_edges=True)``: a ``GraphEngine`` that never captures.

    ``dynamic_edges`` steps on exact edge counts; a force field that compiles must then mark its
    edges dynamic (``EngineConfig.dynamic_edges``). ``False`` pads the edges to buckets.
    """

    def __init__(self, config: Any, force_field: Any, integrator: Any, state: Any, neighbours: Any = None,
                 constraints: Any = None, dynamic_edges: bool = True):
        if config.speculative_graph:
            raise ValueError("speculative combined graphs need CUDA graphs; the compiled tier has none")
        if dynamic_edges and force_field.compile_mode != "none" and not force_field.dynamic_edges:
            raise ValueError("exact edge counts on a compiled model need EngineConfig.dynamic_edges, or every "
                             "neighbour rebuild with a new edge count compiles again")
        self.dynamic_edges = bool(dynamic_edges)
        super().__init__(config, force_field, integrator, state, neighbours=neighbours, capture=False,
                         pad_edges=not dynamic_edges, constraints=constraints)

    def info(self) -> Mapping[str, Any]:
        return {**super().info(), "tier": MODE, "dynamic_edges": self.dynamic_edges,
                "compile_mode": self.force_field.compile_mode}
