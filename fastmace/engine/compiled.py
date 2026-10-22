"""The compiled MD tier (SPEC §8.2): ``torch.compile`` without CUDA graphs (contracts: plan task 3-T01).

3-T06 fills it: MD with bucketed or dynamic edge shapes and the same integrators, output, restart and
events as the graph engine; the planner chooses it for inexact padding (rule 4), frequent host
callbacks (rule 9) and as the fallback's second mode (graph → compiled → eager → reference). Only
the planner instantiates it (SPEC §5).
"""

from __future__ import annotations

from typing import Any

MODE = "compiled"


class CompiledEngine:
    """``CompiledEngine(config, force_field, integrator, state, neighbours=None, constraints=None,
    dynamic_edges=True)``; an ``engine.base.Engine`` once 3-T06 implements it."""

    def __init__(self, config: Any, force_field: Any, integrator: Any, state: Any, neighbours: Any = None,
                 constraints: Any = None, dynamic_edges: bool = True):
        raise NotImplementedError("the compiled MD tier: plan task 3-T06")
