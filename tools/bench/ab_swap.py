"""Same-process A/B of a kernel change: side a runs ``main``'s code for one or more module attributes, side b the
library's (plans/capacity-ideas-20261030.md, fifth list).

``tools/ab_rewrites.py`` with its arguments, plus ``--swap MODULE:ATTR=NAME ...``: while side a builds or steps,
``MODULE.ATTR`` (or ``MODULE.CLASS.METHOD``) is ``tools/bench/reference_kernels.py``'s ``NAME`` (its CUDA graphs
capture that code, also when a new edge size re-captures in the warm-up); otherwise it is the library's own.
Measurement only: the swap is a dispatcher installed in this process. A swapped forward is traced through the
dispatcher, so the compiled model of each side guards on ``SIDE`` and keeps its own code. ``--a``/``--b``
RewriteConfig fields still apply, so the same swap can be measured with an opt-in rewrite on both sides (``--a
gate_epilogue=on --b gate_epilogue=on``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_rewrites  # noqa: E402
import ab_step_time  # noqa: E402

SIDE = "b"  # the side being built or stepped; ab_rewrites sets it to "a" for side a's build (--a-constant)
_ENGINE_SIDES = {}


def install(swaps):
    """Dispatchers for ``swaps`` [(module, attr, reference name)], and each engine's side (recorded at its build,
    set before each of its steps)."""
    import reference_kernels

    from fastmace.planner import minimal

    this = sys.modules[__name__]
    for module_name, attr, name in swaps:
        owner = importlib.import_module(module_name)
        *path, attr = attr.split(".")  # a class attribute: CLASS.METHOD
        for part in path:
            owner = getattr(owner, part)
        library, reference = getattr(owner, attr), getattr(reference_kernels, name)

        def dispatch(*args, _library=library, _reference=reference, **kwargs):
            return (_reference if this.SIDE == "a" else _library)(*args, **kwargs)

        setattr(owner, attr, dispatch)
    build = minimal.MinimalPlanner.build

    def recorded_build(self, *args, **kwargs):
        engine = build(self, *args, **kwargs)
        _ENGINE_SIDES[id(engine)] = this.SIDE
        return engine

    minimal.MinimalPlanner.build = recorded_build
    step = ab_step_time.Tf32Side.step

    def sided_step(self):
        this.SIDE = _ENGINE_SIDES[id(self.side.engine)]
        try:
            step(self)
        finally:
            this.SIDE = "b"

    ab_step_time.Tf32Side.step = sided_step


def main(argv=None):
    sys.modules.setdefault("ab_swap", sys.modules[__name__])
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--swap", nargs="+", required=True, metavar="MODULE:ATTR=NAME")
    args, rest = p.parse_known_args(sys.argv[1:] if argv is None else argv)
    swaps = []
    for item in args.swap:
        target, _, name = item.partition("=")
        module, _, attr = target.partition(":")
        if not (module and attr and name):
            raise SystemExit(f"--swap {item!r}: expected MODULE:ATTR=NAME")
        swaps.append((module, attr, name))
    ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    install(swaps)
    return ab_rewrites.main([*rest, "--a-constant", "ab_swap:SIDE='a'"])


if __name__ == "__main__":
    main()
