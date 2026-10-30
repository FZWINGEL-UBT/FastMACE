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
import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_rewrites  # noqa: E402
import ab_step_time  # noqa: E402

SIDE = "b"  # the side being built or stepped; ab_rewrites sets it to "a" for side a's build (--a-constant)
PHASE = "build"  # "build" while a side builds, "step" while a side steps (its warm-up re-captures included)
_ENGINE_SIDES = {}
# The trace: every Python call of a swapped attribute (captures and compiles; graph replays make none) by
# (side, phase, the code that ran, attribute). Side a's entries must all be "reference", side b's "library".
CALLS = {}


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

        def dispatch(*args, _library=library, _reference=reference, _attr=attr, **kwargs):
            code = "reference" if this.SIDE == "a" else "library"
            key = (this.SIDE, this.PHASE, code, _attr)
            this.CALLS[key] = this.CALLS.get(key, 0) + 1
            return (_reference if code == "reference" else _library)(*args, **kwargs)

        setattr(owner, attr, dispatch)
    build = minimal.MinimalPlanner.build

    def recorded_build(self, *args, **kwargs):
        engine = build(self, *args, **kwargs)
        _ENGINE_SIDES[id(engine)] = this.SIDE
        return engine

    minimal.MinimalPlanner.build = recorded_build
    step = ab_step_time.Tf32Side.step

    def sided_step(self):
        this.SIDE, this.PHASE = _ENGINE_SIDES[id(self.side.engine)], "step"
        try:
            step(self)
        finally:
            this.SIDE, this.PHASE = "b", "build"

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
    result = ab_rewrites.main([*rest, "--a-constant", "ab_swap:SIDE='a'"])
    trace = [dict(side=side, phase=phase, code=code, attribute=attr, calls=n)
             for (side, phase, code, attr), n in sorted(CALLS.items())]
    wrong = [t for t in trace if (t["side"] == "a") != (t["code"] == "reference")]
    print("ab_swap trace:", json.dumps(trace), "wrong side:", len(wrong))
    out = rest[rest.index("--output") + 1] if "--output" in rest else None
    if out:
        Path(f"{out}.swap_trace.json").write_text(json.dumps(dict(swaps=swaps, trace=trace, wrong=wrong), indent=1))
    return result


if __name__ == "__main__":
    main()
