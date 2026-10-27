"""ideas D2 step-level probe: the float64 package with ``w_tables`` attached after the planner's rewrites.

D2_MODE=on attaches ``fastmace.rewrites.radial.attach_w_tables`` to the model right after the registry applied
the plan's records (so before compilation and capture) and runs the record's self-check against the chain it
replaces; D2_MODE=off changes nothing. Then it runs the tool named by the first argument (``sb64b.py`` or
``f64pkg_acc.py`` in tools/bench) with the remaining arguments.
"""
import os
import runpy
import sys

import fastmace  # noqa: F401  (before torch)
from fastmace.rewrites import radial, registry

if os.environ.get("D2_MODE") == "on":
    apply = registry.Registry.apply

    def _apply(self, model, *args, **kwargs):
        applied = apply(self, model, *args, **kwargs)
        if getattr(model, "interactions", None) is not None and any(
                getattr(i, "radial_table", None) is not None for i in model.interactions):
            names = radial.attach_w_tables(model)
            context = args[1] if len(args) > 1 else kwargs.get("context")
            result = radial._w_selfcheck(None, model, context)
            print(f"D2 w_tables: {names}; self-check {result.passed} max rel {result.max_relative_error:.2e} "
                  f"(threshold {result.threshold:.1e})", flush=True)
        return applied

    registry.Registry.apply = _apply

tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
sys.argv = [tool] + sys.argv[2:]
runpy.run_path(tool, run_name="__main__")
