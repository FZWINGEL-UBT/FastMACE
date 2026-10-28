"""ideas F0 ceiling: tools/step_breakdown.py with the chunked backward's folded input gradient stubbed.

F0_MODE=stub replaces ``skinny_linear.input_grad_into`` by a zero fill of its output (same buffers, one memset per
chunk instead of the Triton product), so the MD step shows how much of the input gradient is on the critical
path. Forces are wrong in stub mode (the second interaction's radial term is missing): timing only.
Usage: F0_MODE=base|stub python f0_stub.py <step_breakdown.py arguments>
"""
import os
import runpy
import sys

import fastmace  # noqa: F401  (before torch)
from fastmace.kernels.triton import skinny_linear

if os.environ.get("F0_MODE") == "stub":
    from fastmace.rewrites import registry

    original, applying = skinny_linear.input_grad_into, [0]

    def _stub(grad, weight, main, last, out, tile_rule="fixed"):
        if applying[0]:  # the rewrites' self-checks at setup see the real product
            return original(grad, weight, main, last, out, tile_rule)
        return out.zero_()

    apply = registry.Registry.apply

    def _apply(self, *args, **kwargs):
        applying[0] += 1
        try:
            return apply(self, *args, **kwargs)
        finally:
            applying[0] -= 1

    registry.Registry.apply = _apply
    skinny_linear.input_grad_into = _stub
    print("F0 stub: folded input gradient replaced by a zero fill", flush=True)

tool = os.path.join(os.path.dirname(fastmace.__file__), "..", "tools", "step_breakdown.py")
sys.argv = [tool] + sys.argv[1:]
runpy.run_path(tool, run_name="__main__")
