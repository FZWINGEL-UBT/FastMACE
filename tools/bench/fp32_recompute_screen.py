"""fp32 idea 2 screen: let AOTAutograd's min-cut partitioner recompute more pointwise work in the backward.

The default partitioner (``torch._functorch.partitioners.min_cut_rematerialization_partition``) keeps saved
activations small, but four heuristics ban some recomputations: ``ban_recompute_used_far_apart``,
``ban_recompute_long_fusible_chains``, ``ban_recompute_materialized_backward`` and
``ban_recompute_not_in_allowlist``. The fourth keeps matrix products and other non-pointwise ops from being
recomputed (the activation budgets of 15 October recomputed matrix products and lost 10-20 %), so it stays on
here. ``FP32_RECOMPUTE`` names the heuristics to turn off: ``base`` (none), ``far``, ``chains``, ``materialized``
or ``all`` (the three). The partitioner is set process-wide as Inductor's ``custom_partitioner_fn`` and prints,
per partitioned graph, the saved values' count and bytes (the backward's non-tangent inputs).

Then it runs ``tools/step_breakdown.py`` with the remaining arguments. Run with the Inductor FX-graph and
AOTAutograd caches off (``TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0``).
"""
import os
import runpy
import sys

import fastmace  # noqa: F401  (before torch)
import torch
from torch._inductor.custom_graph_pass import CustomPartitionerFn

FLAGS = {"far": "ban_recompute_used_far_apart", "chains": "ban_recompute_long_fusible_chains",
         "materialized": "ban_recompute_materialized_backward"}
MODE = os.environ.get("FP32_RECOMPUTE", "base")
OFF = list(FLAGS.values()) if MODE == "all" else [FLAGS[m] for m in MODE.split(",") if m in FLAGS]


def _bytes(node):
    value = node.meta.get("val")
    try:
        return int(value.numel()) * value.element_size()
    except (AttributeError, TypeError, RuntimeError):
        return 0


class Recompute(CustomPartitionerFn):
    def __call__(self, gm, joint_inputs, **kwargs):
        from torch._functorch import config, partitioners

        saved_flags = {name: getattr(config, name) for name in OFF}
        try:
            for name in OFF:
                setattr(config, name, False)
            fw, bw = partitioners.min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)
        finally:
            for name, value in saved_flags.items():
                setattr(config, name, value)
        saved = [n for n in bw.graph.nodes if n.op == "placeholder" and not n.name.startswith("tangents")]
        print(f"FP32_RECOMPUTE {MODE}: graph with {len(list(fw.graph.nodes))} forward nodes saves {len(saved)} values,"
              f" {sum(_bytes(n) for n in saved) / 2**20:.1f} MiB", flush=True)
        return fw, bw

    def uuid(self):
        return f"fastmace-fp32-recompute-{MODE}"


torch._inductor.config.custom_partitioner_fn = Recompute()
print(f"FP32_RECOMPUTE {MODE}: heuristics off {OFF}", flush=True)
tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "step_breakdown.py")
sys.argv = [tool] + sys.argv[1:]
runpy.run_path(tool, run_name="__main__")
