"""Which interactions run through folded_streaming_convolution, and with the fused forward? argv: label structure
[extra RewriteConfig field=value ...]. One eager force call with fused_folded_forward=on."""
import sys
from pathlib import Path

import fastmace as fm

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import torch  # noqa: E402

import fastmace.kernels.oeq_streaming as streaming  # noqa: E402
from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

label, spec = sys.argv[1:3]
extra = dict(kv.split("=") for kv in sys.argv[3:])
calls = []
original = streaming.folded_streaming_convolution


def wrapped(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule="fixed", parts=None,
            fused=None):
    calls.append(dict(a=tuple(a.shape), w=tuple(weight.shape), x=tuple(x.shape), fused=fused is not None,
                      rule=getattr(fused, "rule", None)))
    return original(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule, parts, fused)


streaming.folded_streaming_convolution = wrapped
ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(spec, None)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
hw = planner.hardware()
model = fm.load(label)
field = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(fused_folded_forward="on", **extra))
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hw)
print("plan:", plan.rewrites.fused_folded_forward, plan.rewrites.chunked_conv_backward, plan.rewrites.streaming_conv)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
engine.force_field(engine.state.positions)
torch.cuda.synchronize()
print(label, spec, "calls", calls)
