"""f64-T04 step 2: record the streaming convolution's float64 operands from one eager force call of a float64 plan.

Usage: TORCHDYNAMO_DISABLE=1 python record_f64.py LABEL STRUCTURE OUTDIR  ->  OUTDIR/ops64-LABEL-STRUCTURE.pt
Adapted from the OPT7 recorder (~/tmp/opt7/record.py): the plan is fm.ForceField(..., precision="float64") with
MinimalPlanner(engine=EngineConfig(backend="hybrid")). Saves per streaming call x, sh, the per-edge weights w
(a @ W^T for the folded path), senders, receivers, row_ptr, the plan's own output and the OEq problem.
"""
import sys
from pathlib import Path

import fastmace as fm

sys.path.insert(0, str(Path(fm.__file__).resolve().parents[1] / "tools"))
import ab_step_time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fastmace.kernels.oeq_streaming as streaming  # noqa: E402
from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

label, spec, outdir = sys.argv[1:4]
records = []


def problem_dict(kernels):
    problem = kernels.problem
    instructions = [dict(i_in1=i.i_in1, i_in2=i.i_in2, i_out=i.i_out, mode=i.connection_mode,
                         has_weight=bool(i.has_weight), path_weight=float(i.path_weight))
                    for i in problem.instructions]
    return dict(irreps_in1=str(problem.irreps_in1), irreps_in2=str(problem.irreps_in2),
                irreps_out=str(problem.irreps_out), instructions=instructions, weight_numel=problem.weight_numel,
                layout=kernels.layout, num_threads=kernels.num_threads)


def keep(x, sh, w, senders, receivers, row_ptr, kernels, out):
    records.append(dict(x=x.detach().clone(), sh=sh.detach().clone(), w=w.detach().clone(), senders=senders.clone(),
                        receivers=receivers.clone(), row_ptr=row_ptr.clone(), problem=problem_dict(kernels),
                        out=out.detach().clone()))


original_folded, original_plain = streaming.folded_streaming_convolution, streaming.streaming_convolution


def folded(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule="fixed", parts=None):
    out = original_folded(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule, parts)
    keep(x, sh, a.detach() @ weight.detach().T, senders, receivers, row_ptr, kernels, out)
    return out


def plain(x, sh, weights, senders, receivers, row_ptr, kernels):
    out = original_plain(x, sh, weights, senders, receivers, row_ptr, kernels)
    keep(x, sh, weights, senders, receivers, row_ptr, kernels, out)
    return out


streaming.folded_streaming_convolution, streaming.streaming_convolution = folded, plain
original_init = streaming.StreamingKernels.__init__


def init(self, problem, *args, **kwargs):
    original_init(self, problem, *args, **kwargs)
    self.problem = problem


streaming.StreamingKernels.__init__ = init
ab_step_time.prepare_fastmace_hybrid()
name, atoms = ab_step_time.read_structure(spec, None)
atoms.positions += np.random.default_rng(20261027).uniform(-0.05, 0.05, atoms.positions.shape)
planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
hw = planner.hardware()
model = fm.load(label)
field = fm.ForceField(model, atoms, precision="float64")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hw)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
records.clear()
out = engine.force_field(engine.state.positions)
torch.cuda.synchronize()
print(label, spec, "calls", len(records), [(tuple(r["x"].shape), str(r["x"].dtype), tuple(r["w"].shape),
                                            int(r["row_ptr"][-1]), r["problem"]["layout"]) for r in records],
      flush=True)
torch.save(dict(label=label, structure=spec, records=[{k: (v.cpu() if torch.is_tensor(v) else v)
                                                       for k, v in r.items()} for r in records]),
           f"{outdir}/ops64-{label}-{spec}.pt")
