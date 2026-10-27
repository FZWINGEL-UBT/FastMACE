"""OPT8-T01: record the second MH-1 interaction's folded streaming convolution operands from one eager force call.

Usage: TORCHDYNAMO_DISABLE=1 python opt8_record_operands.py LABEL STRUCTURE OUTDIR  ->  OUTDIR/ops-LABEL-STRUCTURE.pt

Saves, per ``folded_streaming_convolution`` call (the radial tables' output feeds it): the folded layer's input
``a`` [E, 65], its weight ``weight`` [1280, 65], x, sh, senders, receivers, row_ptr, the message ``out``, its
gradient ``dout``, ``chunk_edges`` and ``tile_rule`` of the plan, and the OEq problem. Default plan (hybrid float32,
rattled cell, seed 20261027) as in the OPT7 records.
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
                layout=kernels.layout)


original_folded = streaming.folded_streaming_convolution


def folded(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule="fixed", parts=None):
    out = original_folded(a, weight, x, sh, senders, receivers, row_ptr, kernels, chunk_edges, tile_rule, parts)
    entry = dict(a=a.detach().clone(), weight=weight.detach().clone(), x=x.detach().clone(),
                 sh=sh.detach().clone(), senders=senders.clone(), receivers=receivers.clone(),
                 row_ptr=row_ptr.clone(), problem=problem_dict(kernels), out=out.detach().clone(),
                 chunk_edges=int(chunk_edges), tile_rule=tile_rule)
    records.append(entry)
    if out.requires_grad:
        out.register_hook(lambda g: entry.__setitem__("dout", g.detach().clone()))
    return out


streaming.folded_streaming_convolution = folded
import fastmace.rewrites.mh1.modules as modules  # noqa: E402

for name in dir(modules):
    if getattr(modules, name) is original_folded:
        setattr(modules, name, folded)
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
field = fm.ForceField(model, atoms, precision="float32")
plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0), hw)
engine = planner.build(plan, field, atoms.copy(), temperature_K=300.0)
records.clear()
out = engine.force_field(engine.state.positions)
torch.cuda.synchronize()
records = [r for r in records if "dout" in r]
print(label, spec, "calls", len(records), [(tuple(r["x"].shape), tuple(r["a"].shape), tuple(r["weight"].shape),
                                            int(r["row_ptr"][-1]), r["chunk_edges"], r["tile_rule"])
                                           for r in records], flush=True)
torch.save(dict(label=label, structure=spec, records=[{k: (v.cpu() if torch.is_tensor(v) else v)
                                                       for k, v in r.items()} for r in records]),
           str(Path(outdir) / f"ops-{label}-{spec}.pt"))
