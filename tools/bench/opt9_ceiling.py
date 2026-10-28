"""OPT9-T02 ceiling: the default float32 MD step with the radial backward's removable work stubbed (timing only).

Side ``a`` is the minimal planner's default plan. Side ``b`` is the same plan built while two stubs are active:

- the chunked backward's input gradient (``skinny_linear.input_grad_into``, called by
  ``streaming_conv_folded_backward`` for every chunk) only zeroes its output rows;
- the radial tables' backward kernel (both interactions) only zeroes ``dL/dr``.

The stubs act while side ``b`` is planned, built and warmed up, which is when its CUDA graphs are captured, except
inside ``Registry.apply`` and ``Registry.selfcheck`` (the rewrites' self-checks and the radial tables' certificates
compare against the original model); side ``a`` is
built and warmed up first with them off, so its graphs hold the real kernels. Side ``b``'s forces are wrong (the
radial r-gradient is missing): its timing is a ceiling for the w'-dot backward of plans/opt-9.md, which would
remove both and the ``dw`` chunk stores (not stubbed here; standalone 0.16-0.40 ms in OPT9-T01). It is an
over-estimate in that a fused kernel adds its own work (OPT9-T01's ``wdot``) and that the table backward of
interaction 0 is stubbed too. Timed blocks alternate (``fastmace.bench.ab.run_ab``); a recompile while timed makes
the structure invalid. Writes ``<output>.json``. Use an idle GPU.

Usage: python tools/bench/opt9_ceiling.py --output OUT --checkpoint mace-mh-1.model [--structures NaNbO3 NbCl5]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402

STUB = {"active": False, "input_grad": 0, "table_backward": 0}


def install_stubs():
    from fastmace.kernels.triton import radial_table, skinny_linear

    real_input_grad = skinny_linear.input_grad_into

    def input_grad_into(grad, weight, main, last, out, tile_rule="fixed"):
        if STUB["active"]:
            STUB["input_grad"] += 1
            return out.zero_()
        return real_input_grad(grad, weight, main, last, out, tile_rule)

    skinny_linear.input_grad_into = input_grad_into
    forward, backward = radial_table._kernels()

    class Backward:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if STUB["active"]:
                    STUB["table_backward"] += 1
                    args[8].zero_()  # grad_r (r, pair, dcoef, starts, scales, bins, grad_g, grad_density, grad_r, ...)
                    return None
                return backward[grid](*args, **kwargs)
            return launch

    stub = (forward, Backward())
    radial_table._kernels = lambda: stub

    # Applying the rewrites runs their self-checks and the radial tables' certificates against the original
    # model (inside planner.build -> MACEForces -> compile_model): run all of it with the stubs off.
    from fastmace.rewrites import registry

    def stubs_off(real):
        def wrapped(self, *args, **kwargs):
            active, STUB["active"] = STUB["active"], False
            try:
                return real(self, *args, **kwargs)
            finally:
                STUB["active"] = active
        return wrapped

    registry.Registry.apply = stubs_off(registry.Registry.apply)
    registry.Registry.selfcheck = stubs_off(registry.Registry.selfcheck)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaNbO3", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=200)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--label", default="")
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    install_stubs()
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    meta = dict(tool="opt9_ceiling", label=args.label, created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                protocol=protocol.as_dict(), pattern=args.pattern, structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup, stubbed = {}, {}, {}
        for side in ("a", "b"):
            STUB.update(active=side == "b", input_grad=0, table_backward=0)
            started = time.perf_counter()
            field = ForceField(loaded, base, precision="float32", rewrites="auto")
            integrator = IntegratorConfig(temperature_K=args.temperature_K)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
            sides[side] = ab_step_time.FastMACESide(engine, torch)
            step_time.warm_up(sides[side], protocol.warmup)
            setup[side] = time.perf_counter() - started
            stubbed[side] = {k: STUB[k] for k in ("input_grad", "table_backward")}
        STUB["active"] = False
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        late = {k: STUB[k] for k in ("input_grad", "table_backward")}  # stub calls while timed: a recapture
        entry = dict(natoms=len(base), ab=result.as_dict(), ratio_b_over_a=medians["b"] / medians["a"],
                     ceiling_ms=medians["a"] - medians["b"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])],
                     setup_s=setup, stub_calls_during_build=stubbed, stub_calls_while_timed=late,
                     compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides},
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = (not recompiled and not entry["other_gpu_processes"] and stubbed["b"]["input_grad"] > 0
                          and stubbed["b"]["table_backward"] > 0)
        meta["structures"][name] = entry
        print(name, f"a {medians['a']:.4f} ms b {medians['b']:.4f} ms b/a {entry['ratio_b_over_a']:.4f}",
              f"stubs {stubbed['b']}", "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
