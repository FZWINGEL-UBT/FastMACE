"""OPT10-T04 A/B: evict-first (streaming) cache hints on the per-edge weights ``w`` of the streaming kernels.

``w [E, 1280]`` is written once by ``fused_folded_forward`` and read once by the convolution backward; both
pass through the L2 and may evict data that is reused (the chunked backward's ``dw`` slots, ``x``, ``dout``).
Side ``b`` compiles its kernels with ``__stcs`` on the fused forward's ``w`` stores and ``__ldcs`` on the
streaming kernels' ``w`` loads (the values are unchanged: exact). It is built by the minimal planner like side
``a``, in the same process, while ``runtime.compile_cubin`` rewrites the source; the number of substitutions is
recorded and a structure is invalid unless both kinds were made. Timed blocks alternate
(``fastmace.bench.ab.run_ab``). Writes ``<output>.json``. Use an idle GPU.

Usage: python tools/bench/opt10_cache_hints.py --output OUT --checkpoint mace-mh-1.model [--structures ...]
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

HINTS = {"active": False, "loads": 0, "stores": 0}
LOAD = "const scalar_t radial_weight = weights[radial_weight_index];"
LOAD_CS = "const scalar_t radial_weight = __ldcs(&weights[radial_weight_index]);"
STORE = "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];"
STORE_CS = "__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);"


def install_hints():
    from fastmace.kernels.oeq_streaming import runtime

    real = runtime.compile_cubin

    def compile_cubin(source, capability, extra_options=()):
        if HINTS["active"]:
            HINTS["loads"] += source.count(LOAD)
            HINTS["stores"] += source.count(STORE)
            source = source.replace(LOAD, LOAD_CS).replace(STORE, STORE_CS)
        return real(source, capability, extra_options)

    runtime.compile_cubin = compile_cubin


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaNbO3", "NbCl5"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--pattern", choices=("abab", "abba"), default="abab")
    p.add_argument("--order", choices=("ab", "ba"), default="ab")
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

    install_hints()
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    protocol = step_time.TimingProtocol(warmup=args.warmup, block=args.block, blocks=args.blocks)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    meta = dict(tool="opt10_cache_hints", label=args.label, created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, checkpoint_sha256=sha,
                protocol=protocol.as_dict(), pattern=args.pattern, order=args.order, structures={})
    for spec in args.structures:
        name, base = ab_step_time.read_structure(spec, None)
        sides, setup, substituted = {}, {}, {}
        for side in args.order:
            HINTS.update(active=side == "b", loads=0, stores=0)
            started = time.perf_counter()
            field = ForceField(loaded, base, precision="float32", rewrites="auto")
            integrator = IntegratorConfig(temperature_K=args.temperature_K)
            plan = planner.plan(field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
            engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
            sides[side] = ab_step_time.FastMACESide(engine, torch)
            setup[side] = time.perf_counter() - started
            substituted[side] = {k: HINTS[k] for k in ("loads", "stores")}
        HINTS["active"] = False
        for side in args.order:
            step_time.warm_up(sides[side], protocol.warmup)
        counted = ab_step_time.compile_counters(counters)
        result = ab.run_ab(sides, dataclasses.replace(protocol, warmup=0), pattern=args.pattern)
        changes, recompiled = ab_step_time.compile_changes(counted, ab_step_time.compile_counters(counters))
        medians = {side: result.statistics[side].median_ms for side in sides}
        blocks = {side: result.statistics[side].block_ms for side in sides}
        entry = dict(natoms=len(base), ab=result.as_dict(), ratio_b_over_a=medians["b"] / medians["a"],
                     pair_ratios=[b / a for a, b in zip(blocks["a"], blocks["b"])], setup_s=setup,
                     substituted=substituted, compile_during_timing=dict(changes=changes, recompiled=recompiled),
                     observables={side: sides[side].observables() for side in sides},
                     other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()])
        entry["valid"] = (not recompiled and not entry["other_gpu_processes"] and substituted["b"]["loads"] > 0
                          and substituted["b"]["stores"] > 0)
        meta["structures"][name] = entry
        print(name, f"a {medians['a']:.4f} ms b {medians['b']:.4f} ms b/a {entry['ratio_b_over_a']:.4f}",
              f"substituted {substituted['b']}", "valid" if entry["valid"] else "INVALID", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        sides = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
