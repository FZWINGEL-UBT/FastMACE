"""Controlled MD A/B screen of the existing receiver backward or PyTorch cuBLASLt.

Run under tools/ci/local_gpu.py with FASTMACE_STRUCTURES set to an export holding
Structures/Benchmark/R2SCAN. Each side has its own model and graph engine. The
candidate is selected before model build and graph capture; launch counts verify
that each receiver comparison captured the intended backward kernel. This screen
is not a production dispatch or a force/stress qualification.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import fastmace  # noqa: F401 -- before torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("receiver", "cublaslt"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structures", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--block", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--order", choices=("ab", "ba"), default="ab")
    parser.add_argument("--lt-workspace-mib", type=int, default=32)
    args = parser.parse_args()
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from torch._dynamo.utils import counters

    from fastmace.bench import ab, step_time
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.kernels.oeq_streaming.runtime import StreamingKernels
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload
    from fastmace.rewrites.mh1 import modules
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    if ab_step_time.gpu_processes():
        raise RuntimeError("GPU must be idle before benchmark")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine("mh1-r2scan", checkpoint=str(args.checkpoint), sha256=sha)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    original_attach = modules.attach_streaming_kernels
    original_edge, original_receiver = StreamingKernels.backward, StreamingKernels.backward_receiver
    launches = {}

    def counted(method, name):
        def call(self, *values):
            key = (self.id, name)
            launches[key] = launches.get(key, 0) + 1
            return method(self, *values)
        return call

    StreamingKernels.backward = counted(original_edge, "edge")
    StreamingKernels.backward_receiver = counted(original_receiver, "receiver")

    old_lt_workspace = torch.backends.cuda.cublaslt_workspace_size()
    if args.candidate == "cublaslt":
        torch.backends.cuda.cublaslt_workspace_size(args.lt_workspace_mib * 1024**2)

    def set_blas(side):
        if args.candidate == "cublaslt":
            torch.backends.cuda.preferred_blas_library("cublaslt" if side == "b" else "cublas")

    record = dict(candidate=args.candidate, model="mh1-r2scan", checkpoint_sha256=sha,
                  created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                  python=sys.executable, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, fastmace_commit=ab_step_time.fastmace_commit(),
                  packages={name: version(name) for name in ("mace-torch", "openequivariance",
                                                           "cuequivariance", "triton", "ase")},
                  backend="hybrid", precision="float32", tf32=False, temperature_K=300,
                  cublaslt_workspace_mib=args.lt_workspace_mib if args.candidate == "cublaslt" else None,
                  order=args.order, pattern="abba", warmup=args.warmup, block=args.block,
                  blocks=args.blocks, structures={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for spec in args.structures:
            name, atoms = ab_step_time.read_structure(spec, None)
            sides, ids, capture_calls, setup = {}, {}, {}, {}
            for side in args.order:
                set_blas(side)
                def attach(interaction, *, target=side):
                    original_attach(interaction)
                    interaction.streaming_kernels.receiver_backward = (
                        args.candidate == "receiver" and target == "b")
                modules.attach_streaming_kernels = attach
                t0 = time.perf_counter()
                field = ForceField(loaded, atoms, precision="float32", rewrites="auto")
                integrator = IntegratorConfig(temperature_K=300, seed=20261026)
                plan = planner.plan(field, atoms, integrator, Workload(max_temperature_K=300), hardware)
                engine = planner.build(plan, field, atoms.copy(), temperature_K=300)
                if not engine.capture:
                    raise RuntimeError("The screen requires a graph-captured engine")
                ks = [getattr(interaction, "streaming_kernels", None)
                      for interaction in engine.force_field.model.interactions]
                ks = [k for k in ks if k is not None]
                if not ks:
                    raise RuntimeError("No streaming kernels on the execution model")
                ids[side] = [k.id for k in ks]
                expected = args.candidate == "receiver" and side == "b"
                if any(k.receiver_backward != expected for k in ks):
                    raise RuntimeError(f"Wrong backward selection on side {side}")
                before = dict(launches)
                engine.prepare()
                kind = "receiver" if expected else "edge"
                delta = sum(launches.get((k.id, kind), 0) - before.get((k.id, kind), 0) for k in ks)
                capture_calls[side] = dict(kind=kind, launches=delta, graph_keys=list(map(str, engine._graphs)))
                if delta < 1:
                    raise RuntimeError(f"No {kind} backward launches observed while capturing side {side}")
                sides[side] = ab_step_time.FastMACESide(engine, torch)
                setup[side] = time.perf_counter() - t0
            modules.attach_streaming_kernels = original_attach
            if set(ids["a"]) & set(ids["b"]):
                raise RuntimeError("The two models share streaming kernel instances")
            a, b = sides["a"].engine.state, sides["b"].engine.state
            force_difference = float((a.forces - b.forces).abs().max().item())
            energy_difference = float((a.energy - b.energy).abs().max().item())
            for side in args.order:
                set_blas(side)
                step_time.warm_up(sides[side], args.warmup)
            before_counters = ab_step_time.compile_counters(counters)
            before_runtime = {s: sides[s].engine.runtime_counters() for s in sides}
            protocol = step_time.TimingProtocol(warmup=0, block=args.block, blocks=args.blocks)
            result = ab.run_ab({"a": sides["a"], "b": sides["b"]}, protocol, pattern="abba", before=set_blas)
            changes, recompiled = ab_step_time.compile_changes(before_counters, ab_step_time.compile_counters(counters))
            other = [pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()]
            after_runtime = {s: sides[s].engine.runtime_counters() for s in sides}
            timed_captures = {s: after_runtime[s]["captures"] - before_runtime[s]["captures"] for s in sides}
            item = dict(spec=spec, natoms=len(atoms), ab=result.as_dict(), setup_s=setup,
                        capture_calls=capture_calls, initial_max_force_difference_eV_per_A=force_difference,
                        initial_energy_difference_eV=energy_difference,
                        compile_during_timing=dict(changes=changes, recompiled=recompiled),
                        captures_during_timing=timed_captures,
                        observables={s: sides[s].observables() for s in sides}, other_gpu_processes=other,
                        valid=not recompiled and not other and not any(timed_captures.values()))
            record["structures"][spec] = item
            args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")
            print(f"{args.candidate} {name} {spec} atoms={len(atoms)} b/a={result.ratio:.5f} "
                  f"a={result.statistics['a'].median_ms:.4f}ms b={result.statistics['b'].median_ms:.4f}ms "
                  f"valid={item['valid']} capture={capture_calls}", flush=True)
            sides.clear()
            del engine, field, a, b, ks
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        modules.attach_streaming_kernels = original_attach
        StreamingKernels.backward, StreamingKernels.backward_receiver = original_edge, original_receiver
        if args.candidate == "cublaslt":
            torch.backends.cuda.preferred_blas_library("cublas")
            torch.backends.cuda.cublaslt_workspace_size(old_lt_workspace)


if __name__ == "__main__":
    main()
