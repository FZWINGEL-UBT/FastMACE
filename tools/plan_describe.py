"""The default plan of each model on this GPU, as ``plan.describe(full=True)`` (fastmace/off-polar-alex).

Loads each label's checkpoint, builds its ForceField for one structure and asks the minimal planner (hybrid
backend, as tools/ab_rewrites.py) for the NVT plan on the visible GPU. Writes ``<output>/<label>.txt`` and
prints each plan's rewrite lines. Builds no engine:

    python tools/plan_describe.py --output runs/plans --models off23-small:MACE-OFF23_small.model \
        --structure $ORG/water_216.extxyz@1x1x1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, metavar="LABEL:CHECKPOINT",
                   help="checkpoint relative to FASTMACE_MODELS or absolute")
    p.add_argument("--structure", required=True)
    p.add_argument("--temperature-K", type=float, default=300.0)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    args.output.mkdir(parents=True, exist_ok=True)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = planner.hardware()
    _, base = ab_step_time.read_structure(args.structure, None)
    for item in args.models:
        label, _, ckpt = item.partition(":")
        path = Path(ckpt) if os.path.isabs(ckpt) else Path(os.environ.get("FASTMACE_MODELS", ".")) / ckpt
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = loading.load_pristine(label, checkpoint=str(path), sha256=sha)
        field = ForceField(loaded, base, precision="float32", rewrites="auto")
        plan = planner.plan(field, base, IntegratorConfig(temperature_K=args.temperature_K),
                            Workload(max_temperature_K=args.temperature_K), hardware)
        text = (f"# {label} {path.name} sha256 {sha}\n# gpu {torch.cuda.get_device_name()} natoms {len(base)}\n"
                + plan.describe(full=True) + "\n")
        (args.output / f"{label}.txt").write_text(text)
        print(f"== {label}", flush=True)
        print("\n".join(line for line in text.splitlines() if "RewriteConfig." in line or "backend" in line),
              flush=True)


if __name__ == "__main__":
    main()
