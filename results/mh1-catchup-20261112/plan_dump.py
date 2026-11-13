"""Which plan MH-1 gets on the H200 and the A100 family today, next to OMAT-0 medium (fastmace/mh1-catchup, CPU only).

The fixed planner with the CPU tests' stand-in traits (tests/cpu/test_planner.py) and the hardware name set: every
RewriteConfig field and the planner's reason when "auto" turned it off. Writes plan_dump.json.

    CUDA_VISIBLE_DEVICES='' python results/mh1-catchup-20261112/plan_dump.py
"""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "cpu"), str(ROOT / "tests"), str(ROOT)]

from fastmace.config import IntegratorConfig  # noqa: E402
from fastmace.planner.plan import Hardware  # noqa: E402
from test_planner import nacl, planner, traits  # noqa: E402

GPUS = {
    "H200": Hardware("NVIDIA H200", (9, 0), 141 * 2**30),
    "A100-SXM4-80GB": Hardware("NVIDIA A100-SXM4-80GB", (8, 0), 80 * 2**30),
    "A100 80GB PCIe": Hardware("NVIDIA A100 80GB PCIe", (8, 0), 80 * 2**30),
    "Ada": Hardware("NVIDIA RTX 4000 Ada Generation", (8, 9), 20 * 2**30),
}
LABELS = ("mh1-r2scan", "mh1-omat", "mh1-mp", "omat0-medium")


def value(v):
    return v if isinstance(v, (bool, int, float, str, type(None))) else str(v)


def main():
    out = {}
    for label in LABELS:
        for gpu, hw in GPUS.items():
            plan = planner().plan(traits(label), nacl(), IntegratorConfig(), hw, model_label=label)
            rw = {f.name: value(getattr(plan.rewrites, f.name)) for f in dataclasses.fields(plan.rewrites)}
            nb = {f.name: value(getattr(plan.neighbours, f.name)) for f in dataclasses.fields(plan.neighbours)}
            out[f"{label}|{gpu}"] = dict(rewrites=rw, edge_index_int32=nb.get("edge_index_int32"),
                                        auto_off=dict(plan.auto_options_off))
            print(label, gpu, sorted(k for k, v in rw.items() if v in (True, "on")), flush=True)
    Path(__file__).with_name("plan_dump.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
