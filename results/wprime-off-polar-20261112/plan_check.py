"""Which OFF / POLAR plans run the w' kernel on the H200 and the A100 family (fastmace/wprime-off-polar, CPU only).

Uses the CPU planner tests' stand-in traits (tests/cpu/test_planner.py) and asks the fixed planner for each label's
"auto" plan with the hardware name set. Writes plan_check.json: per label and GPU, the resolved wprime_backward /
fused_folded_forward and the planner's reason.

    CUDA_VISIBLE_DEVICES='' python results/wprime-off-polar-20261112/plan_check.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "cpu"), str(ROOT / "tests")]

from fastmace.config import IntegratorConfig  # noqa: E402
from fastmace.planner.plan import Hardware  # noqa: E402
from test_planner import nacl, planner, traits  # noqa: E402

GPUS = {
    "H200": Hardware("NVIDIA H200", (9, 0), 141 * 2**30),
    "A100-SXM4-80GB": Hardware("NVIDIA A100-SXM4-80GB", (8, 0), 80 * 2**30),
    "A100 80GB PCIe": Hardware("NVIDIA A100 80GB PCIe", (8, 0), 80 * 2**30),
    "A100-SXM4-40GB": Hardware("NVIDIA A100-SXM4-40GB", (8, 0), 40 * 2**30),
}
LABELS = ("off23-small", "off23-medium", "off23-large", "off23b-medium", "off24-medium",
          "polar1-s", "polar1-m", "polar1-l", "omat0-medium", "mh1-r2scan")
FIELDS = ("wprime_backward", "fused_folded_forward", "radial_tables")


def main():
    out = {}
    for label in LABELS:
        for gpu, hw in GPUS.items():
            plan = planner().plan(traits(label), nacl(), IntegratorConfig(), hw, model_label=label)
            row = {f: getattr(plan.rewrites, f) for f in FIELDS}
            off = dict(plan.auto_options_off)
            row["why_off"] = {f: off[f] for f in FIELDS if f in off}
            out[f"{label}|{gpu}"] = row
            print(label, gpu, {k: row[k] for k in FIELDS}, flush=True)
    Path(__file__).with_name("plan_check.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
