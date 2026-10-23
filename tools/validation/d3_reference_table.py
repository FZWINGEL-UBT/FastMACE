"""Phase 3b acceptance item 1 as numbers: the D3 kernel against torch-dftd and simple-dftd3 (3b-T10).

The tests (tests/cpu/test_d3_kernel.py, tests/gpu/test_d3_kernel_gpu.py) assert the runbook's item 1 rule; this
tool records the deviations behind it for the report. On the six rattled and strained fixtures
(``_support.d3_cases``), for each device given (cpu, cuda):

- the two references' agreement (energy per atom, max force component, max stress component);
- float64 against each reference, with the rule's bound 2 x agreement + floor and pass/fail;
- the negative control without pass 3 (the CN chain term) against torch-dftd: its force deviation must exceed
  the bound;
- float32 against float64 on the same device, against the gate's floors.

    PYTHONPATH=<checkout>:<checkout>/tests python tools/validation/d3_reference_table.py --output DIR [--devices cpu cuda]

Writes DIR/d3_references.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "cpu"))

from _support.d3_cases import CASES, FLOORS, GATE, deviations, within  # noqa: E402
from test_d3_kernel import RECORD, evaluate  # noqa: E402


def references():
    from fastmace.reference.d3 import simple_dftd3, torch_dftd3

    out = {}
    for name, make in CASES.items():
        atoms = make()
        first, second = torch_dftd3(atoms, RECORD), simple_dftd3(atoms, RECORD)
        out[name] = (first, second, deviations(first, second, len(atoms)))
    return out


def bound(agreement):
    return [2.0 * a + f for a, f in zip(agreement, FLOORS)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=("cpu", "cuda"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    import dftd3
    import torch_dftd

    record = dict(tool="d3_reference_table", created_utc=datetime.now(timezone.utc).isoformat(),
                  torch=torch.__version__, torch_dftd=getattr(torch_dftd, "__version__", None),
                  dftd3=getattr(dftd3, "__version__", None),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  floors=FLOORS, gate=GATE, cases={})
    refs = references()
    passed = True
    for name, make in CASES.items():
        atoms = make()
        first, second, agreement = refs[name]
        entry = dict(atoms=len(atoms), agreement=agreement, bound=bound(agreement), devices={})
        for device in args.devices:
            ours = evaluate(atoms, device=device)
            single = evaluate(atoms, device=device, dtype=torch.float32)
            chainless = evaluate(atoms, device=device, chain=False)
            row = dict(torch_dftd=deviations(ours, first, len(atoms)), simple_dftd3=deviations(ours, second, len(atoms)),
                       no_chain_vs_torch_dftd=deviations(chainless, first, len(atoms)),
                       float32_vs_float64=deviations(single, ours, len(atoms)))
            row["pass_torch_dftd"] = within(row["torch_dftd"], agreement)
            row["pass_simple_dftd3"] = within(row["simple_dftd3"], agreement)
            row["negative_control_fails"] = row["no_chain_vs_torch_dftd"][1] > entry["bound"][1]
            row["pass_float32_gate"] = all(d <= g for d, g in zip(row["float32_vs_float64"], GATE))
            row["pass"] = (row["pass_torch_dftd"] and row["pass_simple_dftd3"] and row["negative_control_fails"]
                           and row["pass_float32_gate"])
            passed &= row["pass"]
            entry["devices"][device] = row
        record["cases"][name] = entry
    record["pass"] = passed
    (args.output / "d3_references.json").write_text(json.dumps(record, indent=1) + "\n")
    print("case device | agreement E F S | vs torch-dftd E F S | vs dftd3 E F S | no-chain F | f32 E F S | pass")
    for name, entry in record["cases"].items():
        for device, row in entry["devices"].items():
            fmt = lambda v: " ".join(f"{x:.1e}" for x in v)  # noqa: E731
            print(f"{name} {device} | {fmt(entry['agreement'])} | {fmt(row['torch_dftd'])} | {fmt(row['simple_dftd3'])}"
                  f" | {row['no_chain_vs_torch_dftd'][1]:.1e} | {fmt(row['float32_vs_float64'])} | {row['pass']}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
