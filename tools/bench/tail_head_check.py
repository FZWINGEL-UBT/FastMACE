"""Readout head slice (``fastmace.rewrites.species.TAIL_HEAD`` prototype): energies and forces against the default.

``fold_readout_tail`` keeps all 96 hidden columns of MH-1's last readout (6 heads of 16). With ``TAIL_HEAD`` set to
the active head's index the folded products keep only that head's 16 columns. This evaluates the default float32
plan (``ForceField.evaluate``) with ``TAIL_HEAD`` None and with the head's index on the same rattled cells and
reports the largest energy-per-atom and force differences. Rounding-class means differences at float32 noise
(compare: the hybrid default against itself is not bitwise reproducible either). Measurement only; idle GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import ab_step_time  # noqa: E402
from step_breakdown import sha256  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--head", type=int, default=0, help="Index of the model's head (mh1-r2scan: 0)")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NbCl5", "NaNbO3"])
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import numpy as np

    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.rewrites import species

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.expanduser()
    loaded = loading.load_pristine(args.model, checkpoint=str(checkpoint), sha256=sha256(checkpoint))
    results = {}
    for name in args.structures:
        _, atoms = ab_step_time.read_structure(name, None)
        atoms.rattle(0.05, seed=3)
        out = {}
        for label, head in (("default", None), ("sliced", args.head)):
            species.TAIL_HEAD = head
            field = ForceField(loaded, atoms, precision="float32")
            energy, forces, _ = field.evaluate(atoms)
            out[label] = (float(energy), np.asarray(forces, dtype=np.float64))
            field = None
        species.TAIL_HEAD = None
        de = abs(out["sliced"][0] - out["default"][0]) / len(atoms)
        df = float(np.abs(out["sliced"][1] - out["default"][1]).max())
        results[name] = dict(atoms=len(atoms), energy_default_eV=out["default"][0], energy_sliced_eV=out["sliced"][0],
                             d_energy_per_atom_eV=de, max_d_force_eV_A=df,
                             max_force_eV_A=float(np.abs(out["default"][1]).max()))
        print(name, json.dumps(results[name]), flush=True)
    (args.output / "tail_head.json").write_text(json.dumps(dict(
        tool="tail_head_check", created_utc=datetime.now(timezone.utc).isoformat(), head=args.head,
        fastmace_commit=ab_step_time.fastmace_commit(), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
