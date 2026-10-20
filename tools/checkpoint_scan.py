"""OPT-T04 checkpoint scan (plan opt-1, CPU only, report only): exact zeros and unused parts of MACE checkpoints.

For each checkpoint it loads the pickled MACE model on the CPU (no conversion, no rewrites) and records:

- every parameter and buffer that is entirely zero, and the share of exact zeros in the rest;
- 2-D weights with all-zero rows or columns;
- e3nn ``o3.Linear`` layers (``linear_up``, ``linear``, ``linear_2``, ``skip_tp``-free linears, readouts):
  instructions (input irrep -> output irrep paths) whose weight block is entirely zero;
- symmetric contractions: per element (species) and correlation order, weight blocks that are entirely zero,
  i.e. species whose product basis contributes nothing for that output irrep;
- heads: their number and names, the readout output widths and the per-head atomic energies, since a run
  uses one head and the others' readout columns are computed and thrown away;
- the species table (``atomic_numbers``).

It changes nothing; the JSON summary feeds ``docs/validation/opt-T04-checkpoint-scan.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, type=Path, help="Directory for summary.json")
    p.add_argument("--labels", nargs="*", default=None, help="Model labels (default: benchmark and MACE-OFF)")
    p.add_argument("--models", default=os.environ.get("FASTMACE_MODELS"), help="Directory of the checkpoints")
    return p.parse_args(argv)


def _zero_share(tensor):
    return float((tensor == 0).sum()) / max(tensor.numel(), 1)


def _linear_blocks(linear):
    """(path, zero) for each instruction of an e3nn o3.Linear with internal weights."""
    import torch

    weight = getattr(linear, "weight", None)
    instructions = getattr(linear, "instructions", None)
    if weight is None or instructions is None or not isinstance(weight, torch.Tensor):
        return []
    out, offset = [], 0
    irreps_in, irreps_out = linear.irreps_in, linear.irreps_out
    flat = weight.reshape(-1) if weight.dim() == 1 else weight.reshape(weight.shape[0], -1)[0]
    for ins in instructions:
        size = 1
        for n in ins.path_shape:
            size *= n
        block = flat[offset:offset + size]
        offset += size
        out.append({"path": f"{irreps_in[ins.i_in]} -> {irreps_out[ins.i_out]}", "size": size,
                    "zero": bool(torch.count_nonzero(block) == 0), "zero_share": _zero_share(block)})
    return out


def _symmetric_blocks(module, elements):
    """Per contraction (output irrep) and weight tensor: elements whose block is entirely zero."""
    import torch

    out = []
    for c_index, contraction in enumerate(getattr(module, "contractions", [])):
        tensors = []
        if hasattr(contraction, "weights_max"):
            tensors.append(("max", contraction.weights_max))
        for k, w in enumerate(getattr(contraction, "weights", [])):
            tensors.append((f"weights.{k}", w))
        for name, w in tensors:
            if not isinstance(w, torch.Tensor) or w.dim() < 2:
                continue
            zero_elements = [elements[e] if e < len(elements) else e for e in range(w.shape[0])
                             if torch.count_nonzero(w[e]) == 0]
            out.append({"contraction": c_index, "weights": name, "shape": list(w.shape),
                        "zero_elements": zero_elements, "zero_share": _zero_share(w)})
    return out


def scan(model):
    import torch

    elements = [int(z) for z in getattr(model, "atomic_numbers", torch.tensor([])).tolist()]
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    zero_tensors, partial, zero_lines = [], [], []
    total = zeros = 0
    for name, t in tensors:
        if not t.is_floating_point() or t.numel() == 0:
            continue
        total += t.numel()
        z = int((t == 0).sum())
        zeros += z
        if z == t.numel():
            zero_tensors.append({"name": name, "shape": list(t.shape)})
        elif z:
            partial.append({"name": name, "shape": list(t.shape), "zero_share": z / t.numel()})
        if t.dim() == 2 and z and z != t.numel():
            rows = int((t == 0).all(dim=1).sum())
            cols = int((t == 0).all(dim=0).sum())
            if rows or cols:
                zero_lines.append({"name": name, "shape": list(t.shape), "zero_rows": rows, "zero_cols": cols})
    linears, symmetric = {}, {}
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind == "Linear" and hasattr(module, "instructions"):
            blocks = _linear_blocks(module)
            if blocks:
                linears[name] = {"paths": len(blocks), "zero_paths": [b["path"] for b in blocks if b["zero"]],
                                 "weights": sum(b["size"] for b in blocks)}
        elif kind == "SymmetricContraction":
            symmetric[name] = _symmetric_blocks(module, elements)
    heads = list(getattr(model, "heads", []) or [])
    readouts = []
    for index, readout in enumerate(getattr(model, "readouts", [])):
        irreps = [str(getattr(m, "irreps_out", "")) for _, m in readout.named_modules() if hasattr(m, "irreps_out")]
        readouts.append({"index": index, "type": type(readout).__name__, "irreps_out": irreps[-1] if irreps else ""})
    energies = getattr(model, "atomic_energies_fn", None)
    e0 = getattr(energies, "atomic_energies", None)
    return {
        "type": type(model).__name__,
        "elements": elements,
        "heads": heads,
        "readouts": readouts,
        "atomic_energies_shape": list(e0.shape) if e0 is not None else None,
        "parameters": total,
        "zero_share": zeros / max(total, 1),
        "zero_tensors": zero_tensors,
        "partially_zero": sorted(partial, key=lambda e: -e["zero_share"])[:40],
        "zero_rows_or_cols": zero_lines,
        "linears": linears,
        "symmetric_contractions": symmetric,
    }


def main(argv=None):
    args = parse_args(argv)
    import torch

    from fastmace.model import labels as model_labels

    names = args.labels or [*model_labels.BENCHMARK_LABELS, *model_labels.MACE_OFF_LABELS]
    checkpoints = {}
    for label in names:
        entry = model_labels.LABELS[label]
        checkpoints.setdefault(entry.checkpoint.file_name, []).append(label)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for file_name, used_by in checkpoints.items():
        path = Path(args.models) / file_name
        if not path.exists():
            results[file_name] = {"labels": used_by, "missing": str(path)}
            print(f"{file_name}: missing", flush=True)
            continue
        model = torch.load(path, map_location="cpu", weights_only=False)
        out = scan(model)
        out["labels"] = used_by
        results[file_name] = out
        print(f"{file_name}: {out['parameters']} parameters, zero share {out['zero_share']:.2e}, "
              f"{len(out['zero_tensors'])} zero tensors, heads {out['heads']}", flush=True)
        del model
    summary = {"tool": "checkpoint_scan", "created_utc": datetime.now(timezone.utc).isoformat(),
               "torch": torch.__version__, "python": sys.executable, "checkpoints": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
