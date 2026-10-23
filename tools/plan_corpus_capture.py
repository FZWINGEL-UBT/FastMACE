"""Read the dimensions of each released checkpoint once, on the CPU (plan task 5-T11; the plan corpus).

``tools/plan_corpus.py`` plans from stand-ins (``tools/plan_corpus_labels.py``), which carry only what
``model_traits`` reads. Rule 12's estimate (5-T03) also needs the model's sizes. This tool loads each
checkpoint in its own process (the large ones need several GB), reads the sizes generically from the
module tree, and writes them to ``tools/plan_corpus_dims.json`` with the file's sha256. It is a
superset of what an estimate may read; ``plan_corpus_labels.dims(label)`` returns the record.

Usage (WSL, CPU environment, check free memory first; not during a timed stretch)::

    FASTMACE_MODELS=~/models python tools/plan_corpus_capture.py [--labels a,b] [--out tools/plan_corpus_dims.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "plan_corpus_dims.json"
if str(ROOT) not in sys.path:  # run from a checkout, as on the GPU environments (AGENTS.md)
    sys.path.insert(0, str(ROOT))


def _irreps(value):
    return None if value is None else str(value)


def _dim(value):
    return None if value is None else int(getattr(value, "dim", 0))


def read_dims(model) -> dict:
    """The sizes of a loaded MACE model, read from its modules (no MACE-version-specific API)."""
    import torch

    def scalar(name):
        value = getattr(model, name, None)
        if value is None:
            return None
        return value.item() if isinstance(value, torch.Tensor) else value

    params = list(model.parameters())
    record = {
        "model_class": type(model).__name__,
        "r_max": float(scalar("r_max")),
        "num_interactions": int(scalar("num_interactions")),
        "num_elements": len(getattr(model, "atomic_numbers", ())),
        "heads": list(getattr(model, "heads", None) or ()),
        "parameters": sum(p.numel() for p in params),
        "parameter_dtype": str(params[0].dtype).replace("torch.", "") if params else None,
        "edge_attrs_irreps": _irreps(getattr(getattr(model, "spherical_harmonics", None), "irreps_out", None)),
        "num_bessel": getattr(getattr(model, "radial_embedding", None), "out_dim", None),
        "node_embedding_irreps": _irreps(getattr(getattr(getattr(model, "node_embedding", None), "linear", None),
                                                 "irreps_out", None)),
        "interactions": [],
        "products": [],
    }
    for block in getattr(model, "interactions", ()):
        conv_tp = getattr(block, "conv_tp", None)
        weights = getattr(block, "conv_tp_weights", None)
        record["interactions"].append({
            "class": type(block).__name__,
            "node_feats_irreps": _irreps(getattr(block, "node_feats_irreps", None)),
            "edge_feats_irreps": _irreps(getattr(block, "edge_feats_irreps", None)),
            "target_irreps": _irreps(getattr(block, "target_irreps", None)),
            "hidden_irreps": _irreps(getattr(block, "hidden_irreps", None)),
            "irreps_out": _irreps(getattr(block, "irreps_out", None)),
            "conv_tp_irreps_out": _irreps(getattr(conv_tp, "irreps_out", None)),
            "conv_tp_irreps_out_dim": _dim(getattr(conv_tp, "irreps_out", None)),
            "conv_tp_weight_numel": getattr(conv_tp, "weight_numel", None),
            "radial_mlp": list(getattr(weights, "hs", None) or ()) or None,
        })
    for product in getattr(model, "products", ()):
        contraction = getattr(product, "symmetric_contractions", None)
        inner = getattr(contraction, "contractions", None)
        first = inner[0] if inner else None
        record["products"].append({
            "irreps_out": _irreps(getattr(getattr(product, "linear", None), "irreps_out", None)),
            "correlation": getattr(first, "correlation", None) or getattr(contraction, "correlation", None),
        })
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _child(label: str) -> None:
    """Load one label's checkpoint on the CPU and print its record as JSON (one process per label)."""
    import torch

    from fastmace.model import labels

    entry = labels.LABELS[label]
    path = Path(os.environ["FASTMACE_MODELS"]).expanduser() / entry.checkpoint.file_name
    model = torch.load(path, map_location="cpu", weights_only=False)
    record = {"label": label, "file": entry.checkpoint.file_name, "sha256": _sha256(path),
              "sha256_expected": entry.checkpoint.sha256, "dims": read_dims(model)}
    print(json.dumps(record, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", default="", help="comma-separated labels (default: all 19)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        _child(args.child)
        return 0
    if not os.environ.get("FASTMACE_MODELS"):
        parser.error("set FASTMACE_MODELS to the directory with the released checkpoints")
    from fastmace.model import labels

    wanted = [name for name in args.labels.split(",") if name] or list(labels.LABELS)
    records = json.loads(args.out.read_text()) if args.out.exists() else {}
    for label in wanted:
        run = subprocess.run([sys.executable, __file__, "--child", label], capture_output=True, text=True,
                             cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if run.returncode != 0:
            print(f"{label}: failed\n{run.stderr[-2000:]}", file=sys.stderr)
            return 1
        record = json.loads(run.stdout.strip().splitlines()[-1])
        if record["sha256"] != record.pop("sha256_expected"):
            print(f"{label}: sha256 {record['sha256']} is not the release's", file=sys.stderr)
            return 1
        records[label] = record
        args.out.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")
        print(f"{label}: {record['dims']['model_class']}, {record['dims']['parameters']} parameters", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
