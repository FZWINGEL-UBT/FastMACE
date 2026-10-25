"""Shared command-line options of the examples: --checkpoint, --steps, --structure, --output, --seed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_ROOT = Path(os.path.expanduser("~/tmp/fastmace-examples"))


def parser(description: str, *, name: str, steps: int, structure: str | None = None) -> argparse.ArgumentParser:
    """An ``ArgumentParser`` with the options every example takes."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", metavar="PATH", default=None,
                   help="the MH-1 checkpoint file (mace-mh-1.model); default: $FASTMACE_MODELS or the cache")
    p.add_argument("--steps", type=int, default=steps, help="MD or optimiser steps (small for a smoke run)")
    if structure is not None:
        p.add_argument("--structure", metavar="FILE", default=structure, help="input structure, any format ASE reads")
    p.add_argument("--output", metavar="DIR", default=str(DEFAULT_ROOT / name), help="output directory")
    p.add_argument("--seed", type=int, default=1, help="noise seed")
    return p


def load_mh1(checkpoint: str | None, head: str = "matpes_r2scan"):
    """MH-1 with ``head``; ``checkpoint`` is checked against the label table's SHA-256."""
    import fastmace as fm

    if checkpoint is None:
        return fm.load("mh1", head=head)
    from fastmace.model.loading import MODELS_VARIABLE, load_pristine

    # Child processes (the gate's and MC's reference calculators) find the label's file through
    # $FASTMACE_MODELS, not through the explicit path: point it at the checkpoint's directory.
    path = Path(os.path.expanduser(checkpoint)).resolve()
    os.environ.setdefault(MODELS_VARIABLE, str(path.parent))
    return load_pristine("mh1", head, checkpoint=str(path))
