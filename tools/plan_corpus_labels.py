"""The 19 labels as the planner sees them, without loading a checkpoint (plan task 5-T11; the plan corpus).

``LABEL_FACTS`` and ``stand_in`` came from ``tests/cpu/test_planner.py`` (1a-T15), which imports them
from here: one copy for the tests and ``tools/plan_corpus.py``. The facts were read from the released
checkpoints with ``fastmace.model.capabilities.model_traits`` (``~/models``, 18 October 2026;
``tests/gpu/test_planner_gpu.py`` re-reads them when ``FASTMACE_MODELS`` is set). The traits only look at
class names and attributes, so the stand-ins give the checkpoints' verdicts.

``dims(label)``: the model's sizes (channels, irreps, radial basis, interactions, parameters), read once
from each checkpoint by ``tools/plan_corpus_capture.py`` into ``tools/plan_corpus_dims.json`` with the
file's sha256. Rule 12's estimate (SPEC §9.3, decision 51) reads them; ``model_traits`` does not.
"""

from __future__ import annotations

import json
import types
from functools import lru_cache
from pathlib import Path

DIMS_FILE = Path(__file__).resolve().with_name("plan_corpus_dims.json")

MH1 = "RealAgnosticResidualNonLinearInteractionBlock"
RESIDUAL = "RealAgnosticResidualInteractionBlock"
DENSITY_FIRST, DENSITY = "RealAgnosticDensityInteractionBlock", "RealAgnosticDensityResidualInteractionBlock"
PLAIN = "RealAgnosticInteractionBlock"
PER_GRAPH = (("total_charge", "graph"), ("total_spin", "graph"))

# label -> (model class, interaction classes, apply_cutoff, joint embeddings), read from the released
# checkpoints (model_traits; every node embedding is MACE's LinearNodeEmbeddingBlock).
LABEL_FACTS = {
    "mp0-medium": ("ScaleShiftMACE", (RESIDUAL, RESIDUAL), True, ()),
    "mp0-large": ("ScaleShiftMACE", (RESIDUAL, RESIDUAL), True, ()),
    "mp0b3-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "mpa0-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "omat0-medium": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "matpes-pbe": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "matpes-r2scan": ("ScaleShiftMACE", (DENSITY_FIRST, DENSITY), True, ()),
    "mh1-omat": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "mh1-mp": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "mh1-r2scan": ("ScaleShiftMACE", (MH1, MH1), False, ()),
    "polar1-s": ("PolarMACE", (MH1,), False, ()),
    "polar1-m": ("PolarMACE", (MH1, MH1), False, ()),
    "polar1-l": ("PolarMACE", (MH1, MH1, MH1), False, ()),
    "omol0-xl": ("ScaleShiftMACE", (MH1, MH1, MH1), False, PER_GRAPH),
    "off23-small": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23-large": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off23b-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
    "off24-medium": ("ScaleShiftMACE", (PLAIN, RESIDUAL), True, ()),
}
LABELS = tuple(LABEL_FACTS)


def stand_in(label):
    """An object with the attributes model_traits reads, as the released checkpoint of ``label`` has them."""
    model_class, interactions, apply_cutoff, joint = LABEL_FACTS[label]
    model = type(model_class, (), {})()
    model.interactions = [type(name, (), {})() for name in interactions]
    model.radial_embedding = types.SimpleNamespace(apply_cutoff=apply_cutoff)
    model.node_embedding = type("LinearNodeEmbeddingBlock", (), {})()
    if joint:
        model.joint_embedding = object()
        model.embedding_specs = {name: {"per": per} for name, per in joint}
    return model


@lru_cache(maxsize=1)
def _dims_file() -> dict:
    return json.loads(DIMS_FILE.read_text()) if DIMS_FILE.exists() else {}


def dims(label) -> dict | None:
    """The sizes read from ``label``'s checkpoint (``plan_corpus_capture.read_dims``), or ``None`` if not captured."""
    entry = _dims_file().get(label)
    return None if entry is None else entry["dims"]


def checkpoint_sha256(label) -> str | None:
    """The sha256 of the checkpoint file the sizes were read from."""
    entry = _dims_file().get(label)
    return None if entry is None else entry["sha256"]
