"""Load every label of the table once (plain cuEq, float64, cuda:0) and record hash, licence and heads.

    FASTMACE_MODELS=<store> python labels_probe.py OUT.json
"""

import json
import sys
import time
import warnings

import fastmace  # noqa: F401  (before torch)
from fastmace.model import labels, loading

rows = []
for label in labels.LABELS:
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", loading.ModelLicenceWarning)
        loaded = loading.load(label, backend="cueq", dtype="float64", device="cuda:0")
    shown = [str(w.message) for w in caught if issubclass(w.category, loading.ModelLicenceWarning)]
    rows.append({"label": label, "file": loaded.path.name, "sha256": loaded.sha256, "trust": loaded.trust,
                 "licence": loaded.licence, "licence_warning": shown, "head": loaded.head, "heads": list(loaded.heads),
                 "model_class": type(loaded.model).__name__, "model_type": loaded.model_type,
                 "functional": loaded.functional, "fused_cueq": loaded.fused_cueq,
                 "cueq_conv_fusion": loaded.cueq_conv_fusion, "embedding_specs": loaded.embedding_specs,
                 "load_s": round(time.perf_counter() - start, 2)})
    print(json.dumps({k: rows[-1][k] for k in ("label", "sha256", "licence", "head", "model_class", "load_s")}),
          flush=True)
    del loaded
with open(sys.argv[1], "w") as handle:
    json.dump(rows, handle, indent=1)
