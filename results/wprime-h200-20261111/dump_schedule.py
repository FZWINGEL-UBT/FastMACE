"""Dump interaction 2's schedule terms (kernel coefficients, FOLD_ALPHA alphas) to schedule_terms.json (CPU)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from render_sources import problem
from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels

schedule, _ = render_kernels(problem(), np.float32, NUM_THREADS, "ir_mul")
alphas = [fp["alpha"] for fp in wp.factored_paths(schedule, True)]
paths = []
for path, alpha in zip(schedule.paths, alphas):
    outs = []
    for o, out in enumerate(path.outputs):
        outs.append([dict(o=o, i=int(t.input_component), j=int(t.edge_component), c=float(t.coefficient))
                     for t in out.terms])
    paths.append(dict(input_start=int(path.input_start), input_dim=int(path.input_irrep_dim),
                      edge_start=int(path.edge_start), edge_dim=int(path.edge_irrep_dim),
                      weight_start=int(path.weight_start), alpha=alpha, outputs=outs))
(HERE / "schedule_terms.json").write_text(json.dumps(dict(channels=int(schedule.channels), paths=paths), indent=1))
print(len(paths), "paths", sum(len(t) for p in paths for t in p["outputs"]), "terms", alphas)
