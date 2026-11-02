"""Rule 12's terms for matrix cases: python terms.py FM_SOURCE CASES.json STRUCTURE ATOMS"""
import json
import sys
from pathlib import Path

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM))
from fastmace.model.capabilities import ModelDims  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402

FIT = json.loads((FM / "tests/_support/data/rule12_fit.json").read_text())
registry = minimal._rewrite_registry()
for c in json.loads(Path(sys.argv[2]).read_text()):
    if c["structure"] != sys.argv[3] or c["natoms"] != int(sys.argv[4]):
        continue
    dims = ModelDims.from_record(FIT["dims"][c["label"]])
    dt = c["dtype"]
    rewrite_terms = memory.rewrite_memory_terms(registry, dims, c["rewrites"], dt)
    est = memory.estimate(dims, edges=c["edges"], nodes=c["nodes"], dtype=dt, capacity=c["capacity"], list_dtype=dt,
                          fused_conv=c["fused_conv"], rewrite_terms=rewrite_terms,
                          selfcheck_terms=memory.selfcheck_memory_terms(registry, dims, c["selfchecks"], dt, c["sms"]),
                          resident_terms=memory.resident_memory_terms(registry, dims, c["resident"], dt))
    print(c["stage"], "rewrites", c["rewrites"])
    print("   terms", [(n, round(v / 2**30, 2)) for n, v in est.terms], "rewrite terms",
          [(n, round(v / 2**30, 2)) for n, v in est.rewrite_terms], "total", round(est.total / 2**30, 2),
          "peak", round(c["peak_bytes"] / 2**30, 2))
    print("   per edge", [(n, t.edge_bytes, t.node_bytes) for n, t in rewrite_terms])
