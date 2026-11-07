"""Bound for a POLAR-1 megakernel from local profiles (docs/design/research/2026-11-07-polar-megakernel.md).

Usage: python tools/bench/polar_profile_bound.py <profile-dir> <run> [<run> ...], where each run has
<run>.step.log and <run>/module/sources.csv (tools/module_profile.py output).
"""
import csv, json, sys, collections, os, ast
base = os.path.expanduser(sys.argv[1])
def cat(r):
    op, k, d = r["op"], r["kernel"], r["input_dims"]
    if "streaming_conv" in op or "oeq_" in k or "edge_outer" in op or "radial_table" in op or "uniform_1d" in op or "segmented_polynomial" in k:
        return "conv/radial/TP"
    if op in ("aten::mm", "aten::addmm", "aten::bmm"):
        try:
            dims = ast.literal_eval(d)
            if op == "aten::addmm": dims = dims[1:]
            (m, kk), (_, n) = dims[0], dims[1]
        except Exception:
            return "gemm-other"
        if "splitK" in k or "Memset" in k: return "gemm-narrow(splitK/memset)"
        if kk >= 256 and n >= 256: return "gemm-wide(K,N>=256)"
        return "gemm-narrow"
    return "pointwise/reduce(triton etc)"
for name in sys.argv[2:]:
    rows = list(csv.DictReader(open(f"{base}/{name}/module/sources.csv")))
    step = [json.loads(l) for l in open(f"{base}/{name}.step.log") if l.startswith("{")][0]
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        c = cat(r); agg[c][0] += float(r["us_per_call"]); agg[c][1] += float(r["launches_per_call"])
    span = step["traced_span_ms_per_step"] * 1000
    print(f"== {name}: wall {step['wall_ms_per_step']} ms, span {span/1000:.3f} ms, gaps {step['idle_gaps_ms_per_step']} ms ({100*step['idle_gaps_ms_per_step']*1000/span:.1f}%), kernels {step['kernels_per_step']}")
    for c, (us, n) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"   {c:32s} {us:8.1f} us {100*us/span:5.1f}%  {n:5.0f} launches")
