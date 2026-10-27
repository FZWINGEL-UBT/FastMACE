"""OPT7-T01 (a): what the sender-gradient atomics cost the edge-owned streaming backward (timing only).

Variants of the generated source, in oeq_projected_backward only:
- base: unchanged (atomicAdd into dx);
- racy: the same read-modify-write without atomics (*(dx + i) += v; wrong result under races, same traffic);
- drop: no dx at all (the compiler also drops its arithmetic): the ceiling of a dx that costs nothing.
Same process, one CUDA graph per variant, alternating rounds. Usage: python t01a.py ops-*.pt
"""
import dataclasses
import json
import re
import sys

import fastmace  # noqa: F401
import numpy as np
from openequivariance.core.e3nn_lite import TPProblem

import fastmace.kernels.oeq_streaming.runtime as runtime
from common import graph_time, load
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward

PATTERN = re.compile(r"atomicAdd\(\s*(dx \+ [^,]+?),\s*(scalar_t\([^;]*?\))\);")
original_render = runtime.render_kernels


def variant_source(source, mode):
    start = source.index("void oeq_projected_backward(")
    stop = source.index("void oeq_projected_backward_receiver(")
    body = source[start:stop]
    count = len(PATTERN.findall(body))
    if mode == "racy":
        body = PATTERN.sub(r"*(\1) += \2;", body)
    elif mode == "drop":
        body = PATTERN.sub(r"(void)(\2);", body)
    return source[:start] + body + source[stop:], count


def build(problem, layout, mode):
    def render(*args, **kwargs):
        schedule, kernel = original_render(*args, **kwargs)
        if mode == "base":
            return schedule, kernel
        source, _ = variant_source(kernel.jit_kernel, mode)
        return schedule, dataclasses.replace(kernel, jit_kernel=source)

    runtime.render_kernels = render
    try:
        return runtime.StreamingKernels(problem, np.float32, layout=layout)
    finally:
        runtime.render_kernels = original_render


for path in sys.argv[1:]:
    r = load(path)
    p = r["problem"]
    problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                        [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                         for i in p["instructions"]],
                        irrep_normalization="none", path_normalization="none", shared_weights=False,
                        internal_weights=False)
    _, kernel = original_render(problem, layout=p["layout"])
    _, atomics = variant_source(kernel.jit_kernel, "base")
    kernels = {mode: build(problem, p["layout"], mode) for mode in ("base", "racy", "drop")}
    x, sh, w, s, rcv, rp, dout = (r[k] for k in ("x", "sh", "w", "senders", "receivers", "row_ptr", "dout"))
    times = {mode: [] for mode in kernels}
    for _ in range(3):
        for mode, k in kernels.items():
            times[mode].append(graph_time(lambda k=k: streaming_conv_backward(x, sh, w, s, rcv, rp, dout, k.id)))
    row = dict(file=path, edges=int(sh.shape[0]), dx_atomic_sites=atomics,
               **{f"{mode}_ms": sorted(v)[1] for mode, v in times.items()})
    row["racy_over_base"] = row["racy_ms"] / row["base_ms"]
    row["drop_over_base"] = row["drop_ms"] / row["base_ms"]
    print(json.dumps(row), flush=True)
