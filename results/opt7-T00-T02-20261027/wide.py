"""Wider loads in the edge-owned streaming backward (owner, 27 October 2026; after LunaTP's vector-width-2 result).

Source transform of the generated oeq_projected_backward (float32 only): the channel loop
    for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) { BODY }
becomes
    for (int pair = lane; pair < CHANNEL_DIM / 2; pair += LOGICAL_GROUP_SIZE) {
        { const int channel = 2 * pair;     BODY with float2 loads, the .y halves kept }
        { const int channel = 2 * pair + 1; BODY reading the kept halves }
    }
so every per-edge weight, output-gradient and sender-feature element is fetched by one 8-byte load for two adjacent
channels, and the weight gradients are stored as float2. Sender-gradient atomics stay scalar.
Usage: python wide.py ops-*.pt  (same-process A/B against the unchanged kernel, exactness first)
"""
import dataclasses
import json
import re
import sys

import fastmace  # noqa: F401
import numpy as np
import torch
from openequivariance.core.e3nn_lite import TPProblem

import fastmace.kernels.oeq_streaming.runtime as runtime
from common import graph_time, load, rel
from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward

LOOP = re.compile(r"for \(int channel = lane; channel < CHANNEL_DIM;\s*channel \+= LOGICAL_GROUP_SIZE\) \{")
LOAD = re.compile(r"= (weights|dout|x)\[(radial_weight_index|output_index|input_index)\];")
STORE = re.compile(r"dweights\[radial_weight_index\] = scalar_t\(weight_gradient\);")
HELPERS = """
__device__ __forceinline__ float wide_take(const float* p, float& kept) {
    const float2 t = *reinterpret_cast<const float2*>(p);
    kept = t.y;
    return t.x;
}
"""


def matching_brace(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def widen(source):
    start = source.index("void oeq_projected_backward(")
    stop = source.index("void oeq_projected_backward_receiver(")
    function = source[start:stop]
    match = LOOP.search(function)
    body_open = match.end() - 1
    body_close = matching_brace(function, body_open)
    body = function[body_open + 1:body_close]
    loads, stores = LOAD.findall(body), STORE.findall(body)
    first, second, counter = body, body, iter(range(10 ** 6))
    first = LOAD.sub(lambda m: f"= wide_take({m.group(1)} + {m.group(2)}, kept_{next(counter)});", first)
    counter = iter(range(10 ** 6))
    second = LOAD.sub(lambda m: f"= kept_{next(counter)};", second)
    store_counter = iter(range(10 ** 6))
    first = STORE.sub(lambda m: f"kept_store_{next(store_counter)} = scalar_t(weight_gradient);", first)
    store_counter = iter(range(10 ** 6))
    second = STORE.sub(lambda m: (f"*reinterpret_cast<float2*>(dweights + radial_weight_index - 1) = make_float2("
                                  f"kept_store_{next(store_counter)}, scalar_t(weight_gradient));"), second)
    declarations = "".join(f"scalar_t kept_{i};\n" for i in range(len(loads)))
    declarations += "".join(f"scalar_t kept_store_{i};\n" for i in range(len(stores)))
    new_loop = ("for (int pair = lane; pair < CHANNEL_DIM / 2; pair += LOGICAL_GROUP_SIZE) {\n" + declarations
                + "{ const int channel = 2 * pair;\n" + first + "\n}\n"
                + "{ const int channel = 2 * pair + 1;\n" + second + "\n}\n}")
    function = function[:match.start()] + new_loop + function[body_close + 1:]
    kernel_start = source.index('extern "C" __global__')
    out = source[:kernel_start] + HELPERS + source[kernel_start:start] + function + source[stop:]
    return out, len(loads), len(stores)


original_render = runtime.render_kernels


def build(problem, layout, wide):
    def render(*args, **kwargs):
        schedule, kernel = original_render(*args, **kwargs)
        if not wide:
            return schedule, kernel
        source, _, _ = widen(kernel.jit_kernel)
        return schedule, dataclasses.replace(kernel, jit_kernel=source)

    runtime.render_kernels = render
    try:
        return runtime.StreamingKernels(problem, np.float32, layout=layout)
    finally:
        runtime.render_kernels = original_render


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = load(path)
        p = r["problem"]
        problem = TPProblem(p["irreps_in1"], p["irreps_in2"], p["irreps_out"],
                            [(i["i_in1"], i["i_in2"], i["i_out"], i["mode"], i["has_weight"], i["path_weight"] ** 2)
                             for i in p["instructions"]],
                            irrep_normalization="none", path_normalization="none", shared_weights=False,
                            internal_weights=False)
        _, kernel = original_render(problem, layout=p["layout"])
        _, n_loads, n_stores = widen(kernel.jit_kernel)
        kernels = {"base": build(problem, p["layout"], False), "wide": build(problem, p["layout"], True)}
        x, sh, w, s, rcv, rp, dout = (r[k] for k in ("x", "sh", "w", "senders", "receivers", "row_ptr", "dout"))
        grads = {m: streaming_conv_backward(x, sh, w, s, rcv, rp, dout, k.id) for m, k in kernels.items()}
        row = dict(file=path, gpu=torch.cuda.get_device_name(), edges=int(sh.shape[0]), load_sites=n_loads,
                   store_sites=n_stores, dx_vs=rel(grads["wide"][0], grads["base"][0]),
                   dsh_vs=rel(grads["wide"][1], grads["base"][1]), dw_vs=rel(grads["wide"][2], grads["base"][2]))
        times = {m: [] for m in kernels}
        for _ in range(5):
            for m, k in kernels.items():
                times[m].append(graph_time(lambda k=k: streaming_conv_backward(x, sh, w, s, rcv, rp, dout, k.id)))
        row.update({f"{m}_ms": sorted(v)[2] for m, v in times.items()})
        row["ratios"] = [round(b / a, 4) for a, b in zip(times["base"], times["wide"])]
        row["wide_over_base"] = row["wide_ms"] / row["base_ms"]
        print(json.dumps(row), flush=True)
