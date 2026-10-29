"""Shared helpers of the OPT7 kernel benches: operands, layouts and CUDA-graph timing with an L2 flush."""
import statistics

import torch


def irreps_blocks(irreps):
    """[(mul, ir_dim)] of an irreps string like '128x0e+128x1o' (e3nn-free)."""
    out = []
    for term in irreps.replace(" ", "").split("+"):
        mul, ir = term.split("x") if "x" in term else ("1", term)
        out.append((int(mul), 2 * int(ir[:-1]) + 1))
    return out


def to_mul_ir(t, irreps, layout):
    """Rows of t from the record's layout to mul_ir (e3nn's)."""
    if layout == "mul_ir":
        return t.contiguous()
    parts, start = [], 0
    for mul, dim in irreps_blocks(irreps):
        block = t[:, start:start + mul * dim]
        parts.append(block.reshape(t.shape[0], dim, mul).transpose(1, 2).reshape(t.shape[0], mul * dim))
        start += mul * dim
    return torch.cat(parts, dim=1).contiguous()


def load(path, device="cuda", dtype=torch.float32):
    record = torch.load(path)["records"][0]
    out = {}
    for key, value in record.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out


def graph_time(fn, calls=20, blocks=9, flush_bytes=96 << 20):
    """Median ms per call of fn, captured as a CUDA graph of `calls` calls, each after an L2 flush; the flush is
    captured and timed alone and subtracted. Alternates nothing: one function per call of this helper."""
    junk = torch.empty(flush_bytes // 4, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            junk.fill_(1.0)
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs = []
    for body in (lambda: (junk.fill_(1.0), fn()), lambda: junk.fill_(1.0)):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                body()
        graphs.append(g)
    times = {0: [], 1: []}
    for _ in range(blocks):
        for index, g in enumerate(graphs):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()
            stop.record()
            torch.cuda.synchronize()
            times[index].append(start.elapsed_time(stop) / calls)
    return statistics.median(times[0]) - statistics.median(times[1])


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))
