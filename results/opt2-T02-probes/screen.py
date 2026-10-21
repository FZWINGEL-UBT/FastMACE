"""OPT2-T02 screen (a)+(b): store-vs-recompute and rows/warps configurations of the fused radial hidden stack.

    python screen.py [--rows 8192 56320 82688] [--repeat 200] [--blocks 6] [--json OUT]

As ``microbench.py`` (one MH-1 hidden stack, D = 64, K = 2, forward plus input gradient, each side captured in a
CUDA graph, median of alternating blocks of replays), with the fused side in every configuration of
CONFIGS = (rows per program, warps, store). For each configuration it also records the registers and spills of
its two Triton kernels (compiled with warmup()), which the shape rule must respect: no spills.
Not library code.
"""

import argparse
import json
import statistics

import fastmace  # noqa: F401  (before torch)

# isort: split
import torch

CONFIGS = [(32, 4, False), (64, 8, False), (16, 4, True), (32, 4, True), (32, 8, True), (64, 4, True),
           (64, 8, True), (128, 8, True)]


def stack(d=64, k=2, seed=0):
    torch.manual_seed(seed)
    modules = [torch.nn.LayerNorm(d), torch.nn.SiLU()]
    for _ in range(k):
        modules += [torch.nn.Linear(d, d), torch.nn.LayerNorm(d), torch.nn.SiLU()]
    with torch.no_grad():
        for module in modules:
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return [m.cuda() for m in modules]


def timed(step, repeat):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat * 1e3  # microseconds


def kernel_resources(radial_mlp, modules, config, rows=4096):
    """Registers and spills of the forward and backward kernels of ``config``, compiled with Triton's warmup()."""
    block_rows, num_warps, store = config
    stack_module = radial_mlp.RadialStack(modules, config=config)
    gamma, beta, weight, bias, eps = stack_module.operands()
    k, d = weight.shape[0], weight.shape[1]
    x = torch.randn(rows, d, device="cuda")
    out, grad, dx = torch.empty_like(x), torch.randn_like(x), torch.empty_like(x)
    saved = x.new_empty((k, rows, d) if store else (0,))
    kernels = radial_mlp._kernels()
    common = dict(BE=block_rows, D=d, K=k, STORE=store, num_warps=num_warps, grid=(1,))
    wt = weight.transpose(1, 2).contiguous()
    forward = kernels["forward"].warmup(x, x.stride(0), out, saved, gamma, beta, wt, bias, rows, eps, **common)
    backward = kernels["backward"].warmup(x, x.stride(0), saved, grad, dx, gamma, beta, wt, weight.contiguous(),
                                          bias, rows, eps, **common)
    for compiled in (forward, backward):
        if hasattr(compiled, "_init_handles"):
            compiled._init_handles()  # Triton fills n_regs and n_spills when it loads the binary
    return {name: dict(n_regs=int(getattr(c, "n_regs", -1)), n_spills=int(getattr(c, "n_spills", -1)))
            for name, c in (("forward", forward), ("backward", backward))}


def graphed(side, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            side(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        side(x)
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[8192, 56320, 82688])
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--json")
    args = p.parse_args()
    from fastmace.kernels.triton import radial_mlp

    # Every side is a torch.compile of the same inner function (one code object): without a higher limit,
    # 8 models x 3 row counts exceed Dynamo's per-code recompile limit and the later sides run eagerly
    # (the first screen's compiled-modules baseline at 56,320 and 82,688 rows was eager and invalid).
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 4096

    modules = stack()
    models = {"compiled modules": torch.nn.Sequential(*modules)}
    for config in CONFIGS:
        models[f"rows{config[0]}-warps{config[1]}-{'store' if config[2] else 'recompute'}"] = \
            radial_mlp.RadialStack(modules, config=config)
    out = {"gpu": torch.cuda.get_device_name(0), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "torch": torch.__version__, "configs": {}, "rows": {}}
    for config in CONFIGS:
        name = f"rows{config[0]}-warps{config[1]}-{'store' if config[2] else 'recompute'}"
        out["configs"][name] = kernel_resources(radial_mlp, modules, config)
        print(name, out["configs"][name])
    for rows in args.rows:
        x = torch.randn(rows, 64, device="cuda") * 2.0
        grad = torch.randn(rows, 64, device="cuda")

        def make(model):
            def forward_backward(inputs):
                inputs = inputs.detach().requires_grad_(True)
                h = model(inputs)
                (dx,) = torch.autograd.grad(h, inputs, grad)
                return h, dx
            return torch.compile(forward_backward, fullgraph=False, dynamic=False)

        graphs, results = {}, {}
        for name, model in models.items():
            side = make(model)
            results[name] = tuple(t.detach() for t in side(x))
            graphs[name] = graphed(side, x)
        reference = results["compiled modules"]
        times = {name: [] for name in graphs}
        names = list(graphs)
        for block in range(args.blocks):
            for name in (names if block % 2 == 0 else list(reversed(names))):
                times[name].append(timed(graphs[name].replay, args.repeat))
        medians = {name: statistics.median(values) for name, values in times.items()}
        base = medians["compiled modules"]
        entry = out["rows"][rows] = {}
        for name in names:
            h, dx = results[name]
            entry[name] = dict(median_us=medians[name], saving_us_per_stack=base - medians[name],
                               saving_us_per_force_call_mh1=2 * (base - medians[name]), blocks_us=times[name],
                               output_rel=float((h - reference[0]).abs().max() / reference[0].abs().max()),
                               input_gradient_rel=float((dx - reference[1]).abs().max() / reference[1].abs().max()))
        recompiles = torch._dynamo.utils.counters["stats"].get("unique_graphs", None)
        print(rows, {name: round(value, 1) for name, value in medians.items()}, "dynamo unique graphs", recompiles)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
