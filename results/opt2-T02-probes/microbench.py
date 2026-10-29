"""OPT2-T02 microbenchmark: the radial hidden stack, forward plus input gradient, compiled modules vs the fused pair.

    python microbench.py [--rows 8192 56320 82688] [--repeat 200] [--json OUT]

Per row count, in one process, alternating A/B blocks: (A) torch.compile of the original modules (LayerNorm,
SiLU, Linear 64x64, LayerNorm, SiLU, Linear 64x64, LayerNorm, SiLU: MH-1's weight network before its last
layer), forward and input gradient, as Inductor builds them in the force call; (B) the same through
``RadialStack`` (fused_radial_mlp). CUDA-event time per forward+backward, median of the blocks. MH-1 has two such
stacks per force call (one per interaction); the saving per force call is twice the difference. Each side's
forward and backward is captured in a CUDA graph after warm-up and the replays are timed, as the engine replays
its force stage (without the graph, Python, Dynamo guard and autograd overhead of ~0.6 ms per call hides the
kernels).
Also the largest relative difference of output and input gradient between A and B. Not library code.
"""

import argparse
import json
import statistics

import fastmace  # noqa: F401  (before torch)

# isort: split
import torch


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[8192, 56320, 82688])
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--json")
    args = p.parse_args()
    from fastmace.kernels.triton.radial_mlp import RadialStack

    modules = stack()
    plain = torch.nn.Sequential(*modules)
    fused = RadialStack(modules)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": {}}
    for rows in args.rows:
        x = torch.randn(rows, 64, device="cuda") * 2.0
        grad = torch.randn(rows, 64, device="cuda")

        def make(model):
            def forward_backward(inputs):
                inputs = inputs.detach().requires_grad_(True)
                h = model(inputs)
                (dx,) = torch.autograd.grad(h, inputs, grad)
                return h, dx
            return torch.compile(forward_backward, fullgraph=False)

        compiled = {"compiled modules": make(plain), "fused_radial_mlp": make(fused)}
        results = {name: side(x) for name, side in compiled.items()}  # compile and warm up
        sides = {}
        for name, side in compiled.items():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    side(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                side(x)
            sides[name] = graph.replay
        h_a, dx_a = (t.detach() for t in results["compiled modules"])
        h_b, dx_b = (t.detach() for t in results["fused_radial_mlp"])
        diff = dict(output=float((h_b - h_a).abs().max() / h_a.abs().max()),
                    input_gradient=float((dx_b - dx_a).abs().max() / dx_a.abs().max()))
        times = {name: [] for name in sides}
        for block in range(args.blocks):
            order = list(sides) if block % 2 == 0 else list(reversed(sides))
            for name in order:
                times[name].append(timed(sides[name], args.repeat))
        medians = {name: statistics.median(values) for name, values in times.items()}
        saving = medians["compiled modules"] - medians["fused_radial_mlp"]
        out["rows"][rows] = dict(median_us=medians, blocks_us=times, saving_us_per_stack=saving,
                                 saving_us_per_force_call_mh1=2 * saving, relative_difference=diff)
        print(rows, {k: round(v, 1) for k, v in medians.items()}, "saving per stack", round(saving, 1), "us", diff)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
