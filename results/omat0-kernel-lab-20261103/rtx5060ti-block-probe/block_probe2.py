"""Layout or generated code? The block kernel's gradient reads, varied alone on tensors from a real step."""
import re
import statistics
import time

here = "/mnt/c/Users/fzwin/AppData/Local/Temp/claude/b76838efd91e3437/0ef98cc0-d449-4a52-8eb3-e80a2316ed81/scratchpad/"
src = open(here + "block_probe.py", encoding="utf-8").read().split("def placed")[0]
exec(compile(src, "block_probe_head", "exec"))
from fastmace.kernels.oeq_streaming import fused_forward as ff  # noqa: E402

READ = re.compile(r"dout(\d+)\[int64_t\(receiver\) \* (\d+)\s*\+ (\d+) \+ channel\]")
real_render, real_map = wb.render, ff.block_map


def build(owner, transform=None, map_=None):
    def render(*a, **k):
        s = real_render(*a, **k)
        return transform(s) if transform else s
    wb.render = render
    if map_:
        ff.block_map = map_
        if hasattr(wb, "block_map"):
            wb.block_map = map_
    try:
        return wb.WPrimeBackward(owner.kernels, owner.terms, owner.kernels_device, vector=owner.vector, table_w=True,
                                 blocks=owner.block[1])
    finally:
        wb.render = real_render
        ff.block_map = real_map
        if hasattr(wb, "block_map"):
            wb.block_map = real_map


def one_pointer(offsets):
    """Block layout, one base pointer: ``dout0[offset_g + ...]`` in a joined allocation."""
    return lambda s: READ.sub(lambda m: f"dout0[int64_t(receiver) * {m[2]} + {offsets[int(m[1])] + int(m[3])} + channel]", s)


def hoisted(s):
    """Block layout, each group's row start computed once per edge."""
    widths = {int(m[1]): int(m[2]) for m in READ.finditer(s)}
    s = READ.sub(lambda m: f"row{m[1]}[{m[3]} + channel]", s)
    decl = "".join(f"    const scalar_t* __restrict__ row{g} = dout{g} + int64_t(receiver) * {w};\n"
                   for g, w in widths.items())
    return s.replace("    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};", decl + "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};")


def int_index(s):
    """Block layout, a 32-bit element index per group, computed once per edge."""
    widths = {int(m[1]): int(m[2]) for m in READ.finditer(s)}
    s = READ.sub(lambda m: f"dout{m[1]}[row{m[1]} + {m[3]} + channel]", s)
    decl = "".join(f"    const int32_t row{g} = receiver * {w};\n" for g, w in widths.items())
    return s.replace("    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};", decl + "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};")


for key, (kernel, head, groups, tail, floats, dr) in captured.items():
    owner = owners[key]
    owner.kernels_device = groups[0].device
    dense_kernel = wb.WPrimeBackward(owner.kernels, owner.terms, groups[0].device, vector=owner.vector, table_w=True)
    dense = wb.dense_message(groups, kernel.blocks, kernel.channels, kernel.output_dim)
    dx, dsh, r, pair, gprime, starts, scales, bins = tail
    sizes = [g.numel() for g in groups]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    n = groups[0].shape[0] // kernel.blocks[0][0]
    offsets_per_atom = None
    joined = torch.cat([g.reshape(-1) for g in groups])
    joined_groups = [joined[o:o + s].view(g.shape) for o, s, g in zip(offsets, sizes, groups)]

    def dense_map(blocks, channels):
        return {int(st): (g, kernel.output_dim, int(st), channels) for g, (d, ss) in enumerate(blocks) for st in ss}

    sides = {"dense": (dense_kernel, dense),
             "block": (kernel, groups),
             "4ptr-denselayout": (build(owner, map_=dense_map), [dense] * len(groups)),
             "1ptr-blocklayout": (build(owner, one_pointer(offsets)), joined_groups),
             "hoisted": (build(owner, hoisted), groups),
             "int-index": (build(owner, int_index), groups)}
    times = {n: [] for n in sides}
    outs = {}
    for round_ in range(45):
        for name_, (ker, msg) in sides.items():
            dx.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ker(*head, msg, dx, dsh, r, pair, gprime, starts, scales, bins, *floats, dr)
            torch.cuda.synchronize()
            if round_ >= 5:
                times[name_].append((time.perf_counter() - t0) * 1e3)
        if round_ == 0:
            pass
    # the variants must give the dense kernel's answer
    for name_, (ker, msg) in sides.items():
        dx.zero_()
        ker(*head, msg, dx, dsh, r, pair, gprime, starts, scales, bins, *floats, dr)
        torch.cuda.synchronize()
        outs[name_] = (dx.clone(), dsh.clone(), dr.clone())
    ref = statistics.median(times["dense"])
    for name_, v in times.items():
        med = statistics.median(v)
        err = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(outs[name_], outs["dense"]))
        print(f"{name_:18s} {med:7.3f} ms  /dense {med / ref:.3f}  spread {min(v):.3f}-{max(v):.3f}  rel err {err:.2e}",
              flush=True)
print("done", flush=True)
