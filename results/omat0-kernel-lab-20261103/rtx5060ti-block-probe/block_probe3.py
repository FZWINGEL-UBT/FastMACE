"""Forms of the one-pointer block read that work without compile-time offsets (the kernel is independent of N)."""
import statistics
import time

here = "/mnt/c/Users/fzwin/AppData/Local/Temp/claude/b76838efd91e3437/0ef98cc0-d449-4a52-8eb3-e80a2316ed81/scratchpad/"
src = open(here + "block_probe2.py", encoding="utf-8").read().split("for key, (kernel, head")[0].replace(
    'split("def placed")[0]', 'split("def placed")[0]')
exec(compile(src, "probe2_head", "exec"))

ANCHOR = "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};"


def delta_runtime(s):
    """Separate allocations: one base pointer, the other groups as a 64-bit element distance read per edge."""
    groups = sorted({int(m[1]) for m in READ.finditer(s)})
    s = READ.sub(lambda m: f"dout0[int64_t(receiver) * {m[2]} + delta{m[1]} + {m[3]} + channel]", s)
    decl = "    const int64_t delta0 = 0;\n" + "".join(
        f"    const int64_t delta{g} = (reinterpret_cast<int64_t>(dout{g}) - reinterpret_cast<int64_t>(dout0))"
        f" / int64_t(sizeof(scalar_t));\n" for g in groups if g)
    return s.replace(ANCHOR, decl + ANCHOR)


def row_runtime(s):
    """As delta_runtime with the whole row offset of each group kept once per edge."""
    groups = {int(m[1]): int(m[2]) for m in READ.finditer(s)}
    s = READ.sub(lambda m: f"dout0[row{m[1]} + {m[3]} + channel]", s)
    decl = "".join(
        f"    const int64_t row{g} = int64_t(receiver) * {w} + (reinterpret_cast<int64_t>(dout{g})"
        f" - reinterpret_cast<int64_t>(dout0)) / int64_t(sizeof(scalar_t));\n" for g, w in groups.items())
    return s.replace(ANCHOR, decl + ANCHOR)


def node_scaled(per_atom):
    """Joined allocation: the offsets as node_count * a compile-time constant."""
    def transform(s):
        s = READ.sub(lambda m: f"dout0[int64_t(receiver) * {m[2]} + base{m[1]} + {m[3]} + channel]", s)
        decl = "".join(f"    const int64_t base{g} = node_count * {c};\n" for g, c in enumerate(per_atom))
        return s.replace(ANCHOR, decl + ANCHOR)
    return transform


def assume_aligned(s):
    """Four pointers, each assumed 512-byte aligned."""
    groups = sorted({int(m[1]) for m in READ.finditer(s)})
    s = READ.sub(lambda m: f"al{m[1]}[int64_t(receiver) * {m[2]} + {m[3]} + channel]", s)
    decl = "".join(f"    const scalar_t* al{g} = static_cast<const scalar_t*>(__builtin_assume_aligned(dout{g}, 512));\n"
                   for g in groups)
    return s.replace(ANCHOR, decl + ANCHOR)


def ldcs(s):
    """Four pointers, the gradient read with __ldcs (no cache)."""
    return READ.sub(lambda m: f"__ldcs(dout{m[1]} + int64_t(receiver) * {m[2]} + {m[3]} + channel)", s)


def ldg(s):
    """Four pointers, the gradient read with __ldg (read-only path)."""
    return READ.sub(lambda m: f"__ldg(dout{m[1]} + int64_t(receiver) * {m[2]} + {m[3]} + channel)", s)


for key, (kernel, head, groups, tail, floats, dr) in captured.items():
    owner = owners[key]
    owner.kernels_device = groups[0].device
    dense_kernel = wb.WPrimeBackward(owner.kernels, owner.terms, groups[0].device, vector=owner.vector, table_w=True)
    dense = wb.dense_message(groups, kernel.blocks, kernel.channels, kernel.output_dim)
    dx, dsh, r, pair, gprime, starts, scales, bins = tail
    sizes = [g.numel() for g in groups]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    n = head[0].shape[0]
    per_atom = [o // n for o in offsets]
    joined = torch.cat([g.reshape(-1) for g in groups])
    joined_groups = [joined[o:o + s].view(g.shape) for o, s, g in zip(offsets, sizes, groups)]
    variants = {"dense": (dense_kernel, dense), "block": (kernel, groups),
                "1ptr-static": (one_pointer(offsets), joined_groups),
                "delta-runtime": (delta_runtime, groups),
                "row-runtime": (row_runtime, groups),
                "delta-joined": (delta_runtime, joined_groups),
                "node-scaled": (node_scaled(per_atom), joined_groups),
                "aligned": (assume_aligned, groups),
                "ldg": (ldg, groups)}
    sides = {}
    for name_, (t, msg) in variants.items():
        try:
            sides[name_] = (t if isinstance(t, wb.WPrimeBackward) else build(owner, t), msg)
        except Exception as e:
            print(name_, "failed", str(e)[:200], flush=True)
    times = {n_: [] for n_ in sides}
    for round_ in range(45):
        for name_, (ker, msg) in sides.items():
            dx.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ker(*head, msg, dx, dsh, r, pair, gprime, starts, scales, bins, *floats, dr)
            torch.cuda.synchronize()
            if round_ >= 5:
                times[name_].append((time.perf_counter() - t0) * 1e3)
    outs = {}
    for name_, (ker, msg) in sides.items():
        dx.zero_()
        ker(*head, msg, dx, dsh, r, pair, gprime, starts, scales, bins, *floats, dr)
        torch.cuda.synchronize()
        outs[name_] = (dx.clone(), dsh.clone(), dr.clone())
    ref = statistics.median(times["dense"])
    for name_, v in times.items():
        med = statistics.median(v)
        err = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(outs[name_], outs["dense"]))
        print(f"{name_:14s} {med:7.3f} ms  /dense {med / ref:.3f}  spread {min(v):.3f}-{max(v):.3f}  rel err {err:.2e}",
              flush=True)
print("done", flush=True)
