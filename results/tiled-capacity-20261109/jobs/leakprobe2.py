"""leakprobe2.py ARGS: tiled_capacity.py with the device's allocated memory sampled every 200 tiles, and after each
call a census of the live CUDA tensors by (shape, dtype) with a few referrers of the largest groups."""
import collections, gc, json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tools", "bench"))
import fastmace  # noqa
import torch
from fastmace.model import tiled
import tiled_capacity
samples, count = [], [0]
orig = tiled.TiledForceField._eager_tile
def wrapped(self, sweep, *a, **k):
    count[0] += 1
    if count[0] % 200 == 0:
        samples.append((count[0], sweep, round(torch.cuda.memory_allocated() / 2**30, 3)))
    return orig(self, sweep, *a, **k)
tiled.TiledForceField._eager_tile = wrapped

def owner(obj, depth=0):
    out = []
    for ref in gc.get_referrers(obj)[:6]:
        if ref is globals() or isinstance(ref, list) and ref is _objs:
            continue
        name = type(ref).__name__
        if isinstance(ref, dict):
            keys = [k for k, v in ref.items() if v is obj][:2]
            mod = ref.get("__name__") if "__name__" in ref else None
            name += f" keys={keys!r}"[:120] + (f" module={mod}" if mod else "")
            if depth < 2:
                name += " <- [" + "; ".join(owner(ref, depth + 1))[:400] + "]"
        elif depth < 2 and isinstance(ref, (tuple, list)):
            name += f" len={len(ref)} <- [" + "; ".join(owner(ref, depth + 1))[:400] + "]"
        out.append(name)
    return out

_objs = []
def census(tag):
    global _objs
    torch.cuda.synchronize()
    _objs = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
    groups, seen = collections.defaultdict(list), set()
    for t in _objs:
        try:
            if not t.is_cuda:
                continue
            s = t.untyped_storage()
            if s.data_ptr() in seen:
                continue
            seen.add(s.data_ptr())
            groups[(str(t.dtype), len(t.shape), t.shape[1:] if t.dim() > 1 else ())].append((s.nbytes(), t))
        except Exception:
            pass
    total = sum(b for g in groups.values() for b, _ in g)
    print(f"census {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, python-reachable {total / 2**30:.2f} GiB in {sum(len(g) for g in groups.values())} storages", flush=True)
    rows = sorted(groups.items(), key=lambda kv: -sum(b for b, _ in kv[1]))[:12]
    for key, g in rows:
        print(f"  {sum(b for b, _ in g) / 2**30:8.3f} GiB  x{len(g):6d}  {key}", flush=True)
    for key, g in rows[:4]:
        biggest = max(g, key=lambda bt: bt[0])[1]
        print(f"  owners of {key}: {owner(biggest)}", flush=True)
    _objs = []

orig_call = tiled.TiledForceField._call
def call(self, *a, **k):
    try:
        return orig_call(self, *a, **k)
    finally:
        census("after call")
tiled.TiledForceField._call = call
try:
    tiled_capacity.main(sys.argv[1:])
finally:
    for s in samples[::max(1, len(samples) // 25)]:
        print("probe", s, flush=True)
