"""fm-ideas: numerical rank of the per-edge weights w(r) = [h c, c] W^T over all species pairs
(MH-1, float64 chain on the CPU). If the rank is well below 65 at float32 / float64 tolerance, the folded
layer's inner dimension K could shrink (a certified low-rank basis)."""
import copy
import sys

import fastmace as fm  # noqa: F401  (before torch)
import torch

sys.path.insert(0, "/mnt/d/Margraf/FastMACE/tools")
import ab_step_time  # noqa: E402
from fastmace.rewrites import mh1, radial, registry as reg  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

torch.set_default_dtype(torch.float64)
spec = sys.argv[1]
name, atoms = ab_step_time.read_structure(spec, None)
calc = MACECalculator(model_paths="/home/fzwin/models/mace-mh-1.model", device="cpu", default_dtype="float64",
                      head="matpes_r2scan")
model = calc.models[0]
batch = calc._atoms_to_batch(atoms).to_dict()
regs = reg.Registry()
for rec in mh1.RECORDS.values():
    regs.register(rec)
asked = {n: ("on" if n in ("factor_species", "fold_cutoff") else "off") for n in regs.names()}
out = copy.deepcopy(model)
# Only the structure's own species: without species_columns the superset is every element (89^2 pairs, 97 GB).
present = sorted(set(batch["node_attrs"].argmax(1).tolist()))
ctx = reg.Context(inputs=reg.structure_inputs(batch, species_columns=present))
regs.apply(out, regs.resolve(out, asked, strict=False), ctx, original=model)
cols = radial._superset(out, ctx)
print(spec, "columns", cols, flush=True)
pairs = [(s, t) for s in range(len(cols)) for t in range(len(cols))]
for idx in range(len(out.interactions)):
    ch = radial._Chain(out, out.interactions[idx], cols, torch.float64, torch.device("cpu"))
    W = ch.folded
    for lo in (0.3, 1.0):
        r = torch.linspace(lo, float(out.r_max) * 0.999, 3000)
        with torch.no_grad():
            A = torch.cat([ch(r, s, t)[:, :-1] for s, t in pairs])
        if A.shape[1] != W.shape[1]:
            print("shape mismatch", A.shape, W.shape)
            break
        w = A @ W.T
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        s = S / S[0]
        scale = w.abs().amax(0).clamp_min(1e-300)
        line = []
        for k in (8, 16, 24, 32, 40, 48, 56, 64):
            wk = (U[:, :k] * S[:k]) @ Vh[:k]
            line.append(f"k{k}:{float(((w - wk).abs() / scale).max()):.1e}")
        print(f"int{idx} r>={lo} W{tuple(W.shape)} A{tuple(A.shape)} rank(1e-7)={int((s > 1e-7).sum())} "
              f"rank(1e-9)={int((s > 1e-9).sum())} rank(1e-13)={int((s > 1e-13).sum())}", *line, flush=True)
